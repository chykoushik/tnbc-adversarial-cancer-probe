from __future__ import annotations

import gc
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TILE_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_tiles\tiles"
)

TILE_MANIFEST = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_tiles\cptac_tile_manifest.csv"
)

MODELS_DIR = Path(
    r"E:\apply\journal publication\onco-probe\models"
)

RESNET_WEIGHTS = MODELS_DIR / "resnet50_bh_best.pth"
EFFICIENTNET_WEIGHTS = MODELS_DIR / "efficientnet_bh_best.pth"

OUTPUT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_adversarial_validation"
)

MAP_ROOT = OUTPUT_ROOT / "patient_maps"
LOG_FILE = OUTPUT_ROOT / "cptac_adversarial_validation.log"

TILE_RESULTS_CSV = OUTPUT_ROOT / "cptac_tile_results.csv"
PATIENT_RESULTS_CSV = OUTPUT_ROOT / "cptac_patient_results.csv"
MODEL_AGREEMENT_CSV = OUTPUT_ROOT / "cptac_model_agreement.csv"
RUN_METADATA_JSON = OUTPUT_ROOT / "run_metadata.json"

IMAGE_SIZE = 224
TILES_PER_PATIENT = 100

BATCH_SIZE = 8
NUM_WORKERS = 0

TARGET_CLASS = 1

PGD_EPSILON = 0.03
PGD_ALPHA = 0.007
PGD_STEPS = 10

RANDOM_SEED = 2026

# Set to True to rerun completed patients.
OVERWRITE_PATIENT_RESULTS = False


# ---------------------------------------------------------------------
# Reproducibility and logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MAP_ROOT.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class PatientTileDataset(Dataset):
    def __init__(
        self,
        patient_id: str,
        tile_paths: list[Path],
    ) -> None:
        self.patient_id = patient_id
        self.tile_paths = tile_paths

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, str]:
        tile_path = self.tile_paths[index]

        with Image.open(tile_path) as image:
            image = image.convert("RGB")
            tensor = IMAGE_TRANSFORM(image)

        return tensor, str(tile_path)


def load_patient_tiles() -> dict[str, list[Path]]:
    if not TILE_ROOT.exists():
        raise FileNotFoundError(
            f"Tile directory does not exist: {TILE_ROOT}"
        )

    patients: dict[str, list[Path]] = {}

    for patient_directory in sorted(TILE_ROOT.iterdir()):
        if not patient_directory.is_dir():
            continue

        tile_paths = sorted(patient_directory.glob("*.jpg"))

        if not tile_paths:
            continue

        patient_id = patient_directory.name

        patients[patient_id] = tile_paths[:TILES_PER_PATIENT]

    if not patients:
        raise RuntimeError(
            f"No patient tiles were found in {TILE_ROOT}"
        )

    return patients


# ---------------------------------------------------------------------
# Model construction and checkpoint loading
# ---------------------------------------------------------------------

def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        clean_key = key

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]

        cleaned[clean_key] = value

    return cleaned


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for candidate_key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]:
            candidate = checkpoint.get(candidate_key)

            if isinstance(candidate, dict):
                return clean_state_dict(candidate)

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return clean_state_dict(checkpoint)

    raise RuntimeError(
        "Could not identify a model state dictionary in checkpoint."
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            f"Missing checkpoint parameters for "
            f"{checkpoint_path.name}: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint parameters for "
            f"{checkpoint_path.name}: {unexpected}"
        )

    model = model.to(device)
    model.eval()

    return model


def build_resnet(
    device: torch.device,
) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return load_checkpoint(
        model=model,
        checkpoint_path=RESNET_WEIGHTS,
        device=device,
    )


def build_efficientnet(
    device: torch.device,
) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2,
    )

    return load_checkpoint(
        model=model,
        checkpoint_path=EFFICIENTNET_WEIGHTS,
        device=device,
    )


# ---------------------------------------------------------------------
# Adversarial sensitivity
# ---------------------------------------------------------------------

def predict_and_fgsm(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    images = images.to(
        device,
        non_blocking=True,
    )

    images.requires_grad_(True)

    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)

    targets = torch.full(
        size=(images.shape[0],),
        fill_value=TARGET_CLASS,
        dtype=torch.long,
        device=device,
    )

    loss = nn.CrossEntropyLoss()(logits, targets)

    model.zero_grad(set_to_none=True)

    if images.grad is not None:
        images.grad.zero_()

    loss.backward()

    gradients = images.grad.detach().abs()
    fgsm_maps = gradients.mean(dim=1)

    tnbc_probabilities = probabilities[:, TARGET_CLASS]
    predicted_classes = probabilities.argmax(dim=1)

    return (
        tnbc_probabilities.detach().cpu().numpy(),
        predicted_classes.detach().cpu().numpy(),
        fgsm_maps.detach().cpu().numpy(),
    )


def compute_pgd_sensitivity(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    epsilon: float = PGD_EPSILON,
    alpha: float = PGD_ALPHA,
    steps: int = PGD_STEPS,
) -> np.ndarray:
    model.eval()

    original = images.to(
        device,
        non_blocking=True,
    ).detach()

    perturbed = original.clone().detach()

    accumulator = torch.zeros(
        (
            original.shape[0],
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.float32,
        device="cpu",
    )

    targets = torch.full(
        size=(original.shape[0],),
        fill_value=TARGET_CLASS,
        dtype=torch.long,
        device=device,
    )

    for _ in range(steps):
        perturbed.requires_grad_(True)

        logits = model(perturbed)
        loss = nn.CrossEntropyLoss()(logits, targets)

        model.zero_grad(set_to_none=True)

        if perturbed.grad is not None:
            perturbed.grad.zero_()

        loss.backward()

        gradient = perturbed.grad.detach()

        batch_maps = (
            gradient
            .abs()
            .mean(dim=1)
            .detach()
            .cpu()
        )

        accumulator += batch_maps

        perturbed = (
            perturbed
            + alpha * gradient.sign()
        )

        delta = torch.clamp(
            perturbed - original,
            min=-epsilon,
            max=epsilon,
        )

        # This reproduces the original repository's PGD implementation.
        perturbed = torch.clamp(
            original + delta,
            min=0.0,
            max=1.0,
        ).detach()

    return (
        accumulator
        .div(float(steps))
        .numpy()
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def summarize_map(
    sensitivity_map: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    flat = sensitivity_map.astype(
        np.float64
    ).ravel()

    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_p75": float(np.percentile(flat, 75)),
        f"{prefix}_p90": float(np.percentile(flat, 90)),
    }


def safe_spearman(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first_flat = np.asarray(first).ravel()
    second_flat = np.asarray(second).ravel()

    if (
        np.std(first_flat) == 0
        or np.std(second_flat) == 0
    ):
        return float("nan"), float("nan")

    result = spearmanr(
        first_flat,
        second_flat,
    )

    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------
# Existing output handling
# ---------------------------------------------------------------------

def load_existing_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if TILE_RESULTS_CSV.exists():
        tile_results = pd.read_csv(TILE_RESULTS_CSV)
    else:
        tile_results = pd.DataFrame()

    if PATIENT_RESULTS_CSV.exists():
        patient_results = pd.read_csv(PATIENT_RESULTS_CSV)
    else:
        patient_results = pd.DataFrame()

    return tile_results, patient_results


def completed_patients(
    patient_results: pd.DataFrame,
) -> set[str]:
    if patient_results.empty:
        return set()

    if "patient_id" not in patient_results.columns:
        return set()

    return set(
        patient_results["patient_id"]
        .astype(str)
        .tolist()
    )


def update_csv(
    path: Path,
    new_rows: list[dict[str, Any]],
    duplicate_columns: list[str],
) -> None:
    if not new_rows:
        return

    new_dataframe = pd.DataFrame(new_rows)

    if path.exists():
        old_dataframe = pd.read_csv(path)

        combined = pd.concat(
            [old_dataframe, new_dataframe],
            ignore_index=True,
        )
    else:
        combined = new_dataframe

    available_duplicate_columns = [
        column
        for column in duplicate_columns
        if column in combined.columns
    ]

    if available_duplicate_columns:
        combined = combined.drop_duplicates(
            subset=available_duplicate_columns,
            keep="last",
        )

    combined.to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------------------
# Patient processing
# ---------------------------------------------------------------------

def process_model_for_patient(
    model_name: str,
    model: nn.Module,
    patient_id: str,
    tile_paths: list[Path],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    dataset = PatientTileDataset(
        patient_id=patient_id,
        tile_paths=tile_paths,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    tile_rows: list[dict[str, Any]] = []

    fgsm_maps: list[np.ndarray] = []
    pgd_maps: list[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"{patient_id} {model_name}",
        leave=False,
    )

    tile_counter = 0

    for image_batch, path_batch in progress:
        (
            probabilities,
            predictions,
            batch_fgsm_maps,
        ) = predict_and_fgsm(
            model=model,
            images=image_batch,
            device=device,
        )

        batch_pgd_maps = compute_pgd_sensitivity(
            model=model,
            images=image_batch,
            device=device,
        )

        for batch_index, tile_path in enumerate(path_batch):
            tile_counter += 1

            fgsm_map = batch_fgsm_maps[batch_index]
            pgd_map = batch_pgd_maps[batch_index]

            fgsm_maps.append(fgsm_map)
            pgd_maps.append(pgd_map)

            tile_row: dict[str, Any] = {
                "patient_id": patient_id,
                "model": model_name,
                "tile_number": tile_counter,
                "tile_path": str(tile_path),
                "tnbc_probability": float(
                    probabilities[batch_index]
                ),
                "predicted_class": int(
                    predictions[batch_index]
                ),
                "true_label": 1,
            }

            tile_row.update(
                summarize_map(
                    sensitivity_map=fgsm_map,
                    prefix="fgsm",
                )
            )

            tile_row.update(
                summarize_map(
                    sensitivity_map=pgd_map,
                    prefix="pgd",
                )
            )

            tile_rows.append(tile_row)

        del image_batch
        del batch_fgsm_maps
        del batch_pgd_maps

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fgsm_maps:
        raise RuntimeError(
            f"No valid FGSM maps generated for {patient_id}, "
            f"{model_name}."
        )

    patient_fgsm_map = np.mean(
        np.stack(fgsm_maps, axis=0),
        axis=0,
    )

    patient_pgd_map = np.mean(
        np.stack(pgd_maps, axis=0),
        axis=0,
    )

    return (
        tile_rows,
        patient_fgsm_map,
        patient_pgd_map,
    )


def build_patient_row(
    patient_id: str,
    model_name: str,
    tile_rows: list[dict[str, Any]],
    fgsm_map: np.ndarray,
    pgd_map: np.ndarray,
) -> dict[str, Any]:
    tile_dataframe = pd.DataFrame(tile_rows)

    fgsm_pgd_rho, fgsm_pgd_p = safe_spearman(
        fgsm_map,
        pgd_map,
    )

    row: dict[str, Any] = {
        "patient_id": patient_id,
        "model": model_name,
        "label": 1,
        "tiles_processed": int(len(tile_dataframe)),
        "tnbc_probability_mean": float(
            tile_dataframe["tnbc_probability"].mean()
        ),
        "tnbc_probability_median": float(
            tile_dataframe["tnbc_probability"].median()
        ),
        "tnbc_probability_std": float(
            tile_dataframe["tnbc_probability"].std(ddof=0)
        ),
        "tnbc_probability_min": float(
            tile_dataframe["tnbc_probability"].min()
        ),
        "tnbc_probability_max": float(
            tile_dataframe["tnbc_probability"].max()
        ),
        "tnbc_tile_fraction_at_0_5": float(
            (
                tile_dataframe["tnbc_probability"]
                >= 0.5
            ).mean()
        ),
        "fgsm_pgd_spearman_rho": fgsm_pgd_rho,
        "fgsm_pgd_spearman_p": fgsm_pgd_p,
    }

    row.update(
        summarize_map(
            sensitivity_map=fgsm_map,
            prefix="fgsm",
        )
    )

    row.update(
        summarize_map(
            sensitivity_map=pgd_map,
            prefix="pgd",
        )
    )

    return row


def process_patient(
    patient_id: str,
    tile_paths: list[Path],
    resnet: nn.Module,
    efficientnet: nn.Module,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_tile_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []

    model_definitions = [
        ("resnet50_ts", resnet),
        ("efficientnet_b0_ts", efficientnet),
    ]

    patient_maps: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model in model_definitions:
        (
            tile_rows,
            fgsm_map,
            pgd_map,
        ) = process_model_for_patient(
            model_name=model_name,
            model=model,
            patient_id=patient_id,
            tile_paths=tile_paths,
            device=device,
        )

        patient_maps[model_name] = {
            "fgsm": fgsm_map,
            "pgd": pgd_map,
        }

        patient_map_directory = (
            MAP_ROOT
            / model_name
        )

        patient_map_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            patient_map_directory
            / f"{patient_id}_fgsm.npy",
            fgsm_map,
        )

        np.save(
            patient_map_directory
            / f"{patient_id}_pgd.npy",
            pgd_map,
        )

        patient_row = build_patient_row(
            patient_id=patient_id,
            model_name=model_name,
            tile_rows=tile_rows,
            fgsm_map=fgsm_map,
            pgd_map=pgd_map,
        )

        all_tile_rows.extend(tile_rows)
        patient_rows.append(patient_row)

    resnet_fgsm = patient_maps["resnet50_ts"]["fgsm"]
    efficientnet_fgsm = patient_maps[
        "efficientnet_b0_ts"
    ]["fgsm"]

    resnet_pgd = patient_maps["resnet50_ts"]["pgd"]
    efficientnet_pgd = patient_maps[
        "efficientnet_b0_ts"
    ]["pgd"]

    fgsm_rho, fgsm_p = safe_spearman(
        resnet_fgsm,
        efficientnet_fgsm,
    )

    pgd_rho, pgd_p = safe_spearman(
        resnet_pgd,
        efficientnet_pgd,
    )

    agreement_row = {
        "patient_id": patient_id,
        "fgsm_resnet_efficientnet_spearman_rho": fgsm_rho,
        "fgsm_resnet_efficientnet_spearman_p": fgsm_p,
        "pgd_resnet_efficientnet_spearman_rho": pgd_rho,
        "pgd_resnet_efficientnet_spearman_p": pgd_p,
    }

    update_csv(
        path=MODEL_AGREEMENT_CSV,
        new_rows=[agreement_row],
        duplicate_columns=["patient_id"],
    )

    return all_tile_rows, patient_rows


# ---------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------

def save_run_metadata(
    device: torch.device,
    patient_count: int,
) -> None:
    metadata: dict[str, Any] = {
        "dataset": "CPTAC-BRCA",
        "cohort_definition": "TNBC-positive patients",
        "patient_count": patient_count,
        "tiles_per_patient": TILES_PER_PATIENT,
        "target_class": TARGET_CLASS,
        "image_size": IMAGE_SIZE,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "pgd_epsilon": PGD_EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "resnet_weights": str(RESNET_WEIGHTS),
        "efficientnet_weights": str(EFFICIENTNET_WEIGHTS),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }

    if torch.cuda.is_available():
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024 ** 3),
                    2,
                ),
            }
        )

    with RUN_METADATA_JSON.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            metadata,
            file_handle,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    configure_logging()
    set_reproducibility(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print(
            "GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    else:
        logging.warning(
            "CUDA is unavailable. The script will run on CPU."
        )

    patient_tiles = load_patient_tiles()

    print(
        f"Patients found: {len(patient_tiles)}"
    )

    invalid_patients = {
        patient_id: len(tile_paths)
        for patient_id, tile_paths in patient_tiles.items()
        if len(tile_paths) != TILES_PER_PATIENT
    }

    if invalid_patients:
        raise RuntimeError(
            "Every patient must have exactly "
            f"{TILES_PER_PATIENT} tiles. "
            f"Invalid counts: {invalid_patients}"
        )

    save_run_metadata(
        device=device,
        patient_count=len(patient_tiles),
    )

    logging.info("Loading ResNet50-TS.")
    resnet = build_resnet(device)

    logging.info("Loading EfficientNet-B0-TS.")
    efficientnet = build_efficientnet(device)

    _, existing_patient_results = load_existing_results()

    done = completed_patients(
        existing_patient_results
    )

    if OVERWRITE_PATIENT_RESULTS:
        done = set()

    for patient_index, patient_id in enumerate(
        sorted(patient_tiles),
        start=1,
    ):
        if patient_id in done:
            logging.info(
                "%s already completed. Skipping.",
                patient_id,
            )
            continue

        logging.info(
            "Processing patient %s (%d/%d).",
            patient_id,
            patient_index,
            len(patient_tiles),
        )

        try:
            tile_rows, patient_rows = process_patient(
                patient_id=patient_id,
                tile_paths=patient_tiles[patient_id],
                resnet=resnet,
                efficientnet=efficientnet,
                device=device,
            )

            update_csv(
                path=TILE_RESULTS_CSV,
                new_rows=tile_rows,
                duplicate_columns=[
                    "patient_id",
                    "model",
                    "tile_path",
                ],
            )

            update_csv(
                path=PATIENT_RESULTS_CSV,
                new_rows=patient_rows,
                duplicate_columns=[
                    "patient_id",
                    "model",
                ],
            )

            logging.info(
                "%s completed successfully.",
                patient_id,
            )

        except Exception as error:
            logging.exception(
                "%s failed: %s",
                patient_id,
                error,
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patient_results = pd.read_csv(
        PATIENT_RESULTS_CSV
    )

    agreement_results = pd.read_csv(
        MODEL_AGREEMENT_CSV
    )

    print()
    print("CPTAC adversarial validation finished")
    print("----------------------------------------")
    print(
        f"Patients expected:          {len(patient_tiles)}"
    )
    print(
        "Patient-model rows:        "
        f"{len(patient_results)}"
    )
    print(
        "Patients with agreement:   "
        f"{agreement_results['patient_id'].nunique()}"
    )
    print(
        f"Patient results:           {PATIENT_RESULTS_CSV}"
    )
    print(
        f"Tile results:              {TILE_RESULTS_CSV}"
    )
    print(
        f"Model agreement:           {MODEL_AGREEMENT_CSV}"
    )
    print(
        f"Patient sensitivity maps:  {MAP_ROOT}"
    )
    print(
        f"Run metadata:              {RUN_METADATA_JSON}"
    )
    print(
        f"Log file:                  {LOG_FILE}"
    )


if __name__ == "__main__":
    main()