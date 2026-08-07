from __future__ import annotations

import json
import logging
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from scipy.stats import (
    spearmanr,
    wilcoxon,
    mannwhitneyu,
)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe"
)

MODELS_ROOT = PROJECT_ROOT / "models"

SENSITIVITY_ROOT = (
    PROJECT_ROOT / "sensitivity_v3"
)

HARDATA_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "hardata"
)

REVIEWER_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
)

OUTPUT_ROOT = (
    REVIEWER_OUTPUT_ROOT
    / "reviewer2_remaining_analyses"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

ATTRIBUTION_DIR = (
    OUTPUT_ROOT
    / "attribution_maps"
)

ATTRIBUTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    OUTPUT_ROOT
    / "reviewer2_remaining.log"
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | %(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TARGET_CLASS = 1

N_TNBC_PATIENTS = 5
N_NON_TNBC_PATIENTS = 5

IG_STEPS = 32
SMOOTHGRAD_SAMPLES = 32
SMOOTHGRAD_NOISE_STD = 0.10

OCCLUSION_PATCH = 32
OCCLUSION_STRIDE = 32

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42


IMAGE_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGE_STD = [
    0.229,
    0.224,
    0.225,
]


EXPLAINABILITY_FILE = (
    OUTPUT_ROOT
    / "explainability_comparison.csv"
)

EXPLAINABILITY_SUMMARY_FILE = (
    OUTPUT_ROOT
    / "explainability_summary.csv"
)

FUSION_FILE = (
    OUTPUT_ROOT
    / "alternative_fusion_baselines.csv"
)

FUSION_PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "alternative_fusion_predictions.csv"
)

POWER_FILE = (
    OUTPUT_ROOT
    / "sample_size_power_analysis.csv"
)

MULTIPLE_TEST_FILE = (
    OUTPUT_ROOT
    / "multiple_testing_global_audit.csv"
)

REPRODUCIBILITY_FILE = (
    OUTPUT_ROOT
    / "reproducibility_audit.txt"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "reviewer2_remaining_summary.json"
)


def bh_adjust(
    pvalues: list[float],
) -> np.ndarray:

    values = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        return np.array([])

    order = np.argsort(values)
    ranked = values[order]

    adjusted = np.empty(
        n,
        dtype=float,
    )

    running = 1.0

    for i in range(
        n - 1,
        -1,
        -1,
    ):
        rank = i + 1

        candidate = (
            ranked[i]
            * n
            / rank
        )

        running = min(
            running,
            candidate,
        )

        adjusted[
            order[i]
        ] = min(
            running,
            1.0,
        )

    return adjusted


def normalize_map(
    attribution: np.ndarray,
) -> np.ndarray:

    attribution = np.asarray(
        attribution,
        dtype=np.float64,
    )

    attribution = np.nan_to_num(
        attribution
    )

    minimum = attribution.min()
    maximum = attribution.max()

    if maximum <= minimum:
        return np.zeros_like(
            attribution
        )

    return (
        attribution - minimum
    ) / (
        maximum - minimum
    )


def load_checkpoint(
    path: Path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(
        checkpoint,
        dict,
    ):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
        ]:
            if key in checkpoint:
                checkpoint = (
                    checkpoint[key]
                )
                break

    clean = {}

    for key, value in (
        checkpoint.items()
    ):

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = (
                new_key[7:]
            )

        clean[
            new_key
        ] = value

    return clean


def locate_resnet_ts_weight() -> Path:

    candidates = [
        MODELS_ROOT
        / "resnet50_bh_best.pth",

        MODELS_ROOT
        / "resnet50_best.pth",

        MODELS_ROOT
        / "resnet50_bh_last.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(
                "Using ResNet weight: %s",
                candidate,
            )
            return candidate

    raise FileNotFoundError(
        "Could not locate the ResNet50 TS "
        "checkpoint in the models folder."
    )


def build_resnet_model() -> nn.Module:

    weight_path = (
        locate_resnet_ts_weight()
    )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    state = load_checkpoint(
        weight_path
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        ),
    ]
)


def detect_column(
    frame: pd.DataFrame,
    candidates: list[str],
):

    lookup = {
        str(column).lower():
            column
        for column
        in frame.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    return None


def load_evaluation_tiles():

    test_file = (
        MODELS_ROOT
        / "test_df.csv"
    )

    if not test_file.exists():
        raise FileNotFoundError(
            test_file
        )

    frame = pd.read_csv(
        test_file
    )

    path_column = detect_column(
        frame,
        [
            "path",
            "filepath",
            "file_path",
        ],
    )

    label_column = detect_column(
        frame,
        [
            "label",
            "tnbc",
        ],
    )

    patient_column = detect_column(
        frame,
        [
            "sample_id",
            "Sample ID",
            "patient_id",
            "patient",
        ],
    )

    if path_column is None:
        raise ValueError(
            "No image path column found "
            "in test_df.csv."
        )

    if label_column is None:
        raise ValueError(
            "No label column found "
            "in test_df.csv."
        )

    if patient_column is None:

        frame[
            "_patient_id"
        ] = (
            frame[
                path_column
            ]
            .astype(str)
            .str.extract(
                r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})",
                expand=False,
            )
        )

        patient_column = (
            "_patient_id"
        )

    frame = frame[
        [
            path_column,
            label_column,
            patient_column,
        ]
    ].copy()

    frame.columns = [
        "path",
        "label",
        "patient_id",
    ]

    frame[
        "label"
    ] = pd.to_numeric(
        frame[
            "label"
        ],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "path",
            "label",
            "patient_id",
        ]
    )

    frame[
        "label"
    ] = (
        frame[
            "label"
        ].astype(int)
    )

    frame[
        "exists"
    ] = (
        frame[
            "path"
        ]
        .astype(str)
        .apply(
            lambda value:
                Path(value).exists()
        )
    )

    frame = frame[
        frame[
            "exists"
        ]
    ].copy()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected = []

    for label, count in [
        (
            1,
            N_TNBC_PATIENTS,
        ),
        (
            0,
            N_NON_TNBC_PATIENTS,
        ),
    ]:

        subset = frame[
            frame[
                "label"
            ] == label
        ]

        patients = (
            subset[
                "patient_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        if len(patients) == 0:
            continue

        count = min(
            count,
            len(patients),
        )

        patients = rng.choice(
            patients,
            size=count,
            replace=False,
        )

        for patient in patients:

            patient_rows = subset[
                subset[
                    "patient_id"
                ] == patient
            ]

            row = patient_rows.iloc[
                0
            ]

            selected.append(
                {
                    "patient_id":
                        patient,
                    "label":
                        int(label),
                    "path":
                        row[
                            "path"
                        ],
                }
            )

    selected_frame = pd.DataFrame(
        selected
    )

    logger.info(
        "Explainability patients selected: %d",
        len(
            selected_frame
        ),
    )

    return selected_frame


def load_image_tensor(
    path: str,
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(
        0
    )

    return (
        image,
        tensor.to(
            DEVICE
        ),
    )


def fgsm_sensitivity(
    model,
    x,
):

    x = (
        x.clone()
        .detach()
        .requires_grad_(
            True
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        x
    )

    target = torch.tensor(
        [
            TARGET_CLASS
        ],
        device=DEVICE,
    )

    loss = F.cross_entropy(
        logits,
        target,
    )

    loss.backward()

    gradient = (
        x.grad
        .detach()
        .abs()
        .mean(
            dim=1
        )[0]
        .cpu()
        .numpy()
    )

    return normalize_map(
        gradient
    )


def integrated_gradients(
    model,
    x,
    steps=IG_STEPS,
):

    baseline = torch.zeros_like(
        x
    )

    total_gradient = (
        torch.zeros_like(
            x
        )
    )

    for alpha in np.linspace(
        0.0,
        1.0,
        steps,
        endpoint=True,
    ):

        interpolated = (
            baseline
            + float(alpha)
            * (
                x - baseline
            )
        )

        interpolated = (
            interpolated.detach()
            .requires_grad_(
                True
            )
        )

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            interpolated
        )

        score = logits[
            0,
            TARGET_CLASS,
        ]

        score.backward()

        total_gradient += (
            interpolated.grad
            .detach()
        )

    average_gradient = (
        total_gradient
        / float(
            steps
        )
    )

    attribution = (
        (
            x - baseline
        )
        * average_gradient
    )

    attribution = (
        attribution
        .detach()
        .abs()
        .mean(
            dim=1
        )[0]
        .cpu()
        .numpy()
    )

    return normalize_map(
        attribution
    )


def smoothgrad(
    model,
    x,
):

    rng = torch.Generator(
        device=DEVICE
    )

    rng.manual_seed(
        RANDOM_SEED
    )

    total = torch.zeros_like(
        x
    )

    for _ in range(
        SMOOTHGRAD_SAMPLES
    ):

        noise = torch.randn(
            x.shape,
            generator=rng,
            device=DEVICE,
        )

        noisy = (
            x
            + SMOOTHGRAD_NOISE_STD
            * noise
        )

        noisy = (
            noisy.detach()
            .requires_grad_(
                True
            )
        )

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            noisy
        )

        score = logits[
            0,
            TARGET_CLASS,
        ]

        score.backward()

        total += (
            noisy.grad
            .detach()
            .abs()
        )

    attribution = (
        total
        / float(
            SMOOTHGRAD_SAMPLES
        )
    )

    attribution = (
        attribution
        .mean(
            dim=1
        )[0]
        .cpu()
        .numpy()
    )

    return normalize_map(
        attribution
    )


def occlusion_sensitivity(
    model,
    x,
):

    with torch.no_grad():

        baseline_probability = (
            F.softmax(
                model(
                    x
                ),
                dim=1,
            )[
                0,
                TARGET_CLASS,
            ].item()
        )

    _, _, height, width = (
        x.shape
    )

    heat = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float64,
    )

    counts = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float64,
    )

    for y in range(
        0,
        height,
        OCCLUSION_STRIDE,
    ):

        for x0 in range(
            0,
            width,
            OCCLUSION_STRIDE,
        ):

            y1 = min(
                y
                + OCCLUSION_PATCH,
                height,
            )

            x1 = min(
                x0
                + OCCLUSION_PATCH,
                width,
            )

            masked = (
                x.clone()
                .detach()
            )

            masked[
                :,
                :,
                y:y1,
                x0:x1,
            ] = 0.0

            with torch.no_grad():

                probability = (
                    F.softmax(
                        model(
                            masked
                        ),
                        dim=1,
                    )[
                        0,
                        TARGET_CLASS,
                    ].item()
                )

            importance = (
                baseline_probability
                - probability
            )

            heat[
                y:y1,
                x0:x1,
            ] += importance

            counts[
                y:y1,
                x0:x1,
            ] += 1.0

    heat = np.divide(
        heat,
        counts,
        out=np.zeros_like(
            heat
        ),
        where=counts > 0,
    )

    heat = np.abs(
        heat
    )

    return normalize_map(
        heat
    )


class GradCAM:

    def __init__(
        self,
        model,
        layer,
    ):

        self.model = model
        self.layer = layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    def forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = (
            output.detach()
        )

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[
                0
            ].detach()
        )

    def __call__(
        self,
        x,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            x
        )

        score = logits[
            0,
            TARGET_CLASS,
        ]

        score.backward()

        weights = (
            self.gradients
            .mean(
                dim=(
                    2,
                    3,
                ),
                keepdim=True,
            )
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                224,
                224,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = (
            cam[
                0,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        return normalize_map(
            cam
        )

    def close(
        self,
    ):

        self.forward_handle.remove()
        self.backward_handle.remove()


def top_fraction_mask(
    array,
    fraction=0.10,
):

    threshold = np.quantile(
        array,
        1.0 - fraction,
    )

    return (
        array >= threshold
    )


def jaccard(
    left,
    right,
):

    left_mask = (
        top_fraction_mask(
            left
        )
    )

    right_mask = (
        top_fraction_mask(
            right
        )
    )

    intersection = np.logical_and(
        left_mask,
        right_mask,
    ).sum()

    union = np.logical_or(
        left_mask,
        right_mask,
    ).sum()

    if union == 0:
        return np.nan

    return float(
        intersection
        / union
    )


def compare_maps(
    reference,
    candidate,
):

    rho, _ = spearmanr(
        reference.flatten(),
        candidate.flatten(),
    )

    overlap = jaccard(
        reference,
        candidate,
    )

    return (
        float(rho),
        float(overlap),
    )


def save_numpy_map(
    patient_id,
    method,
    array,
):

    safe_patient = (
        str(
            patient_id
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    output = (
        ATTRIBUTION_DIR
        / (
            f"{safe_patient}_"
            f"{method}.npy"
        )
    )

    np.save(
        output,
        array,
    )


def bootstrap_mean_ci(
    values,
    iterations=BOOTSTRAP_ITERATIONS,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    estimates = np.empty(
        iterations
    )

    for i in range(
        iterations
    ):

        sample = rng.choice(
            values,
            size=len(
                values
            ),
            replace=True,
        )

        estimates[i] = (
            np.mean(
                sample
            )
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.percentile(
                estimates,
                2.5,
            )
        ),
        float(
            np.percentile(
                estimates,
                97.5,
            )
        ),
    )


def explainability_analysis():

    logger.info(
        "Starting explainability baseline comparison."
    )

    model = build_resnet_model()

    selected = (
        load_evaluation_tiles()
    )

    gradcam = GradCAM(
        model,
        model.layer4[
            -1
        ],
    )

    rows = []

    methods = [
        "IntegratedGradients",
        "SmoothGrad",
        "Occlusion",
        "GradCAM",
    ]

    try:

        for index, row in (
            selected.iterrows()
        ):

            patient_id = (
                row[
                    "patient_id"
                ]
            )

            image_path = (
                row[
                    "path"
                ]
            )

            label = int(
                row[
                    "label"
                ]
            )

            logger.info(
                "Explainability %s",
                patient_id,
            )

            _, tensor = (
                load_image_tensor(
                    image_path
                )
            )

            fgsm = fgsm_sensitivity(
                model,
                tensor,
            )

            maps = {
                "IntegratedGradients":
                    integrated_gradients(
                        model,
                        tensor,
                    ),

                "SmoothGrad":
                    smoothgrad(
                        model,
                        tensor,
                    ),

                "Occlusion":
                    occlusion_sensitivity(
                        model,
                        tensor,
                    ),

                "GradCAM":
                    gradcam(
                        tensor
                    ),
            }

            save_numpy_map(
                patient_id,
                "FGSM",
                fgsm,
            )

            for method, attribution in (
                maps.items()
            ):

                save_numpy_map(
                    patient_id,
                    method,
                    attribution,
                )

                rho, overlap = (
                    compare_maps(
                        fgsm,
                        attribution,
                    )
                )

                rows.append(
                    {
                        "patient_id":
                            patient_id,
                        "true_label":
                            label,
                        "method":
                            method,
                        "fgsm_spearman_rho":
                            rho,
                        "top10_jaccard":
                            overlap,
                        "image_path":
                            image_path,
                    }
                )

    finally:

        gradcam.close()

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        EXPLAINABILITY_FILE,
        index=False,
    )

    summary_rows = []

    for method in methods:

        subset = results[
            results[
                "method"
            ] == method
        ]

        (
            mean_rho,
            rho_low,
            rho_high,
        ) = bootstrap_mean_ci(
            subset[
                "fgsm_spearman_rho"
            ]
        )

        (
            mean_overlap,
            overlap_low,
            overlap_high,
        ) = bootstrap_mean_ci(
            subset[
                "top10_jaccard"
            ]
        )

        correlations = (
            subset[
                "fgsm_spearman_rho"
            ]
            .dropna()
            .values
        )

        if len(
            correlations
        ) >= 3:

            try:

                _, pvalue = wilcoxon(
                    correlations
                )

            except Exception:

                pvalue = np.nan

        else:

            pvalue = np.nan

        summary_rows.append(
            {
                "method":
                    method,
                "patients":
                    len(
                        subset
                    ),
                "mean_spearman_rho":
                    mean_rho,
                "rho_ci95_lower":
                    rho_low,
                "rho_ci95_upper":
                    rho_high,
                "mean_top10_jaccard":
                    mean_overlap,
                "jaccard_ci95_lower":
                    overlap_low,
                "jaccard_ci95_upper":
                    overlap_high,
                "wilcoxon_rho_vs_zero_p":
                    pvalue,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    valid = summary[
        "wilcoxon_rho_vs_zero_p"
    ].notna()

    summary.loc[
        valid,
        "bh_adjusted_p",
    ] = bh_adjust(
        summary.loc[
            valid,
            "wilcoxon_rho_vs_zero_p",
        ].tolist()
    )

    summary.to_csv(
        EXPLAINABILITY_SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "Explainability comparison"
    )
    print(
        "-" * 60
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return (
        results,
        summary,
    )


def load_map(
    path: Path,
):

    return np.load(
        path,
        allow_pickle=True,
    ).item()


def load_gene_fusion_inputs():

    gene_scaler_path = (
        HARDATA_ROOT
        / "gene_scaler.pkl"
    )

    top_gene_names_path = (
        HARDATA_ROOT
        / "top_gene_names.npy"
    )

    gene_matched_path = (
        HARDATA_ROOT
        / "gene_matched.csv"
    )

    clinical_path = (
        HARDATA_ROOT
        / "clinical_a2_matched.csv"
    )

    import joblib

    gene_scaler = joblib.load(
        gene_scaler_path
    )

    top_gene_names = np.load(
        top_gene_names_path,
        allow_pickle=True,
    )

    top_gene_names = [
        (
            value.decode(
                "utf-8"
            )
            if isinstance(
                value,
                bytes,
            )
            else str(
                value
            )
        )
        for value
        in top_gene_names
    ]

    gene_df = pd.read_csv(
        gene_matched_path,
        index_col=0,
    )

    clinical = pd.read_csv(
        clinical_path
    )

    clinical_indexed = (
        clinical.set_index(
            "Sample ID"
        )
    )

    common = (
        gene_df.columns
        .intersection(
            clinical_indexed.index
        )
    )

    gene_data = (
        gene_df[
            common
        ].T
    )

    gene_data_top = (
        gene_data[
            top_gene_names
        ]
    )

    return (
        gene_scaler,
        gene_data_top,
    )


def build_combined_features(
    img_maps_tnbc,
    img_maps_non_tnbc,
    pgd_maps_tnbc,
    pgd_maps_non_tnbc,
    gene_data_top,
    gene_scaler,
):

    X = []
    y = []
    pids = []

    for pid in img_maps_tnbc:

        if (
            pid
            not in gene_data_top.index
        ):
            continue

        if (
            pid
            not in pgd_maps_tnbc
        ):
            continue

        fgsm = (
            img_maps_tnbc[
                pid
            ].flatten()
        )

        pgd = (
            pgd_maps_tnbc[
                pid
            ].flatten()
        )

        image_features = [
            fgsm.mean(),
            fgsm.max(),
            fgsm.std(),
            np.percentile(
                fgsm,
                75,
            ),
            np.percentile(
                fgsm,
                90,
            ),
            pgd.mean(),
            pgd.max(),
        ]

        gene_values = (
            gene_scaler.transform(
                gene_data_top.loc[
                    [
                        pid
                    ]
                ].values
            )[0]
        )

        gene_features = [
            gene_values.mean(),
            gene_values.max(),
            gene_values.std(),
            np.percentile(
                gene_values,
                75,
            ),
            np.percentile(
                gene_values,
                90,
            ),
        ]

        X.append(
            image_features
            + gene_features
        )

        y.append(
            1
        )

        pids.append(
            pid
        )

    for pid in img_maps_non_tnbc:

        if (
            pid
            not in gene_data_top.index
        ):
            continue

        if (
            pid
            not in pgd_maps_non_tnbc
        ):
            continue

        fgsm = (
            img_maps_non_tnbc[
                pid
            ].flatten()
        )

        pgd = (
            pgd_maps_non_tnbc[
                pid
            ].flatten()
        )

        image_features = [
            fgsm.mean(),
            fgsm.max(),
            fgsm.std(),
            np.percentile(
                fgsm,
                75,
            ),
            np.percentile(
                fgsm,
                90,
            ),
            pgd.mean(),
            pgd.max(),
        ]

        gene_values = (
            gene_scaler.transform(
                gene_data_top.loc[
                    [
                        pid
                    ]
                ].values
            )[0]
        )

        gene_features = [
            gene_values.mean(),
            gene_values.max(),
            gene_values.std(),
            np.percentile(
                gene_values,
                75,
            ),
            np.percentile(
                gene_values,
                90,
            ),
        ]

        X.append(
            image_features
            + gene_features
        )

        y.append(
            0
        )

        pids.append(
            pid
        )

    return (
        np.asarray(
            X,
            dtype=float,
        ),
        np.asarray(
            y,
            dtype=int,
        ),
        pids,
    )


def bootstrap_auc(
    y_true,
    probabilities,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    positive_indices = np.where(
        y_true == 1
    )[0]

    negative_indices = np.where(
        y_true == 0
    )[0]

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    aucs = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        positives = rng.choice(
            positive_indices,
            size=len(
                positive_indices
            ),
            replace=True,
        )

        negatives = rng.choice(
            negative_indices,
            size=len(
                negative_indices
            ),
            replace=True,
        )

        indices = np.concatenate(
            [
                positives,
                negatives,
            ]
        )

        aucs[
            iteration
        ] = roc_auc_score(
            y_true[
                indices
            ],
            probabilities[
                indices
            ],
        )

    return (
        float(
            np.percentile(
                aucs,
                2.5,
            )
        ),
        float(
            np.percentile(
                aucs,
                97.5,
            )
        ),
    )


def loo_model_predictions(
    X,
    y,
    model_name,
):

    loo = LeaveOneOut()

    probabilities = []
    truths = []
    indices = []

    for train_index, test_index in (
        loo.split(
            X
        )
    ):

        if model_name == "LogisticRegression":

            classifier = Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            )

        elif model_name == "MLP":

            classifier = Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(
                                16,
                                8,
                            ),
                            activation="relu",
                            solver="adam",
                            alpha=0.01,
                            learning_rate_init=0.001,
                            max_iter=1000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            )

        else:

            raise ValueError(
                model_name
            )

        classifier.fit(
            X[
                train_index
            ],
            y[
                train_index
            ],
        )

        probability = (
            classifier.predict_proba(
                X[
                    test_index
                ]
            )[
                0,
                1,
            ]
        )

        probabilities.append(
            probability
        )

        truths.append(
            y[
                test_index
            ][0]
        )

        indices.append(
            int(
                test_index[
                    0
                ]
            )
        )

    return (
        np.asarray(
            truths
        ),
        np.asarray(
            probabilities
        ),
        indices,
    )


def alternative_fusion_analysis():

    logger.info(
        "Running alternative fusion baselines."
    )

    (
        gene_scaler,
        gene_data_top,
    ) = load_gene_fusion_inputs()

    fgsm_resnet_tnbc = load_map(
        SENSITIVITY_ROOT
        / "fgsm_resnet_tnbc_v3.npy"
    )

    fgsm_resnet_non = load_map(
        SENSITIVITY_ROOT
        / "fgsm_resnet_non_tnbc_v3.npy"
    )

    pgd_resnet_tnbc = load_map(
        SENSITIVITY_ROOT
        / "pgd_resnet_tnbc_v3.npy"
    )

    pgd_resnet_non = load_map(
        SENSITIVITY_ROOT
        / "pgd_resnet_non_tnbc_v3.npy"
    )

    fgsm_eff_tnbc = load_map(
        SENSITIVITY_ROOT
        / "fgsm_eff_tnbc_v3.npy"
    )

    pgd_eff_tnbc = load_map(
        SENSITIVITY_ROOT
        / "pgd_eff_tnbc_v3.npy"
    )

    matrices = {}

    matrices[
        "ResNet50"
    ] = build_combined_features(
        fgsm_resnet_tnbc,
        fgsm_resnet_non,
        pgd_resnet_tnbc,
        pgd_resnet_non,
        gene_data_top,
        gene_scaler,
    )

    matrices[
        "EfficientNet"
    ] = build_combined_features(
        fgsm_eff_tnbc,
        fgsm_resnet_non,
        pgd_eff_tnbc,
        pgd_resnet_non,
        gene_data_top,
        gene_scaler,
    )

    result_rows = []
    prediction_rows = []

    for image_model, (
        X,
        y,
        pids,
    ) in matrices.items():

        for fusion_model in [
            "LogisticRegression",
            "MLP",
        ]:

            (
                truths,
                probabilities,
                indices,
            ) = loo_model_predictions(
                X,
                y,
                fusion_model,
            )

            auc = roc_auc_score(
                truths,
                probabilities,
            )

            ci_lower, ci_upper = (
                bootstrap_auc(
                    truths,
                    probabilities,
                )
            )

            result_rows.append(
                {
                    "image_features":
                        image_model,
                    "fusion_model":
                        fusion_model,
                    "loo_auc":
                        auc,
                    "ci95_lower":
                        ci_lower,
                    "ci95_upper":
                        ci_upper,
                    "patients":
                        len(
                            truths
                        ),
                    "tnbc":
                        int(
                            truths.sum()
                        ),
                }
            )

            for truth, probability, index in zip(
                truths,
                probabilities,
                indices,
            ):

                prediction_rows.append(
                    {
                        "image_features":
                            image_model,
                        "fusion_model":
                            fusion_model,
                        "patient_id":
                            pids[
                                index
                            ],
                        "true_label":
                            int(
                                truth
                            ),
                        "loo_probability":
                            float(
                                probability
                            ),
                    }
                )

    result_frame = pd.DataFrame(
        result_rows
    )

    prediction_frame = pd.DataFrame(
        prediction_rows
    )

    result_frame.to_csv(
        FUSION_FILE,
        index=False,
    )

    prediction_frame.to_csv(
        FUSION_PREDICTIONS_FILE,
        index=False,
    )

    print()
    print(
        "Alternative fusion baselines"
    )
    print(
        "-" * 60
    )

    print(
        result_frame.to_string(
            index=False
        )
    )

    return result_frame


def power_analysis():

    logger.info(
        "Running sample size and power analysis."
    )

    input_file = (
        SENSITIVITY_ROOT
        / "image_sensitivity_summary_v3.csv"
    )

    frame = pd.read_csv(
        input_file
    )

    if "label" not in frame.columns:
        raise ValueError(
            "label column missing from "
            "image_sensitivity_summary_v3.csv"
        )

    if "fgsm_mean" not in frame.columns:
        raise ValueError(
            "fgsm_mean missing from "
            "image_sensitivity_summary_v3.csv"
        )

    positive = (
        frame[
            frame[
                "label"
            ] == 1
        ][
            "fgsm_mean"
        ]
        .dropna()
        .values
    )

    negative = (
        frame[
            frame[
                "label"
            ] == 0
        ][
            "fgsm_mean"
        ]
        .dropna()
        .values
    )

    pooled_variance = (
        (
            (
                len(
                    positive
                ) - 1
            )
            * np.var(
                positive,
                ddof=1,
            )
        )
        + (
            (
                len(
                    negative
                ) - 1
            )
            * np.var(
                negative,
                ddof=1,
            )
        )
    ) / (
        len(
            positive
        )
        + len(
            negative
        )
        - 2
    )

    pooled_sd = math.sqrt(
        pooled_variance
    )

    cohens_d = (
        np.mean(
            positive
        )
        - np.mean(
            negative
        )
    ) / pooled_sd

    mw_result = mannwhitneyu(
        positive,
        negative,
        alternative="two-sided",
    )

    n1 = len(
        positive
    )

    n0 = len(
        negative
    )

    rank_biserial = (
        1.0
        - (
            2.0
            * mw_result.statistic
            / (
                n1
                * n0
            )
        )
    )

    achieved_power = np.nan
    detectable_d = np.nan

    try:

        from statsmodels.stats.power import (
            TTestIndPower
        )

        analysis = (
            TTestIndPower()
        )

        ratio = (
            n0
            / n1
        )

        achieved_power = (
            analysis.power(
                effect_size=abs(
                    cohens_d
                ),
                nobs1=n1,
                alpha=0.05,
                ratio=ratio,
                alternative="two-sided",
            )
        )

        detectable_d = (
            analysis.solve_power(
                effect_size=None,
                nobs1=n1,
                alpha=0.05,
                power=0.80,
                ratio=ratio,
                alternative="two-sided",
            )
        )

    except Exception as error:

        logger.warning(
            "statsmodels power analysis "
            "unavailable: %s",
            error,
        )

    output = pd.DataFrame(
        [
            {
                "tnbc_n":
                    n1,
                "non_tnbc_n":
                    n0,
                "cohens_d":
                    cohens_d,
                "mann_whitney_u":
                    mw_result.statistic,
                "mann_whitney_p":
                    mw_result.pvalue,
                "rank_biserial_effect":
                    rank_biserial,
                "approx_achieved_power":
                    achieved_power,
                "minimum_detectable_cohens_d_at_80_power":
                    detectable_d,
            }
        ]
    )

    output.to_csv(
        POWER_FILE,
        index=False,
    )

    print()
    print(
        "Sample size and power"
    )
    print(
        "-" * 60
    )

    print(
        output.to_string(
            index=False
        )
    )

    return output


def multiple_testing_audit():

    logger.info(
        "Running multiple-testing audit."
    )

    rows = []

    for csv_path in (
        REVIEWER_OUTPUT_ROOT.rglob(
            "*.csv"
        )
    ):

        if OUTPUT_ROOT in (
            csv_path.parents
        ):
            continue

        try:

            frame = pd.read_csv(
                csv_path
            )

        except Exception:

            continue

        p_columns = []

        for column in (
            frame.columns
        ):

            normalized = (
                str(
                    column
                )
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            if (
                normalized
                in {
                    "p",
                    "p_value",
                    "pvalue",
                    "wilcoxon_p",
                    "p_value_raw",
                }
                or normalized.endswith(
                    "_p"
                )
                or normalized.endswith(
                    "_p_value"
                )
            ):

                if (
                    "adjusted"
                    not in normalized
                    and "bh_"
                    not in normalized
                ):

                    p_columns.append(
                        column
                    )

        for column in p_columns:

            values = pd.to_numeric(
                frame[
                    column
                ],
                errors="coerce",
            )

            for row_index, value in (
                values.items()
            ):

                if (
                    pd.isna(
                        value
                    )
                    or value < 0
                    or value > 1
                ):
                    continue

                rows.append(
                    {
                        "source_file":
                            str(
                                csv_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),
                        "row_index":
                            int(
                                row_index
                            ),
                        "p_column":
                            str(
                                column
                            ),
                        "raw_p":
                            float(
                                value
                            ),
                    }
                )

    audit = pd.DataFrame(
        rows
    )

    if not audit.empty:

        audit[
            "global_bh_adjusted_p"
        ] = bh_adjust(
            audit[
                "raw_p"
            ].tolist()
        )

    audit.to_csv(
        MULTIPLE_TEST_FILE,
        index=False,
    )

    logger.info(
        "Multiple-testing audit contains %d tests.",
        len(
            audit
        ),
    )

    return audit


def reproducibility_audit():

    logger.info(
        "Running reproducibility audit."
    )

    files = [
        PROJECT_ROOT
        / "final-code"
        / "train_models.py",

        PROJECT_ROOT
        / "final-code"
        / "compute_sensitivity_v3.py",

        PROJECT_ROOT
        / "final-code"
        / "compute_sensitivity_e2.py",

        PROJECT_ROOT
        / "train_models.py",
    ]

    keywords = [
        "optimizer",
        "adam",
        "sgd",
        "learning_rate",
        "lr=",
        "scheduler",
        "batch_size",
        "batch size",
        "augmentation",
        "random",
        "seed",
        "patience",
        "early stopping",
        "early_stopping",
        "epochs",
        "cuda",
        "gpu",
        "weightedrandomsampler",
        "focal",
        "epsilon",
        "eps",
        "alpha",
        "pgd",
        "fgsm",
    ]

    lines = []

    for path in files:

        if not path.exists():
            continue

        lines.append(
            "=" * 80
        )

        lines.append(
            str(
                path
            )
        )

        lines.append(
            "=" * 80
        )

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            lower = line.lower()

            if any(
                keyword
                in lower
                for keyword
                in keywords
            ):

                lines.append(
                    f"{line_number}: {line}"
                )

        lines.append(
            ""
        )

    REPRODUCIBILITY_FILE.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    logger.info(
        "Reproducibility audit saved."
    )

    return len(
        lines
    )


def main():

    print()
    print(
        "Reviewer 2 remaining analyses"
    )
    print(
        "=" * 60
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            + torch.cuda.get_device_name(
                0
            )
        )

        total_memory = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024 ** 3
        )

        print(
            f"GPU memory: "
            f"{total_memory:.2f} GB"
        )

    explainability_results = None
    explainability_summary = None

    try:

        (
            explainability_results,
            explainability_summary,
        ) = explainability_analysis()

    except Exception as error:

        logger.exception(
            "Explainability analysis failed: %s",
            error,
        )

    fusion_results = None

    try:

        fusion_results = (
            alternative_fusion_analysis()
        )

    except Exception as error:

        logger.exception(
            "Fusion baseline analysis failed: %s",
            error,
        )

    power_results = None

    try:

        power_results = (
            power_analysis()
        )

    except Exception as error:

        logger.exception(
            "Power analysis failed: %s",
            error,
        )

    multiple_testing_results = None

    try:

        multiple_testing_results = (
            multiple_testing_audit()
        )

    except Exception as error:

        logger.exception(
            "Multiple-testing audit failed: %s",
            error,
        )

    reproducibility_lines = None

    try:

        reproducibility_lines = (
            reproducibility_audit()
        )

    except Exception as error:

        logger.exception(
            "Reproducibility audit failed: %s",
            error,
        )

    summary = {
        "device":
            str(
                DEVICE
            ),

        "explainability": {
            "completed":
                explainability_summary
                is not None,

            "methods": [
                "FGSM",
                "Integrated Gradients",
                "SmoothGrad",
                "Occlusion Sensitivity",
                "Grad-CAM",
            ],
        },

        "alternative_fusion": {
            "completed":
                fusion_results
                is not None,

            "models": [
                "Logistic Regression",
                "MLP",
            ],
        },

        "power_analysis_completed":
            power_results
            is not None,

        "multiple_testing_audit_completed":
            multiple_testing_results
            is not None,

        "reproducibility_audit_completed":
            reproducibility_lines
            is not None,

        "not_computationally_addressed": [
            "wet-lab validation",
            "independent pathologist validation",
            "clinical workflow integration",
            "prospective deployment evaluation",
        ],
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "Saved outputs"
    )
    print(
        "-" * 60
    )

    print(
        f"Explainability: "
        f"{EXPLAINABILITY_FILE}"
    )

    print(
        f"Explainability summary: "
        f"{EXPLAINABILITY_SUMMARY_FILE}"
    )

    print(
        f"Fusion baselines: "
        f"{FUSION_FILE}"
    )

    print(
        f"Power analysis: "
        f"{POWER_FILE}"
    )

    print(
        f"Multiple testing: "
        f"{MULTIPLE_TEST_FILE}"
    )

    print(
        f"Reproducibility audit: "
        f"{REPRODUCIBILITY_FILE}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Log: "
        f"{LOG_FILE}"
    )


if __name__ == "__main__":
    main()