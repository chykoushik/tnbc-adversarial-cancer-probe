from __future__ import annotations

import logging
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from wsidicom import WsiDicom


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INVENTORY_CSV = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_inventory\cptac_brca_dicom_inventory.csv"
)

OUTPUT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_tiles"
)

TILE_ROOT = OUTPUT_ROOT / "tiles"
QC_ROOT = OUTPUT_ROOT / "qc"
MANIFEST_CSV = OUTPUT_ROOT / "cptac_tile_manifest.csv"
PATIENT_SUMMARY_CSV = OUTPUT_ROOT / "cptac_tile_summary.csv"
LOG_FILE = OUTPUT_ROOT / "cptac_tile_extraction.log"

TILE_SIZE = 224
TILES_PER_PATIENT = 100

MAX_CANDIDATE_ATTEMPTS = 3000
THUMBNAIL_MAX_SIZE = 2048

MIN_SATURATION = 18
MAX_BRIGHTNESS = 235
MIN_TISSUE_FRACTION = 0.55

MAX_WHITE_FRACTION = 0.40
MAX_DARK_FRACTION = 0.25
MIN_RGB_STD = 10.0

RANDOM_SEED = 2026

# Keep False so completed patients are skipped when rerunning.
OVERWRITE_EXISTING_PATIENTS = False

# This series was selected previously for 01BR040 but its thumbnail was blank.
EXCLUDED_SERIES_UIDS = {
    "1.3.6.1.4.1.5962.99.1.149692220.266066100.1640827232060.2.0",
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TILE_ROOT.mkdir(parents=True, exist_ok=True)
    QC_ROOT.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------
# Inventory and series selection
# ---------------------------------------------------------------------

def select_one_series_per_patient(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "patient_id",
        "series_instance_uid",
        "series_description",
        "file_size_mb",
        "absolute_path",
    }

    missing = required_columns.difference(inventory.columns)

    if missing:
        raise ValueError(
            f"Inventory CSV is missing columns: {sorted(missing)}"
        )

    inventory = inventory.copy()

    inventory["patient_id"] = (
        inventory["patient_id"]
        .astype(str)
        .str.strip()
    )

    inventory["series_instance_uid"] = (
        inventory["series_instance_uid"]
        .astype(str)
        .str.strip()
    )

    inventory["series_description"] = (
        inventory["series_description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    inventory["file_size_mb"] = pd.to_numeric(
        inventory["file_size_mb"],
        errors="coerce",
    ).fillna(0.0)

    # Prefer "HE tumor_tissue", but permit "HE" when tumor_tissue
    # is unavailable. This allows 18BR002 to be included.
    candidate_rows = inventory[
        inventory["series_description"]
        .str.lower()
        .isin({"he tumor_tissue", "he"})
    ].copy()

    candidate_rows = candidate_rows[
        ~candidate_rows["series_instance_uid"].isin(
            EXCLUDED_SERIES_UIDS
        )
    ].copy()

    if candidate_rows.empty:
        raise RuntimeError("No usable HE series were found.")

    candidate_rows["series_priority"] = np.where(
        candidate_rows["series_description"].str.lower()
        == "he tumor_tissue",
        0,
        1,
    )

    series_summary = (
        candidate_rows.groupby(
            [
                "patient_id",
                "series_instance_uid",
                "series_description",
                "series_priority",
            ],
            as_index=False,
        )
        .agg(
            total_size_mb=("file_size_mb", "sum"),
            first_file=("absolute_path", "first"),
            dicom_file_count=("absolute_path", "count"),
        )
    )

    selected = (
        series_summary.sort_values(
            [
                "patient_id",
                "series_priority",
                "total_size_mb",
            ],
            ascending=[True, True, False],
        )
        .drop_duplicates("patient_id", keep="first")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    selected["series_directory"] = selected[
        "first_file"
    ].apply(
        lambda value: str(Path(str(value)).parent)
    )

    return selected


# ---------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------

def create_tissue_mask(
    thumbnail: Image.Image,
) -> np.ndarray:
    rgb = np.asarray(
        thumbnail.convert("RGB")
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    mask = (
        (saturation >= MIN_SATURATION)
        & (brightness <= MAX_BRIGHTNESS)
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3,
    )

    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(mask)

    minimum_component_area = max(
        50,
        int(mask.size * 0.0002),
    )

    for component_id in range(
        1,
        component_count,
    ):
        area = stats[
            component_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= minimum_component_area:
            cleaned[
                labels == component_id
            ] = 255

    return cleaned


def save_tissue_mask_qc(
    patient_id: str,
    thumbnail: Image.Image,
    mask: np.ndarray,
) -> None:
    thumbnail_path = (
        QC_ROOT
        / f"{patient_id}_thumbnail.jpg"
    )

    mask_path = (
        QC_ROOT
        / f"{patient_id}_tissue_mask.png"
    )

    thumbnail.convert("RGB").save(
        thumbnail_path,
        quality=90,
    )

    Image.fromarray(mask).save(mask_path)


# ---------------------------------------------------------------------
# Tile quality control
# ---------------------------------------------------------------------

def calculate_tile_metrics(
    tile: Image.Image,
) -> dict[str, float]:
    rgb = np.asarray(
        tile.convert("RGB")
    ).astype(np.uint8)

    grayscale = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    white_pixels = (
        (rgb[:, :, 0] >= 235)
        & (rgb[:, :, 1] >= 235)
        & (rgb[:, :, 2] >= 235)
    )

    dark_pixels = grayscale <= 25

    tissue_pixels = (
        (hsv[:, :, 1] >= MIN_SATURATION)
        & (hsv[:, :, 2] <= MAX_BRIGHTNESS)
    )

    return {
        "white_fraction": float(
            white_pixels.mean()
        ),
        "dark_fraction": float(
            dark_pixels.mean()
        ),
        "tissue_fraction": float(
            tissue_pixels.mean()
        ),
        "rgb_std": float(
            rgb.std()
        ),
        "mean_red": float(
            rgb[:, :, 0].mean()
        ),
        "mean_green": float(
            rgb[:, :, 1].mean()
        ),
        "mean_blue": float(
            rgb[:, :, 2].mean()
        ),
    }


def tile_passes_qc(
    metrics: dict[str, float],
) -> bool:
    return (
        metrics["white_fraction"]
        <= MAX_WHITE_FRACTION
        and metrics["dark_fraction"]
        <= MAX_DARK_FRACTION
        and metrics["tissue_fraction"]
        >= MIN_TISSUE_FRACTION
        and metrics["rgb_std"]
        >= MIN_RGB_STD
    )


# ---------------------------------------------------------------------
# Coordinate sampling
# ---------------------------------------------------------------------

def generate_candidate_locations(
    mask: np.ndarray,
    level0_width: int,
    level0_height: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    mask_height, mask_width = mask.shape

    tissue_y, tissue_x = np.where(
        mask > 0
    )

    if len(tissue_x) == 0:
        return []

    indices = list(
        range(len(tissue_x))
    )

    rng.shuffle(indices)

    scale_x = (
        level0_width / mask_width
    )

    scale_y = (
        level0_height / mask_height
    )

    candidates: list[
        tuple[int, int, int, int]
    ] = []

    used_cells: set[
        tuple[int, int]
    ] = set()

    for index in indices:
        thumbnail_x = int(
            tissue_x[index]
        )

        thumbnail_y = int(
            tissue_y[index]
        )

        coarse_cell = (
            thumbnail_x // 5,
            thumbnail_y // 5,
        )

        if coarse_cell in used_cells:
            continue

        used_cells.add(coarse_cell)

        center_x = int(
            (thumbnail_x + 0.5)
            * scale_x
        )

        center_y = int(
            (thumbnail_y + 0.5)
            * scale_y
        )

        x = (
            center_x
            - TILE_SIZE // 2
        )

        y = (
            center_y
            - TILE_SIZE // 2
        )

        x = max(
            0,
            min(
                x,
                level0_width
                - TILE_SIZE,
            ),
        )

        y = max(
            0,
            min(
                y,
                level0_height
                - TILE_SIZE,
            ),
        )

        candidates.append(
            (
                x,
                y,
                thumbnail_x,
                thumbnail_y,
            )
        )

        if (
            len(candidates)
            >= MAX_CANDIDATE_ATTEMPTS
        ):
            break

    return candidates


# ---------------------------------------------------------------------
# Montage generation
# ---------------------------------------------------------------------

def create_tile_montage(
    patient_id: str,
    tile_paths: list[Path],
) -> None:
    selected_paths = tile_paths[:25]

    if not selected_paths:
        return

    columns = 5

    rows = int(
        np.ceil(
            len(selected_paths)
            / columns
        )
    )

    montage = Image.new(
        "RGB",
        (
            columns * TILE_SIZE,
            rows * TILE_SIZE,
        ),
        "white",
    )

    draw = ImageDraw.Draw(montage)

    for index, tile_path in enumerate(
        selected_paths
    ):
        with Image.open(
            tile_path
        ) as tile_image:
            tile = tile_image.convert(
                "RGB"
            )

            column = (
                index % columns
            )

            row = (
                index // columns
            )

            montage.paste(
                tile,
                (
                    column * TILE_SIZE,
                    row * TILE_SIZE,
                ),
            )

    draw.rectangle(
        (
            0,
            0,
            montage.width - 1,
            montage.height - 1,
        ),
        outline="black",
        width=2,
    )

    montage.save(
        QC_ROOT
        / f"{patient_id}_tile_montage.jpg",
        quality=92,
    )


# ---------------------------------------------------------------------
# Patient extraction
# ---------------------------------------------------------------------

def process_patient(
    patient_row: pd.Series,
    rng: random.Random,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    patient_id = str(
        patient_row["patient_id"]
    )

    series_uid = str(
        patient_row[
            "series_instance_uid"
        ]
    )

    series_description = str(
        patient_row[
            "series_description"
        ]
    )

    series_directory = Path(
        str(
            patient_row[
                "series_directory"
            ]
        )
    )

    patient_output = (
        TILE_ROOT / patient_id
    )

    if patient_output.exists():
        existing_tiles = sorted(
            patient_output.glob("*.jpg")
        )

        if (
            not OVERWRITE_EXISTING_PATIENTS
            and len(existing_tiles)
            >= TILES_PER_PATIENT
        ):
            logging.info(
                "%s already has %d tiles. Skipping.",
                patient_id,
                len(existing_tiles),
            )

            summary = {
                "patient_id": patient_id,
                "series_instance_uid": series_uid,
                "series_description": series_description,
                "series_directory": str(
                    series_directory
                ),
                "requested_tiles": TILES_PER_PATIENT,
                "accepted_tiles": len(
                    existing_tiles
                ),
                "candidate_attempts": 0,
                "status": "existing",
            }

            return [], summary

        if OVERWRITE_EXISTING_PATIENTS:
            shutil.rmtree(
                patient_output
            )

    patient_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info(
        "Processing %s from %s",
        patient_id,
        series_directory,
    )

    logging.info(
        "%s selected series: %s | %s",
        patient_id,
        series_description,
        series_uid,
    )

    accepted_rows: list[
        dict[str, object]
    ] = []

    accepted_paths: list[Path] = []

    attempted = 0

    with WsiDicom.open(
        series_directory
    ) as slide:
        level0_width = int(
            slide.size.width
        )

        level0_height = int(
            slide.size.height
        )

        thumbnail = slide.read_thumbnail(
            (
                THUMBNAIL_MAX_SIZE,
                THUMBNAIL_MAX_SIZE,
            )
        ).convert("RGB")

        mask = create_tissue_mask(
            thumbnail
        )

        save_tissue_mask_qc(
            patient_id=patient_id,
            thumbnail=thumbnail,
            mask=mask,
        )

        candidates = (
            generate_candidate_locations(
                mask=mask,
                level0_width=level0_width,
                level0_height=level0_height,
                rng=rng,
            )
        )

        if not candidates:
            raise RuntimeError(
                f"No tissue candidates found "
                f"for {patient_id}."
            )

        for (
            x,
            y,
            thumbnail_x,
            thumbnail_y,
        ) in candidates:
            attempted += 1

            try:
                tile = slide.read_region(
                    location=(x, y),
                    level=0,
                    size=(
                        TILE_SIZE,
                        TILE_SIZE,
                    ),
                    threads=4,
                ).convert("RGB")

            except Exception as error:
                logging.warning(
                    "%s: failed reading tile "
                    "at (%d, %d): %s",
                    patient_id,
                    x,
                    y,
                    error,
                )
                continue

            metrics = (
                calculate_tile_metrics(
                    tile
                )
            )

            if not tile_passes_qc(
                metrics
            ):
                continue

            tile_number = (
                len(accepted_rows) + 1
            )

            tile_filename = (
                f"{patient_id}"
                f"_tile_{tile_number:03d}"
                f"_x{x}_y{y}.jpg"
            )

            tile_path = (
                patient_output
                / tile_filename
            )

            tile.save(
                tile_path,
                format="JPEG",
                quality=95,
                subsampling=0,
            )

            accepted_paths.append(
                tile_path
            )

            accepted_rows.append(
                {
                    "patient_id": patient_id,
                    "series_instance_uid": series_uid,
                    "series_description": series_description,
                    "series_directory": str(
                        series_directory
                    ),
                    "tile_number": tile_number,
                    "level": 0,
                    "x": x,
                    "y": y,
                    "tile_size": TILE_SIZE,
                    "thumbnail_x": thumbnail_x,
                    "thumbnail_y": thumbnail_y,
                    "white_fraction": metrics[
                        "white_fraction"
                    ],
                    "dark_fraction": metrics[
                        "dark_fraction"
                    ],
                    "tissue_fraction": metrics[
                        "tissue_fraction"
                    ],
                    "rgb_std": metrics[
                        "rgb_std"
                    ],
                    "mean_red": metrics[
                        "mean_red"
                    ],
                    "mean_green": metrics[
                        "mean_green"
                    ],
                    "mean_blue": metrics[
                        "mean_blue"
                    ],
                    "tile_path": str(
                        tile_path
                    ),
                }
            )

            if (
                len(accepted_rows)
                >= TILES_PER_PATIENT
            ):
                break

    create_tile_montage(
        patient_id=patient_id,
        tile_paths=accepted_paths,
    )

    status = (
        "complete"
        if len(accepted_rows)
        == TILES_PER_PATIENT
        else "insufficient_tiles"
    )

    summary = {
        "patient_id": patient_id,
        "series_instance_uid": series_uid,
        "series_description": series_description,
        "series_directory": str(
            series_directory
        ),
        "requested_tiles": TILES_PER_PATIENT,
        "accepted_tiles": len(
            accepted_rows
        ),
        "candidate_attempts": attempted,
        "status": status,
    }

    logging.info(
        "%s: accepted %d/%d tiles "
        "after %d attempts.",
        patient_id,
        len(accepted_rows),
        TILES_PER_PATIENT,
        attempted,
    )

    return accepted_rows, summary


# ---------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------

def update_manifest(
    new_rows: list[
        dict[str, object]
    ],
) -> None:
    if not new_rows:
        return

    new_manifest = pd.DataFrame(
        new_rows
    )

    if MANIFEST_CSV.exists():
        old_manifest = pd.read_csv(
            MANIFEST_CSV
        )

        combined_manifest = pd.concat(
            [
                old_manifest,
                new_manifest,
            ],
            ignore_index=True,
        )

    else:
        combined_manifest = (
            new_manifest
        )

    combined_manifest = (
        combined_manifest
        .drop_duplicates(
            subset=[
                "patient_id",
                "tile_path",
            ],
            keep="last",
        )
        .sort_values(
            [
                "patient_id",
                "tile_number",
            ]
        )
        .reset_index(drop=True)
    )

    combined_manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )


def update_patient_summary(
    new_rows: list[
        dict[str, object]
    ],
) -> None:
    new_summary = pd.DataFrame(
        new_rows
    )

    if PATIENT_SUMMARY_CSV.exists():
        old_summary = pd.read_csv(
            PATIENT_SUMMARY_CSV
        )

        combined_summary = pd.concat(
            [
                old_summary,
                new_summary,
            ],
            ignore_index=True,
        )

    else:
        combined_summary = (
            new_summary
        )

    combined_summary = (
        combined_summary
        .drop_duplicates(
            subset=["patient_id"],
            keep="last",
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    combined_summary.to_csv(
        PATIENT_SUMMARY_CSV,
        index=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    configure_logging()

    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found: "
            f"{INVENTORY_CSV}"
        )

    inventory = pd.read_csv(
        INVENTORY_CSV
    )

    selected_series = (
        select_one_series_per_patient(
            inventory
        )
    )

    logging.info(
        "Selected one usable HE series "
        "for %d patients.",
        len(selected_series),
    )

    all_manifest_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    for _, patient_row in (
        selected_series.iterrows()
    ):
        patient_id = str(
            patient_row["patient_id"]
        )

        patient_seed = (
            RANDOM_SEED
            + sum(
                ord(character)
                for character
                in patient_id
            )
        )

        patient_rng = random.Random(
            patient_seed
        )

        try:
            (
                manifest_rows,
                summary,
            ) = process_patient(
                patient_row=patient_row,
                rng=patient_rng,
            )

            all_manifest_rows.extend(
                manifest_rows
            )

            summary_rows.append(
                summary
            )

        except Exception as error:
            logging.exception(
                "Patient %s failed: %s",
                patient_id,
                error,
            )

            summary_rows.append(
                {
                    "patient_id": patient_id,
                    "series_instance_uid": str(
                        patient_row[
                            "series_instance_uid"
                        ]
                    ),
                    "series_description": str(
                        patient_row[
                            "series_description"
                        ]
                    ),
                    "series_directory": str(
                        patient_row[
                            "series_directory"
                        ]
                    ),
                    "requested_tiles": TILES_PER_PATIENT,
                    "accepted_tiles": 0,
                    "candidate_attempts": 0,
                    "status": (
                        f"failed: {error}"
                    ),
                }
            )

    update_manifest(
        all_manifest_rows
    )

    update_patient_summary(
        summary_rows
    )

    summary_dataframe = pd.read_csv(
        PATIENT_SUMMARY_CSV
    )

    completed = int(
        summary_dataframe["status"]
        .isin(
            [
                "complete",
                "existing",
            ]
        )
        .sum()
    )

    total_patients = len(
        selected_series
    )

    total_tiles = sum(
        len(
            list(
                patient_directory.glob(
                    "*.jpg"
                )
            )
        )
        for patient_directory
        in TILE_ROOT.iterdir()
        if patient_directory.is_dir()
    )

    print()
    print(
        "CPTAC tissue tile extraction finished"
    )
    print(
        "----------------------------------------"
    )
    print(
        f"Patients selected:  {total_patients}"
    )
    print(
        f"Patients complete:  {completed}"
    )
    print(
        f"New tiles written:  "
        f"{len(all_manifest_rows)}"
    )
    print(
        f"Total saved tiles:  {total_tiles}"
    )
    print(
        f"Tile directory:     {TILE_ROOT}"
    )
    print(
        f"Tile manifest:      {MANIFEST_CSV}"
    )
    print(
        f"Patient summary:    "
        f"{PATIENT_SUMMARY_CSV}"
    )
    print(
        f"QC directory:       {QC_ROOT}"
    )
    print(
        f"Log file:           {LOG_FILE}"
    )


if __name__ == "__main__":
    main()