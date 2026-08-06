from __future__ import annotations

import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATASET_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\dataset\CPTAC_BRCA"
)

OUTPUT_DIR = Path(
    r"E:\apply\journal publication\onco-probe\reviewer\output\cptac_inventory"
)

OUTPUT_CSV = OUTPUT_DIR / "cptac_brca_dicom_inventory.csv"
LOG_FILE = OUTPUT_DIR / "cptac_inventory.log"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------

def safe_get(dataset: pydicom.Dataset, attribute: str, default: Any = "") -> Any:
    value = getattr(dataset, attribute, default)

    if value is None:
        return default

    return str(value)


def is_probable_dicom(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True

    try:
        with path.open("rb") as file_handle:
            header = file_handle.read(132)

        return len(header) >= 132 and header[128:132] == b"DICM"

    except OSError:
        return False


def read_dicom_metadata(path: Path) -> dict[str, Any] | None:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "Modality",
                "StudyDescription",
                "SeriesDescription",
                "BodyPartExamined",
                "Manufacturer",
                "ManufacturerModelName",
                "Rows",
                "Columns",
                "TotalPixelMatrixRows",
                "TotalPixelMatrixColumns",
                "NumberOfFrames",
                "PhotometricInterpretation",
                "ImageType",
            ],
        )

    except (InvalidDicomError, OSError, ValueError) as error:
        logging.warning("Could not read %s: %s", path, error)
        return None

    patient_id = safe_get(dataset, "PatientID")

    if not patient_id:
        try:
            patient_id = path.relative_to(DATASET_ROOT).parts[0]
        except (ValueError, IndexError):
            patient_id = path.parent.name

    file_size_bytes = path.stat().st_size

    return {
        "patient_id": patient_id,
        "study_instance_uid": safe_get(dataset, "StudyInstanceUID"),
        "series_instance_uid": safe_get(dataset, "SeriesInstanceUID"),
        "sop_instance_uid": safe_get(dataset, "SOPInstanceUID"),
        "modality": safe_get(dataset, "Modality"),
        "study_description": safe_get(dataset, "StudyDescription"),
        "series_description": safe_get(dataset, "SeriesDescription"),
        "body_part_examined": safe_get(dataset, "BodyPartExamined"),
        "manufacturer": safe_get(dataset, "Manufacturer"),
        "manufacturer_model": safe_get(dataset, "ManufacturerModelName"),
        "rows": safe_get(dataset, "Rows"),
        "columns": safe_get(dataset, "Columns"),
        "total_pixel_matrix_rows": safe_get(
            dataset,
            "TotalPixelMatrixRows",
        ),
        "total_pixel_matrix_columns": safe_get(
            dataset,
            "TotalPixelMatrixColumns",
        ),
        "number_of_frames": safe_get(dataset, "NumberOfFrames"),
        "photometric_interpretation": safe_get(
            dataset,
            "PhotometricInterpretation",
        ),
        "image_type": safe_get(dataset, "ImageType"),
        "file_size_mb": round(file_size_bytes / (1024 ** 2), 3),
        "relative_path": str(path.relative_to(DATASET_ROOT)),
        "absolute_path": str(path),
    }


# ---------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------

def find_candidate_files() -> list[Path]:
    candidates: list[Path] = []

    logging.info("Searching dataset root: %s", DATASET_ROOT)

    for path in DATASET_ROOT.rglob("*"):
        if is_probable_dicom(path):
            candidates.append(path)

    return sorted(candidates)


def write_inventory(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No readable DICOM files were found.")

    fieldnames = list(rows[0].keys())

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    patients = sorted(
        {
            str(row["patient_id"]).strip()
            for row in rows
            if str(row["patient_id"]).strip()
        }
    )

    series = {
        (
            row["patient_id"],
            row["series_instance_uid"],
        )
        for row in rows
        if row["series_instance_uid"]
    }

    modalities = Counter(row["modality"] for row in rows)
    descriptions = Counter(row["series_description"] for row in rows)

    total_size_gb = sum(
        float(row["file_size_mb"])
        for row in rows
    ) / 1024

    print()
    print("CPTAC-BRCA inventory complete")
    print("----------------------------------------")
    print(f"Patient count:       {len(patients)}")
    print(f"DICOM file count:    {len(rows)}")
    print(f"Series count:        {len(series)}")
    print(f"Indexed size:        {total_size_gb:.2f} GB")
    print(f"Modalities:          {dict(modalities)}")
    print()
    print("Series descriptions:")

    for description, count in descriptions.most_common():
        label = description if description else "<missing>"
        print(f"  {label}: {count}")

    print()
    print(f"Inventory CSV: {OUTPUT_CSV}")
    print(f"Log file:      {LOG_FILE}")


def main() -> None:
    configure_logging()

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_ROOT}"
        )

    candidate_files = find_candidate_files()

    logging.info(
        "Found %d probable DICOM files.",
        len(candidate_files),
    )

    inventory_rows: list[dict[str, Any]] = []

    for index, dicom_path in enumerate(candidate_files, start=1):
        metadata = read_dicom_metadata(dicom_path)

        if metadata is not None:
            inventory_rows.append(metadata)

        if index % 100 == 0 or index == len(candidate_files):
            logging.info(
                "Processed %d of %d candidate files.",
                index,
                len(candidate_files),
            )

    write_inventory(inventory_rows)
    print_summary(inventory_rows)


if __name__ == "__main__":
    main()