from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from wsidicom import WsiDicom


INVENTORY_CSV = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_inventory\cptac_brca_dicom_inventory.csv"
)

OUTPUT_DIR = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_wsidicom_test"
)

TEST_PATIENT = "01BR001"
TEST_SERIES_DESCRIPTION = "HE tumor_tissue"


def is_volume_none(value: str) -> bool:
    try:
        image_type = ast.literal_eval(str(value))
        return "VOLUME" in image_type and "NONE" in image_type
    except (ValueError, SyntaxError):
        text = str(value).upper()
        return "VOLUME" in text and "NONE" in text


def select_test_series(inventory: pd.DataFrame) -> Path:
    patient_rows = inventory[
        (inventory["patient_id"].astype(str) == TEST_PATIENT)
        & (
            inventory["series_description"].astype(str)
            == TEST_SERIES_DESCRIPTION
        )
    ].copy()

    if patient_rows.empty:
        raise RuntimeError(
            f"No {TEST_SERIES_DESCRIPTION!r} series found for "
            f"{TEST_PATIENT}."
        )

    patient_rows["is_full_resolution"] = patient_rows[
        "image_type"
    ].apply(is_volume_none)

    series_summary = (
        patient_rows.groupby("series_instance_uid")
        .agg(
            total_size_mb=("file_size_mb", "sum"),
            has_full_resolution=("is_full_resolution", "max"),
            first_path=("absolute_path", "first"),
        )
        .reset_index()
    )

    full_resolution_series = series_summary[
        series_summary["has_full_resolution"]
    ]

    if full_resolution_series.empty:
        raise RuntimeError(
            f"No full-resolution VOLUME/NONE series found for "
            f"{TEST_PATIENT}."
        )

    selected = full_resolution_series.sort_values(
        "total_size_mb",
        ascending=False,
    ).iloc[0]

    first_file = Path(str(selected["first_path"]))
    series_directory = first_file.parent

    print(f"Selected patient: {TEST_PATIENT}")
    print(f"Selected series UID: {selected['series_instance_uid']}")
    print(f"Series size: {selected['total_size_mb']:.2f} MB")
    print(f"Series directory: {series_directory}")

    return series_directory


def print_slide_information(slide: WsiDicom) -> None:
    print()
    print("Slide information")
    print("----------------------------------------")

    for attribute in [
        "size",
        "mm_size",
        "mpp",
        "level_count",
        "levels",
        "pyramids",
    ]:
        try:
            value = getattr(slide, attribute)
            print(f"{attribute}: {value}")
        except Exception as error:
            print(f"{attribute}: unavailable ({error})")


def save_test_images(slide: WsiDicom) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    thumbnail = slide.read_thumbnail((1200, 1200))
    thumbnail_path = OUTPUT_DIR / f"{TEST_PATIENT}_thumbnail.jpg"
    thumbnail.convert("RGB").save(
        thumbnail_path,
        quality=92,
    )

    print(f"Saved thumbnail: {thumbnail_path}")

    level_indices = [0, 1, 2]

    for level in level_indices:
        try:
            region = slide.read_region(
                location=(1000, 1000),
                level=level,
                size=(224, 224),
                threads=4,
            )

            output_path = (
                OUTPUT_DIR
                / f"{TEST_PATIENT}_level_{level}_tile.jpg"
            )

            region.convert("RGB").save(
                output_path,
                quality=95,
            )

            print(f"Saved level {level} tile: {output_path}")

        except Exception as error:
            print(f"Could not read level {level}: {error}")


def main() -> None:
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found: {INVENTORY_CSV}"
        )

    inventory = pd.read_csv(INVENTORY_CSV)
    series_directory = select_test_series(inventory)

    with WsiDicom.open(series_directory) as slide:
        print_slide_information(slide)
        save_test_images(slide)

    print()
    print("DICOM WSI test completed.")


if __name__ == "__main__":
    main()