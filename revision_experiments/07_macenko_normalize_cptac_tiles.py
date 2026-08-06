from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import torchstain
except ImportError as error:
    raise ImportError(
        "torchstain is not installed. Run: "
        "python -m pip install torchstain"
    ) from error


SOURCE_TILE_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_tiles\tiles"
)

OUTPUT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_tiles_macenko"
)

NORMALIZED_TILE_ROOT = OUTPUT_ROOT / "tiles"
LOG_FILE = OUTPUT_ROOT / "macenko_normalization.log"

REFERENCE_TILE = Path(
    r"E:\apply\journal publication\onco-probe\dataset"
    r"\tcga_images_a2\BLOCKS_NORM_MACENKO"
    r"\TCGA-A2-A0CM-01Z-00-DX1.AC4901DE-4B6D-4185-BB9F-156033839828"
    r"\TCGA-A2-A0CM-01Z-00-DX1.AC4901DE-4B6D-4185-BB9F-156033839828_(10300,43260).jpg"
)

OVERWRITE_EXISTING = False


def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    NORMALIZED_TILE_ROOT.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def normalize_tile(
    normalizer: torchstain.normalizers.MacenkoNormalizer,
    image_path: Path,
) -> Image.Image:
    with Image.open(image_path) as image:
        image_array = pil_to_numpy(image)

    normalized, _, _ = normalizer.normalize(
        I=image_array,
        stains=False,
    )

    if hasattr(normalized, "cpu"):
        normalized = normalized.cpu().numpy()

    normalized = np.asarray(normalized)

    if normalized.ndim == 3 and normalized.shape[0] == 3:
        normalized = np.transpose(normalized, (1, 2, 0))

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return Image.fromarray(normalized, mode="RGB")


def main() -> None:
    configure_logging()

    if not SOURCE_TILE_ROOT.exists():
        raise FileNotFoundError(
            f"Source tile directory not found: {SOURCE_TILE_ROOT}"
        )

    if not REFERENCE_TILE.exists():
        raise FileNotFoundError(
            f"Reference tile not found: {REFERENCE_TILE}"
        )

    if OVERWRITE_EXISTING and NORMALIZED_TILE_ROOT.exists():
        shutil.rmtree(NORMALIZED_TILE_ROOT)
        NORMALIZED_TILE_ROOT.mkdir(parents=True, exist_ok=True)

    normalizer = torchstain.normalizers.MacenkoNormalizer(
        backend="numpy"
    )

    with Image.open(REFERENCE_TILE) as reference_image:
        reference_array = pil_to_numpy(reference_image)

    normalizer.fit(reference_array)

    source_tiles = sorted(
        SOURCE_TILE_ROOT.rglob("*.jpg")
    )

    if not source_tiles:
        raise RuntimeError(
            f"No JPG tiles found in {SOURCE_TILE_ROOT}"
        )

    logging.info(
        "Found %d CPTAC tiles for normalization.",
        len(source_tiles),
    )

    completed = 0
    failed = 0

    for source_path in tqdm(
        source_tiles,
        desc="Macenko normalization",
    ):
        relative_path = source_path.relative_to(
            SOURCE_TILE_ROOT
        )

        destination_path = (
            NORMALIZED_TILE_ROOT / relative_path
        )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination_path.exists()
            and not OVERWRITE_EXISTING
        ):
            completed += 1
            continue

        try:
            normalized_image = normalize_tile(
                normalizer=normalizer,
                image_path=source_path,
            )

            normalized_image.save(
                destination_path,
                format="JPEG",
                quality=95,
                subsampling=0,
            )

            completed += 1

        except Exception as error:
            failed += 1

            logging.exception(
                "Failed to normalize %s: %s",
                source_path,
                error,
            )

    print()
    print("Macenko normalization finished")
    print("----------------------------------------")
    print(f"Reference tile:       {REFERENCE_TILE}")
    print(f"Source tiles:         {len(source_tiles)}")
    print(f"Completed tiles:      {completed}")
    print(f"Failed tiles:         {failed}")
    print(f"Normalized root:      {NORMALIZED_TILE_ROOT}")
    print(f"Log file:             {LOG_FILE}")


if __name__ == "__main__":
    main()