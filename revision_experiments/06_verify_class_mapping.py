from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MODELS_DIR = Path(
    r"E:\apply\journal publication\onco-probe\models"
)

TEST_CSV = MODELS_DIR / "test_df.csv"

RESNET_WEIGHTS = MODELS_DIR / "resnet50_bh_best.pth"
EFFICIENTNET_WEIGHTS = MODELS_DIR / "efficientnet_bh_best.pth"

BATCH_SIZE = 32
MAX_TEST_IMAGES = 2000


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def identify_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    lower_mapping = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_mapping:
            return lower_mapping[candidate.lower()]

    raise ValueError(
        f"Could not identify a column from {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]:
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return clean_state_dict(value)

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return clean_state_dict(checkpoint)

    raise RuntimeError(
        "Could not locate the model state dictionary."
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def build_resnet(
    device: torch.device,
) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return load_checkpoint(
        model,
        RESNET_WEIGHTS,
        device,
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
        model,
        EFFICIENTNET_WEIGHTS,
        device,
    )


class TestDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        path_column: str,
        label_column: str,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.path_column = path_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, str]:
        row = self.dataframe.iloc[index]

        image_path = Path(
            str(row[self.path_column])
        )

        with Image.open(image_path) as image:
            tensor = IMAGE_TRANSFORM(
                image.convert("RGB")
            )

        label = int(row[self.label_column])

        return tensor, label, str(image_path)


@torch.no_grad()
def evaluate_model(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    total = 0
    correct = 0

    label_counts = {
        0: 0,
        1: 0,
    }

    predicted_counts = {
        0: 0,
        1: 0,
    }

    probability_sum = {
        0: 0.0,
        1: 0.0,
    }

    probability_count = {
        0: 0,
        1: 0,
    }

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        total += labels.numel()
        correct += int(
            (predictions == labels).sum().item()
        )

        for class_index in [0, 1]:
            true_mask = labels == class_index

            label_counts[class_index] += int(
                true_mask.sum().item()
            )

            predicted_counts[class_index] += int(
                (predictions == class_index).sum().item()
            )

            if true_mask.any():
                probability_sum[class_index] += float(
                    probabilities[
                        true_mask,
                        1,
                    ].sum().item()
                )

                probability_count[class_index] += int(
                    true_mask.sum().item()
                )

    print()
    print(model_name)
    print("----------------------------------------")
    print(f"Images evaluated: {total}")
    print(f"Accuracy using stored labels: {correct / total:.4f}")
    print(f"True label counts: {label_counts}")
    print(f"Predicted class counts: {predicted_counts}")

    for true_class in [0, 1]:
        count = probability_count[true_class]

        if count > 0:
            mean_class1_probability = (
                probability_sum[true_class] / count
            )

            print(
                f"Mean P(class 1) for true label "
                f"{true_class}: "
                f"{mean_class1_probability:.4f}"
            )


def main() -> None:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV was not found: {TEST_CSV}"
        )

    dataframe = pd.read_csv(TEST_CSV)

    print("CSV columns:")
    print(list(dataframe.columns))
    print()

    path_column = identify_column(
        dataframe,
        [
            "path",
            "filepath",
            "file_path",
            "image_path",
            "tile_path",
            "filename",
        ],
    )

    label_column = identify_column(
        dataframe,
        [
            "label",
            "target",
            "class",
            "y",
            "tnbc",
        ],
    )

    print(f"Detected path column: {path_column}")
    print(f"Detected label column: {label_column}")
    print()

    print("Stored label distribution:")
    print(
        dataframe[label_column]
        .value_counts(dropna=False)
        .sort_index()
    )

    for optional_column in [
        "class_name",
        "diagnosis",
        "subtype",
        "group",
        "category",
        "tnbc_status",
    ]:
        if optional_column in dataframe.columns:
            print()
            print(
                f"Relationship between {label_column} "
                f"and {optional_column}:"
            )

            print(
                pd.crosstab(
                    dataframe[label_column],
                    dataframe[optional_column],
                    dropna=False,
                )
            )

    dataframe = dataframe[
        dataframe[path_column]
        .astype(str)
        .apply(lambda value: Path(value).exists())
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "None of the image paths in test_df.csv exist. "
            "The original image dataset may have moved."
        )

    if len(dataframe) > MAX_TEST_IMAGES:
        sampled_parts = []

    class_counts = dataframe[label_column].value_counts()
    total_rows = len(dataframe)

    for class_label, class_count in class_counts.items():
        class_sample_size = max(
            1,
            round(
                MAX_TEST_IMAGES
                * class_count
                / total_rows
            ),
        )

        class_subset = dataframe[
            dataframe[label_column] == class_label
        ]

        class_sample_size = min(
            class_sample_size,
            len(class_subset),
        )

        sampled_parts.append(
            class_subset.sample(
                n=class_sample_size,
                random_state=2026,
            )
        )

    dataframe = (
        pd.concat(
            sampled_parts,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=2026,
        )
        .reset_index(drop=True)
    )

    if len(dataframe) > MAX_TEST_IMAGES:
        dataframe = dataframe.iloc[
            :MAX_TEST_IMAGES
        ].copy()

    dataset = TestDataset(
        dataframe=dataframe,
        path_column=path_column,
        label_column=label_column,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    resnet = build_resnet(device)
    efficientnet = build_efficientnet(device)

    evaluate_model(
        "ResNet50-BH",
        resnet,
        loader,
        device,
    )

    evaluate_model(
        "EfficientNet-B0-BH",
        efficientnet,
        loader,
        device,
    )


if __name__ == "__main__":
    main()