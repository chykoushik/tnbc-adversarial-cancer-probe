from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe"
)

SENSITIVITY_ROOT = (
    PROJECT_ROOT / "sensitivity_v3"
)

HARDATA_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "hardata"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "multimodal_gated_fusion"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

FGSM_RESNET_TNBC = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_tnbc_v3.npy"
)

FGSM_RESNET_NON_TNBC = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_non_tnbc_v3.npy"
)

PGD_RESNET_TNBC = (
    SENSITIVITY_ROOT
    / "pgd_resnet_tnbc_v3.npy"
)

PGD_RESNET_NON_TNBC = (
    SENSITIVITY_ROOT
    / "pgd_resnet_non_tnbc_v3.npy"
)

FGSM_EFF_TNBC = (
    SENSITIVITY_ROOT
    / "fgsm_eff_tnbc_v3.npy"
)

PGD_EFF_TNBC = (
    SENSITIVITY_ROOT
    / "pgd_eff_tnbc_v3.npy"
)

GENE_SCALER_FILE = (
    HARDATA_ROOT
    / "gene_scaler.pkl"
)

TOP_GENE_NAMES_FILE = (
    HARDATA_ROOT
    / "top_gene_names.npy"
)

GENE_MATCHED_FILE = (
    HARDATA_ROOT
    / "gene_matched.csv"
)

CLINICAL_FILE = (
    HARDATA_ROOT
    / "clinical_a2_matched.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_FILE = (
    OUTPUT_ROOT
    / "gated_multimodal_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "gated_multimodal_loo_predictions.csv"
)

COMPARISON_FILE = (
    OUTPUT_ROOT
    / "fusion_architecture_comparison.csv"
)

ROC_FILE = (
    OUTPUT_ROOT
    / "fusion_architecture_roc.png"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "gated_multimodal_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

MAX_EPOCHS = 300
PATIENCE = 25

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

LATENT_DIM = 8

BOOTSTRAP_ITERATIONS = 10000


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FILE CHECK
# ============================================================

def check_files():
    required = [
        FGSM_RESNET_TNBC,
        FGSM_RESNET_NON_TNBC,
        PGD_RESNET_TNBC,
        PGD_RESNET_NON_TNBC,
        FGSM_EFF_TNBC,
        PGD_EFF_TNBC,
        GENE_SCALER_FILE,
        TOP_GENE_NAMES_FILE,
        GENE_MATCHED_FILE,
        CLINICAL_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_map(path: Path) -> dict:
    return np.load(
        path,
        allow_pickle=True,
    ).item()


def load_gene_inputs():
    gene_scaler = joblib.load(
        GENE_SCALER_FILE
    )

    top_gene_names = np.load(
        TOP_GENE_NAMES_FILE,
        allow_pickle=True,
    )

    top_gene_names = [
        (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
        )
        for value in top_gene_names
    ]

    gene_df = pd.read_csv(
        GENE_MATCHED_FILE,
        index_col=0,
    )

    clinical = pd.read_csv(
        CLINICAL_FILE
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
        gene_df[common].T
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


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def image_features(
    fgsm_map,
    pgd_map,
):
    fgsm = np.asarray(
        fgsm_map
    ).flatten()

    pgd = np.asarray(
        pgd_map
    ).flatten()

    return [
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


def gene_features(
    patient_id,
    gene_data_top,
    gene_scaler,
):
    values = (
        gene_scaler.transform(
            gene_data_top
            .loc[[patient_id]]
            .values
        )[0]
    )

    return [
        values.mean(),
        values.max(),
        values.std(),
        np.percentile(
            values,
            75,
        ),
        np.percentile(
            values,
            90,
        ),
    ]


def build_dataset(
    fgsm_tnbc,
    fgsm_non_tnbc,
    pgd_tnbc,
    pgd_non_tnbc,
    gene_data_top,
    gene_scaler,
):
    image_X = []
    gene_X = []
    y = []
    patient_ids = []

    for pid in fgsm_tnbc:

        if pid not in gene_data_top.index:
            continue

        if pid not in pgd_tnbc:
            continue

        image_X.append(
            image_features(
                fgsm_tnbc[pid],
                pgd_tnbc[pid],
            )
        )

        gene_X.append(
            gene_features(
                pid,
                gene_data_top,
                gene_scaler,
            )
        )

        y.append(1)
        patient_ids.append(pid)

    for pid in fgsm_non_tnbc:

        if pid not in gene_data_top.index:
            continue

        if pid not in pgd_non_tnbc:
            continue

        image_X.append(
            image_features(
                fgsm_non_tnbc[pid],
                pgd_non_tnbc[pid],
            )
        )

        gene_X.append(
            gene_features(
                pid,
                gene_data_top,
                gene_scaler,
            )
        )

        y.append(0)
        patient_ids.append(pid)

    return (
        np.asarray(
            image_X,
            dtype=np.float32,
        ),
        np.asarray(
            gene_X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
        patient_ids,
    )


# ============================================================
# GATED MULTIMODAL UNIT
# ============================================================

class GatedMultimodalNetwork(nn.Module):

    def __init__(
        self,
        image_dim=7,
        gene_dim=5,
        latent_dim=8,
    ):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Linear(
                image_dim,
                latent_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.gene_encoder = nn.Sequential(
            nn.Linear(
                gene_dim,
                latent_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.gate = nn.Sequential(
            nn.Linear(
                image_dim + gene_dim,
                latent_dim,
            ),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                latent_dim,
                8,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(
                8,
                1,
            ),
        )

    def forward(
        self,
        image_x,
        gene_x,
    ):
        image_hidden = (
            self.image_encoder(
                image_x
            )
        )

        gene_hidden = (
            self.gene_encoder(
                gene_x
            )
        )

        gate_input = torch.cat(
            [
                image_x,
                gene_x,
            ],
            dim=1,
        )

        z = self.gate(
            gate_input
        )

        fused = (
            z
            * image_hidden
            + (
                1.0 - z
            )
            * gene_hidden
        )

        return self.classifier(
            fused
        ).squeeze(1)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    image_train,
    gene_train,
    y_train,
    image_val=None,
    gene_val=None,
    y_val=None,
    max_epochs=MAX_EPOCHS,
    early_stopping=True,
    seed=42,
):
    set_seed(seed)

    model = GatedMultimodalNetwork(
        image_dim=image_train.shape[1],
        gene_dim=gene_train.shape[1],
        latent_dim=LATENT_DIM,
    ).to(
        DEVICE
    )

    positive_count = float(
        np.sum(
            y_train == 1
        )
    )

    negative_count = float(
        np.sum(
            y_train == 0
        )
    )

    pos_weight = (
        negative_count
        / max(
            positive_count,
            1.0,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    image_train_t = torch.tensor(
        image_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    gene_train_t = torch.tensor(
        gene_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    if image_val is not None:

        image_val_t = torch.tensor(
            image_val,
            dtype=torch.float32,
            device=DEVICE,
        )

        gene_val_t = torch.tensor(
            gene_val,
            dtype=torch.float32,
            device=DEVICE,
        )

        y_val_t = torch.tensor(
            y_val,
            dtype=torch.float32,
            device=DEVICE,
        )

    best_loss = np.inf
    best_epoch = max_epochs

    patience_counter = 0

    best_state = None

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            image_train_t,
            gene_train_t,
        )

        loss = criterion(
            logits,
            y_train_t,
        )

        loss.backward()

        optimizer.step()

        if (
            early_stopping
            and image_val is not None
        ):

            model.eval()

            with torch.no_grad():

                val_logits = model(
                    image_val_t,
                    gene_val_t,
                )

                val_loss = criterion(
                    val_logits,
                    y_val_t,
                ).item()

            if val_loss < (
                best_loss
                - 1e-6
            ):

                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                best_state = {
                    key:
                        value.detach()
                        .cpu()
                        .clone()
                    for key, value
                    in model.state_dict().items()
                }

            else:

                patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):
                break

    if (
        early_stopping
        and best_state is not None
    ):
        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_epoch,
    )


# ============================================================
# OUTER LOO-CV
# ============================================================

def gated_loo_predictions(
    image_X,
    gene_X,
    y,
    patient_ids,
    model_name,
):
    loo = LeaveOneOut()

    probabilities = []
    truths = []
    pids = []
    chosen_epochs = []

    total = len(y)

    for fold, (
        outer_train_idx,
        outer_test_idx,
    ) in enumerate(
        loo.split(
            image_X
        ),
        start=1,
    ):
        outer_y = (
            y[
                outer_train_idx
            ]
        )

        splitter = (
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.20,
                random_state=(
                    RANDOM_SEED
                    + fold
                ),
            )
        )

        inner_train_local, inner_val_local = (
            next(
                splitter.split(
                    np.zeros(
                        len(
                            outer_train_idx
                        )
                    ),
                    outer_y,
                )
            )
        )

        inner_train_idx = (
            outer_train_idx[
                inner_train_local
            ]
        )

        inner_val_idx = (
            outer_train_idx[
                inner_val_local
            ]
        )

        image_scaler_inner = (
            StandardScaler()
        )

        gene_scaler_inner = (
            StandardScaler()
        )

        image_train_inner = (
            image_scaler_inner
            .fit_transform(
                image_X[
                    inner_train_idx
                ]
            )
        )

        image_val_inner = (
            image_scaler_inner
            .transform(
                image_X[
                    inner_val_idx
                ]
            )
        )

        gene_train_inner = (
            gene_scaler_inner
            .fit_transform(
                gene_X[
                    inner_train_idx
                ]
            )
        )

        gene_val_inner = (
            gene_scaler_inner
            .transform(
                gene_X[
                    inner_val_idx
                ]
            )
        )

        _, best_epoch = train_model(
            image_train_inner,
            gene_train_inner,
            y[
                inner_train_idx
            ],
            image_val_inner,
            gene_val_inner,
            y[
                inner_val_idx
            ],
            early_stopping=True,
            seed=(
                RANDOM_SEED
                + fold
            ),
        )

        best_epoch = max(
            1,
            int(
                best_epoch
            ),
        )

        chosen_epochs.append(
            best_epoch
        )

        image_scaler_outer = (
            StandardScaler()
        )

        gene_scaler_outer = (
            StandardScaler()
        )

        image_train_outer = (
            image_scaler_outer
            .fit_transform(
                image_X[
                    outer_train_idx
                ]
            )
        )

        image_test_outer = (
            image_scaler_outer
            .transform(
                image_X[
                    outer_test_idx
                ]
            )
        )

        gene_train_outer = (
            gene_scaler_outer
            .fit_transform(
                gene_X[
                    outer_train_idx
                ]
            )
        )

        gene_test_outer = (
            gene_scaler_outer
            .transform(
                gene_X[
                    outer_test_idx
                ]
            )
        )

        final_model, _ = train_model(
            image_train_outer,
            gene_train_outer,
            y[
                outer_train_idx
            ],
            max_epochs=best_epoch,
            early_stopping=False,
            seed=(
                RANDOM_SEED
                + 1000
                + fold
            ),
        )

        final_model.eval()

        with torch.no_grad():

            image_test_t = torch.tensor(
                image_test_outer,
                dtype=torch.float32,
                device=DEVICE,
            )

            gene_test_t = torch.tensor(
                gene_test_outer,
                dtype=torch.float32,
                device=DEVICE,
            )

            logit = final_model(
                image_test_t,
                gene_test_t,
            )

            probability = (
                torch.sigmoid(
                    logit
                )
                .cpu()
                .numpy()[0]
            )

        probabilities.append(
            float(
                probability
            )
        )

        truths.append(
            int(
                y[
                    outer_test_idx[
                        0
                    ]
                ]
            )
        )

        pids.append(
            patient_ids[
                outer_test_idx[
                    0
                ]
            ]
        )

        if (
            fold % 10 == 0
            or fold == total
        ):
            print(
                f"{model_name}: "
                f"LOO fold "
                f"{fold}/{total}"
            )

    return (
        np.asarray(
            truths
        ),
        np.asarray(
            probabilities
        ),
        pids,
        chosen_epochs,
    )


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_auc(
    y_true,
    probabilities,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=42,
):
    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    positive_idx = np.where(
        y_true == 1
    )[0]

    negative_idx = np.where(
        y_true == 0
    )[0]

    rng = np.random.default_rng(
        seed
    )

    aucs = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(
        iterations
    ):

        sampled_positive = (
            rng.choice(
                positive_idx,
                size=len(
                    positive_idx
                ),
                replace=True,
            )
        )

        sampled_negative = (
            rng.choice(
                negative_idx,
                size=len(
                    negative_idx
                ),
                replace=True,
            )
        )

        indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        aucs[i] = roc_auc_score(
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


# ============================================================
# RUN ONE ARCHITECTURE
# ============================================================

def analyse_architecture(
    model_name,
    image_X,
    gene_X,
    y,
    patient_ids,
    bootstrap_seed,
):
    print()
    print(
        model_name
    )
    print(
        "-" * 60
    )

    (
        truths,
        probabilities,
        pids,
        epochs,
    ) = gated_loo_predictions(
        image_X,
        gene_X,
        y,
        patient_ids,
        model_name,
    )

    auc = roc_auc_score(
        truths,
        probabilities,
    )

    ci_lower, ci_upper = (
        bootstrap_auc(
            truths,
            probabilities,
            seed=bootstrap_seed,
        )
    )

    print()
    print(
        f"Patients: {len(truths)}"
    )

    print(
        f"TNBC: {int(truths.sum())}"
    )

    print(
        f"LOO-CV AUC: {auc:.4f}"
    )

    print(
        f"Bootstrap 95% CI: "
        f"{ci_lower:.4f} "
        f"to {ci_upper:.4f}"
    )

    print(
        f"Median selected epochs: "
        f"{np.median(epochs):.1f}"
    )

    result = {
        "model":
            model_name,
        "fusion_type":
            "Gated Multimodal Unit",
        "loo_auc":
            float(
                auc
            ),
        "ci95_lower":
            ci_lower,
        "ci95_upper":
            ci_upper,
        "patients":
            int(
                len(
                    truths
                )
            ),
        "tnbc":
            int(
                truths.sum()
            ),
        "non_tnbc":
            int(
                (
                    truths == 0
                ).sum()
            ),
        "median_selected_epochs":
            float(
                np.median(
                    epochs
                )
            ),
    }

    predictions = pd.DataFrame(
        {
            "model":
                model_name,
            "patient_id":
                pids,
            "true_label":
                truths,
            "loo_probability":
                probabilities,
        }
    )

    return (
        result,
        predictions,
    )


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

def build_comparison_table(
    gated_results,
):
    rows = []

    original_file = (
        PROJECT_ROOT
        / "reviewer"
        / "output"
        / "fusion_bootstrap_ci"
        / "fusion_bootstrap_auc_ci.csv"
    )

    if original_file.exists():

        original = pd.read_csv(
            original_file
        )

        for _, row in (
            original.iterrows()
        ):

            rows.append(
                {
                    "model":
                        row[
                            "model"
                        ],
                    "fusion_type":
                        (
                            "Random Forest/XGBoost"
                        ),
                    "loo_auc":
                        row[
                            "loo_auc"
                        ],
                    "ci95_lower":
                        row[
                            "bootstrap_ci95_lower"
                        ],
                    "ci95_upper":
                        row[
                            "bootstrap_ci95_upper"
                        ],
                }
            )

    alternative_file = (
        PROJECT_ROOT
        / "reviewer"
        / "output"
        / "reviewer2_remaining_analyses"
        / "alternative_fusion_baselines.csv"
    )

    if alternative_file.exists():

        alternative = pd.read_csv(
            alternative_file
        )

        for _, row in (
            alternative.iterrows()
        ):

            rows.append(
                {
                    "model":
                        (
                            str(
                                row[
                                    "image_features"
                                ]
                            )
                            + " + "
                            + str(
                                row[
                                    "fusion_model"
                                ]
                            )
                        ),
                    "fusion_type":
                        row[
                            "fusion_model"
                        ],
                    "loo_auc":
                        row[
                            "loo_auc"
                        ],
                    "ci95_lower":
                        row[
                            "ci95_lower"
                        ],
                    "ci95_upper":
                        row[
                            "ci95_upper"
                        ],
                }
            )

    for result in gated_results:

        rows.append(
            {
                "model":
                    result[
                        "model"
                    ],
                "fusion_type":
                    (
                        "Gated Multimodal Unit"
                    ),
                "loo_auc":
                    result[
                        "loo_auc"
                    ],
                "ci95_lower":
                    result[
                        "ci95_lower"
                    ],
                "ci95_upper":
                    result[
                        "ci95_upper"
                    ],
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison = comparison.sort_values(
        "loo_auc",
        ascending=False,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    return comparison


# ============================================================
# ROC FIGURE
# ============================================================

def make_roc_figure(
    gated_prediction_frames,
):
    plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plotted = False

    for predictions in (
        gated_prediction_frames
    ):

        model_name = (
            predictions[
                "model"
            ].iloc[0]
        )

        y_true = (
            predictions[
                "true_label"
            ].values
        )

        probability = (
            predictions[
                "loo_probability"
            ].values
        )

        auc = roc_auc_score(
            y_true,
            probability,
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            probability,
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{model_name} "
                f"(AUC={auc:.3f})"
            ),
        )

        plotted = True

    original_predictions_file = (
        PROJECT_ROOT
        / "reviewer"
        / "output"
        / "fusion_bootstrap_ci"
        / "fusion_loo_predictions.csv"
    )

    if original_predictions_file.exists():

        original = pd.read_csv(
            original_predictions_file
        )

        preferred_models = [
            "Adversarial-EfficientNet + RF",
            "Adversarial-EfficientNet + XGB",
        ]

        for model_name in (
            preferred_models
        ):

            subset = original[
                original[
                    "model"
                ] == model_name
            ]

            if subset.empty:
                continue

            y_true = (
                subset[
                    "true_label"
                ].values
            )

            probability = (
                subset[
                    "loo_probability"
                ].values
            )

            auc = roc_auc_score(
                y_true,
                probability,
            )

            fpr, tpr, _ = roc_curve(
                y_true,
                probability,
            )

            plt.plot(
                fpr,
                tpr,
                label=(
                    f"{model_name} "
                    f"(AUC={auc:.3f})"
                ),
            )

            plotted = True

    alternative_predictions_file = (
        PROJECT_ROOT
        / "reviewer"
        / "output"
        / "reviewer2_remaining_analyses"
        / "alternative_fusion_predictions.csv"
    )

    if alternative_predictions_file.exists():

        alternative = pd.read_csv(
            alternative_predictions_file
        )

        for fusion_model in [
            "LogisticRegression",
            "MLP",
        ]:

            subset = alternative[
                (
                    alternative[
                        "image_features"
                    ] == "EfficientNet"
                )
                &
                (
                    alternative[
                        "fusion_model"
                    ] == fusion_model
                )
            ]

            if subset.empty:
                continue

            y_true = (
                subset[
                    "true_label"
                ].values
            )

            probability = (
                subset[
                    "loo_probability"
                ].values
            )

            auc = roc_auc_score(
                y_true,
                probability,
            )

            fpr, tpr, _ = roc_curve(
                y_true,
                probability,
            )

            plt.plot(
                fpr,
                tpr,
                label=(
                    f"EfficientNet + "
                    f"{fusion_model} "
                    f"(AUC={auc:.3f})"
                ),
            )

            plotted = True

    if not plotted:
        return

    plt.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Multimodal Fusion Architecture Comparison"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        ROC_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    check_files()

    print()
    print(
        "Gated multimodal fusion analysis"
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

    (
        gene_scaler,
        gene_data_top,
    ) = load_gene_inputs()

    fgsm_resnet_tnbc = load_map(
        FGSM_RESNET_TNBC
    )

    fgsm_resnet_non = load_map(
        FGSM_RESNET_NON_TNBC
    )

    pgd_resnet_tnbc = load_map(
        PGD_RESNET_TNBC
    )

    pgd_resnet_non = load_map(
        PGD_RESNET_NON_TNBC
    )

    fgsm_eff_tnbc = load_map(
        FGSM_EFF_TNBC
    )

    pgd_eff_tnbc = load_map(
        PGD_EFF_TNBC
    )

    (
        image_resnet,
        gene_resnet,
        y_resnet,
        pids_resnet,
    ) = build_dataset(
        fgsm_resnet_tnbc,
        fgsm_resnet_non,
        pgd_resnet_tnbc,
        pgd_resnet_non,
        gene_data_top,
        gene_scaler,
    )

    (
        image_eff,
        gene_eff,
        y_eff,
        pids_eff,
    ) = build_dataset(
        fgsm_eff_tnbc,
        fgsm_resnet_non,
        pgd_eff_tnbc,
        pgd_resnet_non,
        gene_data_top,
        gene_scaler,
    )

    print()
    print(
        "Dataset summary"
    )
    print(
        "-" * 60
    )

    print(
        f"ResNet patients: "
        f"{len(y_resnet)}"
    )

    print(
        f"ResNet TNBC: "
        f"{int(y_resnet.sum())}"
    )

    print(
        f"EfficientNet patients: "
        f"{len(y_eff)}"
    )

    print(
        f"EfficientNet TNBC: "
        f"{int(y_eff.sum())}"
    )

    gated_results = []
    prediction_frames = []

    (
        resnet_result,
        resnet_predictions,
    ) = analyse_architecture(
        "ResNet50 + Gated Multimodal Fusion",
        image_resnet,
        gene_resnet,
        y_resnet,
        pids_resnet,
        bootstrap_seed=42,
    )

    gated_results.append(
        resnet_result
    )

    prediction_frames.append(
        resnet_predictions
    )

    (
        eff_result,
        eff_predictions,
    ) = analyse_architecture(
        "EfficientNet + Gated Multimodal Fusion",
        image_eff,
        gene_eff,
        y_eff,
        pids_eff,
        bootstrap_seed=43,
    )

    gated_results.append(
        eff_result
    )

    prediction_frames.append(
        eff_predictions
    )

    results_frame = pd.DataFrame(
        gated_results
    )

    results_frame.to_csv(
        RESULTS_FILE,
        index=False,
    )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    comparison = build_comparison_table(
        gated_results
    )

    make_roc_figure(
        prediction_frames
    )

    print()
    print(
        "Full fusion architecture comparison"
    )
    print(
        "=" * 60
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    summary = {
        "architecture":
            (
                "Compact gated multimodal "
                "neural fusion network"
            ),
        "outer_evaluation":
            (
                "Patient-level leave-one-out "
                "cross-validation"
            ),
        "inner_model_selection":
            (
                "Stratified 80/20 split within "
                "each outer training fold"
            ),
        "class_imbalance":
            (
                "Positive-class weighting in "
                "BCEWithLogitsLoss"
            ),
        "image_features": 7,
        "gene_features": 5,
        "latent_dimension":
            LATENT_DIM,
        "maximum_epochs":
            MAX_EPOCHS,
        "early_stopping_patience":
            PATIENCE,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "results":
            gated_results,
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
        f"Gated results: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Comparison table: "
        f"{COMPARISON_FILE}"
    )

    print(
        f"ROC figure: "
        f"{ROC_FILE}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()