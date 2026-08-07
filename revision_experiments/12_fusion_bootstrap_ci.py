from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from xgboost import XGBClassifier


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
    PROJECT_ROOT / "reviewer" / "hardata"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "fusion_bootstrap_ci"
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

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "fusion_loo_predictions.csv"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "fusion_bootstrap_auc_ci.csv"
)

BOOTSTRAP_FILE = (
    OUTPUT_ROOT
    / "fusion_bootstrap_distributions.csv"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "fusion_bootstrap_summary.json"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "fusion_bootstrap_ci.log"
)


# ============================================================
# SETTINGS
# ============================================================

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42


# ============================================================
# LOGGING
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def check_files() -> None:
    files = [
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
        for path in files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )


def load_map(path: Path) -> dict:
    data = np.load(
        path,
        allow_pickle=True,
    )

    if hasattr(data, "item"):
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dictionary in {path}"
        )

    return data


def clean_gene_names(
    values: np.ndarray,
) -> list[str]:
    names = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode(
                "utf-8"
            )

        names.append(
            str(value)
        )

    return names


# ============================================================
# LOAD ORIGINAL ANALYSIS INPUTS
# ============================================================

def load_inputs():
    logger.info(
        "Loading original fusion analysis inputs."
    )

    fgsm_resnet_tnbc = load_map(
        FGSM_RESNET_TNBC
    )

    fgsm_resnet_non_tnbc = load_map(
        FGSM_RESNET_NON_TNBC
    )

    pgd_resnet_tnbc = load_map(
        PGD_RESNET_TNBC
    )

    pgd_resnet_non_tnbc = load_map(
        PGD_RESNET_NON_TNBC
    )

    fgsm_eff_tnbc = load_map(
        FGSM_EFF_TNBC
    )

    pgd_eff_tnbc = load_map(
        PGD_EFF_TNBC
    )

    gene_scaler = joblib.load(
        GENE_SCALER_FILE
    )

    top_gene_names = clean_gene_names(
        np.load(
            TOP_GENE_NAMES_FILE,
            allow_pickle=True,
        )
    )

    clinical = pd.read_csv(
        CLINICAL_FILE
    )

    gene_df = pd.read_csv(
        GENE_MATCHED_FILE,
        index_col=0,
    )

    if "Sample ID" not in clinical.columns:
        raise ValueError(
            "clinical_a2_matched.csv does not "
            "contain 'Sample ID'."
        )

    if "TNBC" not in clinical.columns:
        raise ValueError(
            "clinical_a2_matched.csv does not "
            "contain 'TNBC'."
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

    missing_genes = [
        gene
        for gene in top_gene_names
        if gene not in gene_data.columns
    ]

    if missing_genes:
        raise ValueError(
            f"{len(missing_genes)} top genes "
            "are missing from gene_matched.csv."
        )

    gene_data_top = (
        gene_data[top_gene_names]
    )

    logger.info(
        "Clinical patients: %d",
        clinical["Sample ID"].nunique(),
    )

    logger.info(
        "Matched gene patients: %d",
        len(gene_data_top),
    )

    logger.info(
        "Top genes: %d",
        len(top_gene_names),
    )

    return {
        "fgsm_resnet_tnbc":
            fgsm_resnet_tnbc,
        "fgsm_resnet_non_tnbc":
            fgsm_resnet_non_tnbc,
        "pgd_resnet_tnbc":
            pgd_resnet_tnbc,
        "pgd_resnet_non_tnbc":
            pgd_resnet_non_tnbc,
        "fgsm_eff_tnbc":
            fgsm_eff_tnbc,
        "pgd_eff_tnbc":
            pgd_eff_tnbc,
        "gene_scaler":
            gene_scaler,
        "gene_data_top":
            gene_data_top,
        "clinical":
            clinical,
    }


# ============================================================
# REPRODUCE ORIGINAL 12 FEATURE FUSION MATRIX
# ============================================================

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
        if pid not in gene_data_top.index:
            continue

        if pid not in pgd_maps_tnbc:
            continue

        img_map = (
            img_maps_tnbc[pid]
            .flatten()
        )

        pgd_map = (
            pgd_maps_tnbc[pid]
            .flatten()
        )

        img_vec = [
            img_map.mean(),
            img_map.max(),
            img_map.std(),
            np.percentile(
                img_map,
                75,
            ),
            np.percentile(
                img_map,
                90,
            ),
            pgd_map.mean(),
            pgd_map.max(),
        ]

        gene_vals = (
            gene_scaler.transform(
                gene_data_top
                .loc[[pid]]
                .values
            )[0]
        )

        gene_vec = [
            gene_vals.mean(),
            gene_vals.max(),
            gene_vals.std(),
            np.percentile(
                gene_vals,
                75,
            ),
            np.percentile(
                gene_vals,
                90,
            ),
        ]

        X.append(
            img_vec + gene_vec
        )

        y.append(1)
        pids.append(pid)

    for pid in img_maps_non_tnbc:
        if pid not in gene_data_top.index:
            continue

        if pid not in pgd_maps_non_tnbc:
            continue

        img_map = (
            img_maps_non_tnbc[pid]
            .flatten()
        )

        pgd_map = (
            pgd_maps_non_tnbc[pid]
            .flatten()
        )

        img_vec = [
            img_map.mean(),
            img_map.max(),
            img_map.std(),
            np.percentile(
                img_map,
                75,
            ),
            np.percentile(
                img_map,
                90,
            ),
            pgd_map.mean(),
            pgd_map.max(),
        ]

        gene_vals = (
            gene_scaler.transform(
                gene_data_top
                .loc[[pid]]
                .values
            )[0]
        )

        gene_vec = [
            gene_vals.mean(),
            gene_vals.max(),
            gene_vals.std(),
            np.percentile(
                gene_vals,
                75,
            ),
            np.percentile(
                gene_vals,
                90,
            ),
        ]

        X.append(
            img_vec + gene_vec
        )

        y.append(0)
        pids.append(pid)

    return (
        np.asarray(
            X,
            dtype=np.float64,
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
        pids,
    )


# ============================================================
# EXACT ORIGINAL LOO MODELS
# ============================================================

def create_model(
    model_type: str,
    y_train: np.ndarray,
):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            class_weight="balanced",
            random_state=42,
        )

    if model_type == "xgb":
        n_tnbc = int(
            y_train.sum()
        )

        n_non = int(
            (
                y_train == 0
            ).sum()
        )

        scale = (
            n_non
            / max(
                n_tnbc,
                1,
            )
        )

        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            scale_pos_weight=scale,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# ============================================================
# LOO PREDICTIONS
# ============================================================

def loo_predictions(
    X: np.ndarray,
    y: np.ndarray,
    pids: list[str],
    model_type: str,
):
    loo = LeaveOneOut()

    probabilities = []
    truths = []
    patient_ids = []

    total = len(X)

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        loo.split(X),
        start=1,
    ):
        y_train = y[train_idx]

        if (
            len(
                np.unique(
                    y_train
                )
            )
            < 2
        ):
            continue

        model = create_model(
            model_type,
            y_train,
        )

        model.fit(
            X[train_idx],
            y_train,
        )

        probability_matrix = (
            model.predict_proba(
                X[test_idx]
            )
        )

        classes = list(
            model.classes_
        )

        if 1 in classes:
            positive_probability = float(
                probability_matrix[
                    0,
                    classes.index(1),
                ]
            )
        else:
            positive_probability = 0.0

        probabilities.append(
            positive_probability
        )

        truths.append(
            int(
                y[
                    test_idx
                ][0]
            )
        )

        patient_ids.append(
            pids[
                int(
                    test_idx[0]
                )
            ]
        )

        if (
            fold_number % 20 == 0
            or fold_number == total
        ):
            logger.info(
                "%s LOO fold %d/%d",
                model_type.upper(),
                fold_number,
                total,
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    truths = np.asarray(
        truths,
        dtype=np.int64,
    )

    auc = roc_auc_score(
        truths,
        probabilities,
    )

    return (
        auc,
        truths,
        probabilities,
        patient_ids,
    )


# ============================================================
# STRATIFIED PATIENT BOOTSTRAP
# ============================================================

def bootstrap_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
):
    rng = np.random.default_rng(
        seed
    )

    positive_indices = np.where(
        y_true == 1
    )[0]

    negative_indices = np.where(
        y_true == 0
    )[0]

    if len(positive_indices) == 0:
        raise ValueError(
            "No positive patients found."
        )

    if len(negative_indices) == 0:
        raise ValueError(
            "No negative patients found."
        )

    bootstrap_aucs = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(
        iterations
    ):
        sampled_positive = (
            rng.choice(
                positive_indices,
                size=len(
                    positive_indices
                ),
                replace=True,
            )
        )

        sampled_negative = (
            rng.choice(
                negative_indices,
                size=len(
                    negative_indices
                ),
                replace=True,
            )
        )

        sampled_indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        bootstrap_aucs[
            iteration
        ] = roc_auc_score(
            y_true[
                sampled_indices
            ],
            probabilities[
                sampled_indices
            ],
        )

    lower = float(
        np.percentile(
            bootstrap_aucs,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_aucs,
            97.5,
        )
    )

    return (
        bootstrap_aucs,
        lower,
        upper,
    )


# ============================================================
# RUN ONE MODEL
# ============================================================

def analyse_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    pids: list[str],
    model_type: str,
    bootstrap_seed: int,
):
    logger.info(
        "Running %s.",
        name,
    )

    (
        auc,
        truths,
        probabilities,
        patient_ids,
    ) = loo_predictions(
        X=X,
        y=y,
        pids=pids,
        model_type=model_type,
    )

    (
        bootstrap_values,
        ci_lower,
        ci_upper,
    ) = bootstrap_auc(
        y_true=truths,
        probabilities=probabilities,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=bootstrap_seed,
    )

    prediction_frame = pd.DataFrame(
        {
            "model": name,
            "patient_id":
                patient_ids,
            "true_label":
                truths,
            "loo_probability":
                probabilities,
        }
    )

    result = {
        "model": name,
        "loo_auc": float(
            auc
        ),
        "bootstrap_ci95_lower":
            ci_lower,
        "bootstrap_ci95_upper":
            ci_upper,
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "patients": int(
            len(truths)
        ),
        "tnbc_patients": int(
            truths.sum()
        ),
        "non_tnbc_patients": int(
            (
                truths == 0
            ).sum()
        ),
    }

    logger.info(
        "%s AUC %.4f "
        "95%% CI %.4f to %.4f",
        name,
        auc,
        ci_lower,
        ci_upper,
    )

    return (
        result,
        prediction_frame,
        bootstrap_values,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    check_files()

    inputs = load_inputs()

    gene_data_top = (
        inputs["gene_data_top"]
    )

    gene_scaler = (
        inputs["gene_scaler"]
    )

    # --------------------------------------------------------
    # ResNet feature matrix
    # --------------------------------------------------------

    (
        X_resnet,
        y_resnet,
        pids_resnet,
    ) = build_combined_features(
        inputs[
            "fgsm_resnet_tnbc"
        ],
        inputs[
            "fgsm_resnet_non_tnbc"
        ],
        inputs[
            "pgd_resnet_tnbc"
        ],
        inputs[
            "pgd_resnet_non_tnbc"
        ],
        gene_data_top,
        gene_scaler,
    )

    # --------------------------------------------------------
    # EfficientNet feature matrix
    #
    # This deliberately reproduces the original notebook.
    # EfficientNet maps were available for TNBC patients.
    # The original fusion analysis used the ResNet maps for
    # the non-TNBC side of the EfficientNet combination.
    # --------------------------------------------------------

    (
        X_eff,
        y_eff,
        pids_eff,
    ) = build_combined_features(
        inputs[
            "fgsm_eff_tnbc"
        ],
        inputs[
            "fgsm_resnet_non_tnbc"
        ],
        inputs[
            "pgd_eff_tnbc"
        ],
        inputs[
            "pgd_resnet_non_tnbc"
        ],
        gene_data_top,
        gene_scaler,
    )

    print()
    print(
        "Fusion feature matrices"
    )
    print(
        "----------------------------------------"
    )

    print(
        f"ResNet matrix:       "
        f"{X_resnet.shape}"
    )

    print(
        f"ResNet TNBC:         "
        f"{int(y_resnet.sum())}"
    )

    print(
        f"EfficientNet matrix: "
        f"{X_eff.shape}"
    )

    print(
        f"EfficientNet TNBC:   "
        f"{int(y_eff.sum())}"
    )

    analyses = [
        (
            "Adversarial-ResNet50 + RF",
            X_resnet,
            y_resnet,
            pids_resnet,
            "rf",
            42,
        ),
        (
            "Adversarial-ResNet50 + XGB",
            X_resnet,
            y_resnet,
            pids_resnet,
            "xgb",
            43,
        ),
        (
            "Adversarial-EfficientNet + RF",
            X_eff,
            y_eff,
            pids_eff,
            "rf",
            44,
        ),
        (
            "Adversarial-EfficientNet + XGB",
            X_eff,
            y_eff,
            pids_eff,
            "xgb",
            45,
        ),
    ]

    results = []
    prediction_frames = []
    bootstrap_columns = {}

    for (
        name,
        X,
        y,
        pids,
        model_type,
        bootstrap_seed,
    ) in analyses:

        (
            result,
            predictions,
            bootstrap_values,
        ) = analyse_model(
            name=name,
            X=X,
            y=y,
            pids=pids,
            model_type=model_type,
            bootstrap_seed=bootstrap_seed,
        )

        results.append(
            result
        )

        prediction_frames.append(
            predictions
        )

        bootstrap_columns[
            name
        ] = bootstrap_values

    results_frame = pd.DataFrame(
        results
    )

    predictions_frame = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    bootstrap_frame = pd.DataFrame(
        bootstrap_columns
    )

    results_frame.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_frame.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    bootstrap_frame.to_csv(
        BOOTSTRAP_FILE,
        index=False,
    )

    summary = {
        "method": (
            "Patient-level leave-one-out predictions "
            "were generated using the same fusion "
            "features and classifiers as the original "
            "analysis. Stratified patient bootstrap "
            "resampling of the out-of-fold predictions "
            "was performed with replacement within "
            "TNBC and non-TNBC groups. The 2.5th and "
            "97.5th percentiles of 10,000 bootstrap "
            "AUC estimates define the 95% confidence "
            "interval."
        ),
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "bootstrap_seed":
            BOOTSTRAP_SEED,
        "results":
            results,
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
        "Fusion LOO bootstrap analysis"
    )
    print(
        "----------------------------------------"
    )

    for result in results:
        print()
        print(
            result["model"]
        )

        print(
            f"  Patients: "
            f"{result['patients']}"
        )

        print(
            f"  TNBC: "
            f"{result['tnbc_patients']}"
        )

        print(
            f"  LOO AUC: "
            f"{result['loo_auc']:.4f}"
        )

        print(
            "  Bootstrap 95% CI: "
            f"{result['bootstrap_ci95_lower']:.4f} "
            "to "
            f"{result['bootstrap_ci95_upper']:.4f}"
        )

    print()
    print(
        "Saved outputs"
    )
    print(
        "----------------------------------------"
    )

    print(
        f"Results:      {RESULTS_FILE}"
    )

    print(
        f"Predictions:  {PREDICTIONS_FILE}"
    )

    print(
        f"Bootstrap:    {BOOTSTRAP_FILE}"
    )

    print(
        f"Summary:      {SUMMARY_FILE}"
    )

    print(
        f"Log:          {LOG_FILE}"
    )


if __name__ == "__main__":
    main()