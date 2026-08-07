from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


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
    / "reviewer1_biological_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

CLINICAL_A2_FILE = (
    HARDATA_ROOT
    / "clinical_a2_matched.csv"
)

GENE_A2_FILE = (
    HARDATA_ROOT
    / "gene_matched.csv"
)

GENE_E2_FILE = (
    HARDATA_ROOT
    / "gene_e2_all.csv"
)

METABRIC_RESULTS_FILE = (
    HARDATA_ROOT
    / "metabric_validation_results.csv"
)

FGSM_RESNET_TNBC_A2 = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_tnbc_v3.npy"
)

FGSM_RESNET_NON_TNBC_A2 = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_non_tnbc_v3.npy"
)

PGD_RESNET_TNBC_A2 = (
    SENSITIVITY_ROOT
    / "pgd_resnet_tnbc_v3.npy"
)

PGD_RESNET_NON_TNBC_A2 = (
    SENSITIVITY_ROOT
    / "pgd_resnet_non_tnbc_v3.npy"
)

FGSM_RESNET_TNBC_E2 = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_tnbc_e2.npy"
)

FGSM_RESNET_NON_TNBC_E2 = (
    SENSITIVITY_ROOT
    / "fgsm_resnet_non_tnbc_e2.npy"
)

PGD_RESNET_TNBC_E2 = (
    SENSITIVITY_ROOT
    / "pgd_resnet_tnbc_e2.npy"
)

PGD_RESNET_NON_TNBC_E2 = (
    SENSITIVITY_ROOT
    / "pgd_resnet_non_tnbc_e2.npy"
)


# ============================================================
# OUTPUT FILES
# ============================================================

A2_CORRELATIONS_FILE = (
    OUTPUT_ROOT
    / "a2_sensitivity_biology_correlations.csv"
)

E2_CORRELATIONS_FILE = (
    OUTPUT_ROOT
    / "e2_sensitivity_biology_correlations.csv"
)

A2_PATIENT_FILE = (
    OUTPUT_ROOT
    / "a2_patient_biological_scores.csv"
)

E2_PATIENT_FILE = (
    OUTPUT_ROOT
    / "e2_patient_biological_scores.csv"
)

METABRIC_ATTRIBUTES_FILE = (
    OUTPUT_ROOT
    / "metabric_receptor_attribute_candidates.csv"
)

METABRIC_CLINICAL_FILE = (
    OUTPUT_ROOT
    / "metabric_receptor_clinical_data.csv"
)

METABRIC_STRICT_FILE = (
    OUTPUT_ROOT
    / "metabric_strict_ihc_sensitivity.csv"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "reviewer1_remaining_summary.json"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "reviewer1_remaining.log"
)


# ============================================================
# SETTINGS
# ============================================================

CBIOPORTAL_API = (
    "https://www.cbioportal.org/api"
)

METABRIC_STUDY = "brca_metabric"

REQUEST_TIMEOUT = 60


# ============================================================
# PREDEFINED BIOLOGICAL MARKERS
# ============================================================

# Defined before looking at correlations.
# These represent canonical luminal/ER-associated
# and basal breast cancer expression programs.

ER_LUMINAL_GENES = [
    "ESR1",
    "PGR",
    "GATA3",
    "FOXA1",
]

BASAL_GENES = [
    "KRT5",
    "KRT14",
    "KRT17",
    "EGFR",
]

HER2_GENES = [
    "ERBB2",
]

ALL_MARKERS = (
    ER_LUMINAL_GENES
    + BASAL_GENES
    + HER2_GENES
)


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
        logging.StreamHandler(
            sys.stdout
        ),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# GENERAL HELPERS
# ============================================================

def check_files() -> None:
    required = [
        CLINICAL_A2_FILE,
        GENE_A2_FILE,
        GENE_E2_FILE,
        METABRIC_RESULTS_FILE,
        FGSM_RESNET_TNBC_A2,
        FGSM_RESNET_NON_TNBC_A2,
        PGD_RESNET_TNBC_A2,
        PGD_RESNET_NON_TNBC_A2,
        FGSM_RESNET_TNBC_E2,
        FGSM_RESNET_NON_TNBC_E2,
        PGD_RESNET_TNBC_E2,
        PGD_RESNET_NON_TNBC_E2,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n"
            + "\n".join(missing)
        )


def load_map(path: Path) -> dict:
    data = np.load(
        path,
        allow_pickle=True,
    )

    data = data.item()

    if not isinstance(data, dict):
        raise TypeError(
            f"{path} is not a dictionary."
        )

    return data


def clean_patient_id(
    value: object,
) -> str:
    text = str(value).strip().upper()

    match = re.search(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})",
        text,
    )

    if match:
        return match.group(1)

    return text


def map_mean(
    mapping: dict,
    patient_id: str,
) -> float:
    return float(
        np.asarray(
            mapping[patient_id]
        ).mean()
    )


# ============================================================
# BENJAMINI-HOCHBERG
# ============================================================

def bh_adjust(
    pvalues: list[float],
) -> np.ndarray:
    values = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        return np.asarray([])

    order = np.argsort(values)
    ranked = values[order]

    adjusted = np.empty(
        n,
        dtype=float,
    )

    running = 1.0

    for index in range(
        n - 1,
        -1,
        -1,
    ):
        rank = index + 1

        candidate = (
            ranked[index]
            * n
            / rank
        )

        running = min(
            running,
            candidate,
        )

        adjusted[
            order[index]
        ] = min(
            running,
            1.0,
        )

    return adjusted


# ============================================================
# GENE MATRIX LOADING
# ============================================================

def load_a2_gene_matrix() -> pd.DataFrame:
    gene_df = pd.read_csv(
        GENE_A2_FILE,
        index_col=0,
    )

    # Original notebook structure:
    # genes are rows and patients are columns.
    matrix = gene_df.T.copy()

    matrix.index = [
        clean_patient_id(x)
        for x in matrix.index
    ]

    matrix.columns = [
        str(x).strip()
        for x in matrix.columns
    ]

    return matrix


def load_e2_gene_matrix() -> pd.DataFrame:
    raw = pd.read_csv(
        GENE_E2_FILE
    )

    if "sample" not in raw.columns:
        raise ValueError(
            "gene_e2_all.csv does not "
            "contain the 'sample' gene column."
        )

    raw = raw.set_index(
        "sample"
    )

    matrix = raw.T.copy()

    matrix.index = [
        clean_patient_id(x)
        for x in matrix.index
    ]

    matrix.columns = [
        str(x).strip()
        for x in matrix.columns
    ]

    return matrix


# ============================================================
# BIOLOGICAL SCORE CONSTRUCTION
# ============================================================

def zscore_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        std = values.std(
            ddof=1
        )

        if (
            not np.isfinite(std)
            or std == 0
        ):
            output[column] = np.nan
        else:
            output[column] = (
                values - values.mean()
            ) / std

    return output


def build_expression_scores(
    expression: pd.DataFrame,
) -> pd.DataFrame:
    available_markers = [
        gene
        for gene in ALL_MARKERS
        if gene in expression.columns
    ]

    logger.info(
        "Available marker genes: %s",
        available_markers,
    )

    subset = expression[
        available_markers
    ].copy()

    subset = subset.apply(
        pd.to_numeric,
        errors="coerce",
    )

    standardized = zscore_columns(
        subset
    )

    output = pd.DataFrame(
        index=expression.index
    )

    for gene in available_markers:
        output[
            f"{gene}_expression"
        ] = subset[gene]

    er_available = [
        gene
        for gene in ER_LUMINAL_GENES
        if gene in standardized.columns
    ]

    basal_available = [
        gene
        for gene in BASAL_GENES
        if gene in standardized.columns
    ]

    her2_available = [
        gene
        for gene in HER2_GENES
        if gene in standardized.columns
    ]

    if er_available:
        output[
            "er_luminal_score"
        ] = standardized[
            er_available
        ].mean(
            axis=1
        )

    if basal_available:
        output[
            "basal_score"
        ] = standardized[
            basal_available
        ].mean(
            axis=1
        )

    if her2_available:
        output[
            "her2_score"
        ] = standardized[
            her2_available
        ].mean(
            axis=1
        )

    if (
        er_available
        and basal_available
    ):
        output[
            "basal_minus_er_score"
        ] = (
            output[
                "basal_score"
            ]
            - output[
                "er_luminal_score"
            ]
        )

    return output


# ============================================================
# ADVERSARIAL SENSITIVITY TABLE
# ============================================================

def build_sensitivity_table(
    fgsm_tnbc: dict,
    fgsm_non_tnbc: dict,
    pgd_tnbc: dict,
    pgd_non_tnbc: dict,
) -> pd.DataFrame:
    rows = []

    for patient_id in fgsm_tnbc:
        if patient_id not in pgd_tnbc:
            continue

        rows.append(
            {
                "patient_id":
                    clean_patient_id(
                        patient_id
                    ),
                "tnbc": 1,
                "fgsm_mean":
                    map_mean(
                        fgsm_tnbc,
                        patient_id,
                    ),
                "pgd_mean":
                    map_mean(
                        pgd_tnbc,
                        patient_id,
                    ),
            }
        )

    for patient_id in fgsm_non_tnbc:
        if patient_id not in pgd_non_tnbc:
            continue

        rows.append(
            {
                "patient_id":
                    clean_patient_id(
                        patient_id
                    ),
                "tnbc": 0,
                "fgsm_mean":
                    map_mean(
                        fgsm_non_tnbc,
                        patient_id,
                    ),
                "pgd_mean":
                    map_mean(
                        pgd_non_tnbc,
                        patient_id,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ).drop_duplicates(
        subset="patient_id"
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def run_correlations(
    patient_data: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    biological_columns = [
        column
        for column in patient_data.columns
        if (
            column.endswith(
                "_expression"
            )
            or column.endswith(
                "_score"
            )
        )
    ]

    rows = []

    for sensitivity_column in [
        "fgsm_mean",
        "pgd_mean",
    ]:
        for biological_column in biological_columns:
            subset = patient_data[
                [
                    sensitivity_column,
                    biological_column,
                ]
            ].dropna()

            if len(subset) < 5:
                continue

            rho, pvalue = spearmanr(
                subset[
                    sensitivity_column
                ],
                subset[
                    biological_column
                ],
            )

            rows.append(
                {
                    "cohort":
                        cohort,
                    "sensitivity":
                        sensitivity_column,
                    "biological_measure":
                        biological_column,
                    "n":
                        len(subset),
                    "spearman_rho":
                        float(rho),
                    "p_value":
                        float(pvalue),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result[
            "bh_adjusted_p"
        ] = bh_adjust(
            result[
                "p_value"
            ].tolist()
        )

    return result


# ============================================================
# A2 ANALYSIS
# ============================================================

def analyse_a2():
    logger.info(
        "Running A2 biological grounding analysis."
    )

    expression = load_a2_gene_matrix()

    scores = build_expression_scores(
        expression
    )

    sensitivity = build_sensitivity_table(
        load_map(
            FGSM_RESNET_TNBC_A2
        ),
        load_map(
            FGSM_RESNET_NON_TNBC_A2
        ),
        load_map(
            PGD_RESNET_TNBC_A2
        ),
        load_map(
            PGD_RESNET_NON_TNBC_A2
        ),
    )

    patient_data = (
        sensitivity
        .set_index(
            "patient_id"
        )
        .join(
            scores,
            how="inner",
        )
        .reset_index()
    )

    correlations = run_correlations(
        patient_data,
        cohort="TCGA-A2",
    )

    patient_data.to_csv(
        A2_PATIENT_FILE,
        index=False,
    )

    correlations.to_csv(
        A2_CORRELATIONS_FILE,
        index=False,
    )

    logger.info(
        "A2 matched patients: %d",
        len(patient_data),
    )

    return (
        patient_data,
        correlations,
    )


# ============================================================
# E2 ANALYSIS
# ============================================================

def analyse_e2():
    logger.info(
        "Running E2 independent biological validation."
    )

    expression = load_e2_gene_matrix()

    scores = build_expression_scores(
        expression
    )

    sensitivity = build_sensitivity_table(
        load_map(
            FGSM_RESNET_TNBC_E2
        ),
        load_map(
            FGSM_RESNET_NON_TNBC_E2
        ),
        load_map(
            PGD_RESNET_TNBC_E2
        ),
        load_map(
            PGD_RESNET_NON_TNBC_E2
        ),
    )

    patient_data = (
        sensitivity
        .set_index(
            "patient_id"
        )
        .join(
            scores,
            how="inner",
        )
        .reset_index()
    )

    correlations = run_correlations(
        patient_data,
        cohort="TCGA-E2",
    )

    patient_data.to_csv(
        E2_PATIENT_FILE,
        index=False,
    )

    correlations.to_csv(
        E2_CORRELATIONS_FILE,
        index=False,
    )

    logger.info(
        "E2 matched patients: %d",
        len(patient_data),
    )

    return (
        patient_data,
        correlations,
    )


# ============================================================
# CBIOPORTAL
# ============================================================

def cbio_get(
    endpoint: str,
    params: dict | None = None,
):
    url = (
        CBIOPORTAL_API
        + endpoint
    )

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "Accept":
                "application/json"
        },
    )

    response.raise_for_status()

    return response.json()


def get_metabric_attributes() -> pd.DataFrame:
    logger.info(
        "Downloading METABRIC clinical attribute definitions."
    )

    data = cbio_get(
        f"/studies/{METABRIC_STUDY}"
        "/clinical-attributes"
    )

    frame = pd.DataFrame(
        data
    )

    if frame.empty:
        raise RuntimeError(
            "No METABRIC clinical attributes returned."
        )

    return frame


def identify_receptor_attributes(
    attributes: pd.DataFrame,
) -> dict:
    candidate_rows = []

    for _, row in attributes.iterrows():
        attribute_id = str(
            row.get(
                "clinicalAttributeId",
                "",
            )
        )

        display_name = str(
            row.get(
                "displayName",
                "",
            )
        )

        description = str(
            row.get(
                "description",
                "",
            )
        )

        combined = (
            attribute_id
            + " "
            + display_name
            + " "
            + description
        ).lower()

        receptor = None

        if re.search(
            r"\ber\b|estrogen|oestrogen",
            combined,
        ):
            receptor = "ER"

        if re.search(
            r"\bpr\b|progesterone",
            combined,
        ):
            receptor = "PR"

        if re.search(
            r"her2|erbb2",
            combined,
        ):
            receptor = "HER2"

        if receptor:
            candidate_rows.append(
                {
                    "receptor":
                        receptor,
                    "clinicalAttributeId":
                        attribute_id,
                    "displayName":
                        display_name,
                    "description":
                        description,
                    "patientAttribute":
                        row.get(
                            "patientAttribute",
                            None,
                        ),
                }
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates.to_csv(
        METABRIC_ATTRIBUTES_FILE,
        index=False,
    )

    selected = {}

    # Prefer obvious status attributes.
    for receptor in [
        "ER",
        "PR",
        "HER2",
    ]:
        subset = candidates[
            candidates[
                "receptor"
            ] == receptor
        ].copy()

        if subset.empty:
            continue

        subset[
            "score"
        ] = 0

        for index, row in subset.iterrows():
            text = (
                str(
                    row[
                        "clinicalAttributeId"
                    ]
                )
                + " "
                + str(
                    row[
                        "displayName"
                    ]
                )
            ).lower()

            score = 0

            if "status" in text:
                score += 10

            if "ihc" in text:
                score += 5

            if (
                receptor.lower()
                in text
            ):
                score += 3

            subset.loc[
                index,
                "score"
            ] = score

        subset = subset.sort_values(
            "score",
            ascending=False,
        )

        selected[
            receptor
        ] = subset.iloc[0][
            "clinicalAttributeId"
        ]

    logger.info(
        "Selected METABRIC receptor attributes: %s",
        selected,
    )

    return selected


def download_metabric_patient_clinical(
    patient_ids: list[str],
) -> pd.DataFrame:
    rows = []

    total = len(
        patient_ids
    )

    for index, patient_id in enumerate(
        patient_ids,
        start=1,
    ):
        try:
            data = cbio_get(
                f"/studies/{METABRIC_STUDY}"
                f"/patients/{patient_id}"
                "/clinical-data"
            )

            for record in data:
                rows.append(
                    {
                        "patientId":
                            patient_id,
                        "clinicalAttributeId":
                            record.get(
                                "clinicalAttributeId"
                            ),
                        "value":
                            record.get(
                                "value"
                            ),
                    }
                )

        except Exception as error:
            logger.warning(
                "Clinical download failed for %s: %s",
                patient_id,
                error,
            )

        if (
            index % 50 == 0
            or index == total
        ):
            logger.info(
                "METABRIC clinical download %d/%d",
                index,
                total,
            )

    frame = pd.DataFrame(
        rows
    )

    frame.to_csv(
        METABRIC_CLINICAL_FILE,
        index=False,
    )

    return frame


# ============================================================
# RECEPTOR STATUS PARSING
# ============================================================

def parse_receptor_status(
    value: object,
) -> float:
    if pd.isna(value):
        return np.nan

    text = str(
        value
    ).strip().lower()

    negative_terms = [
        "negative",
        "neg",
        "0",
        "false",
    ]

    positive_terms = [
        "positive",
        "pos",
        "1",
        "true",
    ]

    if text in negative_terms:
        return 0.0

    if text in positive_terms:
        return 1.0

    if "negative" in text:
        return 0.0

    if "positive" in text:
        return 1.0

    return np.nan


# ============================================================
# METABRIC SENSITIVITY ANALYSIS
# ============================================================

def analyse_metabric():
    logger.info(
        "Running METABRIC strict receptor analysis."
    )

    validation = pd.read_csv(
        METABRIC_RESULTS_FILE
    )

    attributes = get_metabric_attributes()

    selected = identify_receptor_attributes(
        attributes
    )

    required = {
        "ER",
        "PR",
        "HER2",
    }

    if not required.issubset(
        selected.keys()
    ):
        missing = sorted(
            required
            - set(
                selected.keys()
            )
        )

        logger.warning(
            "Strict IHC analysis cannot be "
            "constructed because receptor "
            "attributes were not identified for: %s",
            missing,
        )

        return {
            "status":
                "not_estimable",
            "reason":
                (
                    "Required ER, PR and HER2 "
                    "clinical attributes were "
                    "not all available."
                ),
            "selected_attributes":
                selected,
        }

    clinical_long = (
        download_metabric_patient_clinical(
            validation[
                "patientId"
            ]
            .astype(str)
            .tolist()
        )
    )

    if clinical_long.empty:
        return {
            "status":
                "not_estimable",
            "reason":
                "No METABRIC clinical data downloaded.",
            "selected_attributes":
                selected,
        }

    clinical_long = clinical_long[
        clinical_long[
            "clinicalAttributeId"
        ].isin(
            list(
                selected.values()
            )
        )
    ].copy()

    reverse_lookup = {
        value: key
        for key, value
        in selected.items()
    }

    clinical_long[
        "receptor"
    ] = clinical_long[
        "clinicalAttributeId"
    ].map(
        reverse_lookup
    )

    clinical_wide = (
        clinical_long
        .pivot_table(
            index="patientId",
            columns="receptor",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    for receptor in [
        "ER",
        "PR",
        "HER2",
    ]:
        if receptor not in clinical_wide.columns:
            clinical_wide[
                receptor
            ] = np.nan

        clinical_wide[
            receptor + "_binary"
        ] = clinical_wide[
            receptor
        ].apply(
            parse_receptor_status
        )

    merged = validation.merge(
        clinical_wide,
        on="patientId",
        how="left",
    )

    confirmed = merged.dropna(
        subset=[
            "ER_binary",
            "PR_binary",
            "HER2_binary",
        ]
    ).copy()

    if confirmed.empty:
        return {
            "status":
                "not_estimable",
            "reason":
                (
                    "No patients had complete "
                    "ER, PR and HER2 status."
                ),
            "selected_attributes":
                selected,
        }

    confirmed[
        "strict_tnbc"
    ] = (
        (
            confirmed[
                "ER_binary"
            ] == 0
        )
        & (
            confirmed[
                "PR_binary"
            ] == 0
        )
        & (
            confirmed[
                "HER2_binary"
            ] == 0
        )
    ).astype(int)

    confirmed[
        "original_label_match"
    ] = (
        confirmed[
            "label"
        ].astype(int)
        == confirmed[
            "strict_tnbc"
        ]
    )

    confirmed.to_csv(
        METABRIC_STRICT_FILE,
        index=False,
    )

    strict_positive = int(
        confirmed[
            "strict_tnbc"
        ].sum()
    )

    strict_negative = int(
        (
            confirmed[
                "strict_tnbc"
            ] == 0
        ).sum()
    )

    rf_auc = None
    xgb_auc = None

    if (
        strict_positive > 0
        and strict_negative > 0
    ):
        rf_auc = float(
            roc_auc_score(
                confirmed[
                    "strict_tnbc"
                ],
                confirmed[
                    "rf_prob"
                ],
            )
        )

        xgb_auc = float(
            roc_auc_score(
                confirmed[
                    "strict_tnbc"
                ],
                confirmed[
                    "xgb_prob"
                ],
            )
        )

    agreement = float(
        confirmed[
            "original_label_match"
        ].mean()
    )

    return {
        "status":
            "completed",
        "selected_attributes":
            selected,
        "patients_with_complete_receptors":
            int(
                len(
                    confirmed
                )
            ),
        "strict_tnbc":
            strict_positive,
        "strict_non_tnbc":
            strict_negative,
        "original_vs_strict_label_agreement":
            agreement,
        "rf_auc_strict":
            rf_auc,
        "xgb_auc_strict":
            xgb_auc,
    }


# ============================================================
# PRINT CORRELATION SUMMARY
# ============================================================

def print_top_correlations(
    title: str,
    correlations: pd.DataFrame,
):
    print()
    print(title)
    print(
        "-" * 60
    )

    if correlations.empty:
        print(
            "No correlations available."
        )
        return

    ordered = correlations.sort_values(
        [
            "bh_adjusted_p",
            "p_value",
        ]
    )

    for _, row in ordered.iterrows():
        print(
            f"{row['sensitivity']} vs "
            f"{row['biological_measure']}"
        )

        print(
            f"  n = {int(row['n'])}"
        )

        print(
            f"  rho = "
            f"{row['spearman_rho']:.4f}"
        )

        print(
            f"  p = "
            f"{row['p_value']:.6g}"
        )

        print(
            f"  BH adjusted p = "
            f"{row['bh_adjusted_p']:.6g}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    check_files()

    a2_data, a2_correlations = (
        analyse_a2()
    )

    e2_data, e2_correlations = (
        analyse_e2()
    )

    metabric_summary = (
        analyse_metabric()
    )

    print()
    print(
        "Reviewer 1 biological validation"
    )
    print(
        "=" * 60
    )

    print(
        f"A2 matched patients: "
        f"{len(a2_data)}"
    )

    print(
        f"E2 matched patients: "
        f"{len(e2_data)}"
    )

    print_top_correlations(
        "TCGA-A2 biological correlations",
        a2_correlations,
    )

    print_top_correlations(
        "TCGA-E2 biological correlations",
        e2_correlations,
    )

    print()
    print(
        "METABRIC strict receptor analysis"
    )
    print(
        "-" * 60
    )

    for key, value in (
        metabric_summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    summary = {
        "predefined_marker_sets": {
            "ER_luminal":
                ER_LUMINAL_GENES,
            "basal":
                BASAL_GENES,
            "HER2":
                HER2_GENES,
        },
        "a2": {
            "matched_patients":
                int(
                    len(
                        a2_data
                    )
                ),
            "correlation_tests":
                int(
                    len(
                        a2_correlations
                    )
                ),
        },
        "e2": {
            "matched_patients":
                int(
                    len(
                        e2_data
                    )
                ),
            "correlation_tests":
                int(
                    len(
                        e2_correlations
                    )
                ),
        },
        "metabric":
            metabric_summary,
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
        f"A2 correlations:     "
        f"{A2_CORRELATIONS_FILE}"
    )

    print(
        f"E2 correlations:     "
        f"{E2_CORRELATIONS_FILE}"
    )

    print(
        f"A2 patient scores:   "
        f"{A2_PATIENT_FILE}"
    )

    print(
        f"E2 patient scores:   "
        f"{E2_PATIENT_FILE}"
    )

    print(
        f"METABRIC attributes: "
        f"{METABRIC_ATTRIBUTES_FILE}"
    )

    print(
        f"METABRIC clinical:   "
        f"{METABRIC_CLINICAL_FILE}"
    )

    print(
        f"METABRIC strict:     "
        f"{METABRIC_STRICT_FILE}"
    )

    print(
        f"Summary:             "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Log:                 "
        f"{LOG_FILE}"
    )


if __name__ == "__main__":
    main()