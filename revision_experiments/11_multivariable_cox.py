from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


CLINICAL_FILE = Path(
    r"E:\apply\journal publication\onco-probe\dataset"
    r"\TCGA-BRCA-A2-CLINI.xlsx"
)

SENSITIVITY_FILE = Path(
    r"E:\apply\journal publication\onco-probe\sensitivity_v3"
    r"\image_sensitivity_summary_v3.csv"
)

OUTPUT_DIR = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\multivariable_cox"
)

MERGED_CSV = OUTPUT_DIR / "cox_merged_cohort.csv"
RESULTS_CSV = OUTPUT_DIR / "multivariable_cox_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "multivariable_cox_summary.json"
LOG_FILE = OUTPUT_DIR / "multivariable_cox.log"

OS_FIGURE = OUTPUT_DIR / "os_multivariable_cox.png"
PFI_FIGURE = OUTPUT_DIR / "pfi_multivariable_cox.png"

PATIENT_COLUMN = "Sample ID"
AGE_COLUMN = "Diagnosis Age"

STAGE_COLUMN = (
    "Neoplasm Disease Stage American Joint Committee "
    "on Cancer Code"
)

OS_EVENT_COLUMN = "OS"
OS_TIME_COLUMN = "OS Time"

PFI_EVENT_COLUMN = "PFI"
PFI_TIME_COLUMN = "PFI Time"

PENALIZER = 0.10


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


def clean_patient_id(value: object) -> str:
    text = str(value).strip().upper()

    match = re.search(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2})",
        text,
    )

    if match:
        return match.group(1)

    return text


def convert_stage_to_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    text = text.replace("STAGE", "")
    text = text.replace(" ", "")
    text = text.replace("_", "")
    text = text.replace("-", "")

    if not text:
        return np.nan

    if text.startswith("IV"):
        return 1.0

    if text.startswith("III"):
        return 1.0

    if text.startswith("II"):
        return 0.0

    if text.startswith("I"):
        return 0.0

    return np.nan


def standardize_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    standard_deviation = numeric.std(ddof=1)

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation == 0
    ):
        raise ValueError(
            f"Cannot standardize column {series.name}."
        )

    return (
        numeric - numeric.mean()
    ) / standard_deviation


def load_and_merge_data() -> pd.DataFrame:
    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Clinical file not found: {CLINICAL_FILE}"
        )

    if not SENSITIVITY_FILE.exists():
        raise FileNotFoundError(
            f"Sensitivity file not found: {SENSITIVITY_FILE}"
        )

    clinical = pd.read_excel(
        CLINICAL_FILE
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILE
    )

    required_clinical_columns = [
        PATIENT_COLUMN,
        AGE_COLUMN,
        STAGE_COLUMN,
        OS_EVENT_COLUMN,
        OS_TIME_COLUMN,
        PFI_EVENT_COLUMN,
        PFI_TIME_COLUMN,
    ]

    missing_clinical = [
        column
        for column in required_clinical_columns
        if column not in clinical.columns
    ]

    if missing_clinical:
        raise ValueError(
            "Missing clinical columns: "
            f"{missing_clinical}"
        )

    required_sensitivity_columns = [
        "patient_id",
        "label",
        "fgsm_mean",
    ]

    missing_sensitivity = [
        column
        for column in required_sensitivity_columns
        if column not in sensitivity.columns
    ]

    if missing_sensitivity:
        raise ValueError(
            "Missing sensitivity columns: "
            f"{missing_sensitivity}"
        )

    clinical_subset = clinical[
        required_clinical_columns
    ].copy()

    clinical_subset["patient_id"] = (
        clinical_subset[PATIENT_COLUMN]
        .apply(clean_patient_id)
    )

    sensitivity["patient_id"] = (
        sensitivity["patient_id"]
        .apply(clean_patient_id)
    )

    clinical_subset = (
        clinical_subset
        .sort_values("patient_id")
        .drop_duplicates(
            subset="patient_id",
            keep="first",
        )
    )

    sensitivity = (
        sensitivity
        .sort_values("patient_id")
        .drop_duplicates(
            subset="patient_id",
            keep="first",
        )
    )

    merged = sensitivity.merge(
        clinical_subset,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    merged["age"] = pd.to_numeric(
        merged[AGE_COLUMN],
        errors="coerce",
    )

    merged["advanced_stage"] = (
        merged[STAGE_COLUMN]
        .apply(convert_stage_to_binary)
    )

    merged["tnbc"] = pd.to_numeric(
        merged["label"],
        errors="coerce",
    )

    merged["fgsm_mean"] = pd.to_numeric(
        merged["fgsm_mean"],
        errors="coerce",
    )

    merged["os_event"] = pd.to_numeric(
        merged[OS_EVENT_COLUMN],
        errors="coerce",
    )

    merged["os_time"] = pd.to_numeric(
        merged[OS_TIME_COLUMN],
        errors="coerce",
    )

    merged["pfi_event"] = pd.to_numeric(
        merged[PFI_EVENT_COLUMN],
        errors="coerce",
    )

    merged["pfi_time"] = pd.to_numeric(
        merged[PFI_TIME_COLUMN],
        errors="coerce",
    )

    merged["fgsm_z"] = standardize_column(
        merged["fgsm_mean"]
    )

    merged["age_z"] = standardize_column(
        merged["age"]
    )

    merged.to_csv(
        MERGED_CSV,
        index=False,
    )

    return merged


def fit_cox_model(
    merged: pd.DataFrame,
    endpoint_name: str,
    duration_column: str,
    event_column: str,
    figure_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model_columns = [
        duration_column,
        event_column,
        "fgsm_z",
        "age_z",
        "advanced_stage",
        "tnbc",
    ]

    analysis_data = merged[
        model_columns
    ].dropna().copy()

    analysis_data = analysis_data[
        analysis_data[duration_column] > 0
    ].copy()

    analysis_data[event_column] = (
        analysis_data[event_column]
        .astype(int)
    )

    patient_count = len(analysis_data)
    event_count = int(
        analysis_data[event_column].sum()
    )

    if patient_count < 20:
        raise RuntimeError(
            f"{endpoint_name} has only "
            f"{patient_count} complete patients."
        )

    if event_count < 5:
        raise RuntimeError(
            f"{endpoint_name} has only "
            f"{event_count} events."
        )

    logging.info(
        "%s model uses %d patients and %d events.",
        endpoint_name,
        patient_count,
        event_count,
    )

    cox = CoxPHFitter(
        penalizer=PENALIZER
    )

    cox.fit(
        analysis_data,
        duration_col=duration_column,
        event_col=event_column,
        robust=True,
    )

    summary = cox.summary.reset_index()

    covariate_column = (
        "covariate"
        if "covariate" in summary.columns
        else summary.columns[0]
    )

    summary = summary.rename(
        columns={
            covariate_column: "covariate",
            "coef": "coefficient",
            "exp(coef)": "hazard_ratio",
            "se(coef)": "standard_error",
            "p": "p_value",
            "exp(coef) lower 95%": "hr_ci95_lower",
            "exp(coef) upper 95%": "hr_ci95_upper",
        }
    )

    summary["endpoint"] = endpoint_name
    summary["patient_count"] = patient_count
    summary["event_count"] = event_count
    summary["penalizer"] = PENALIZER

    desired_columns = [
        "endpoint",
        "covariate",
        "coefficient",
        "hazard_ratio",
        "hr_ci95_lower",
        "hr_ci95_upper",
        "standard_error",
        "p_value",
        "patient_count",
        "event_count",
        "penalizer",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in summary.columns
    ]

    summary = summary[
        available_columns
    ]

    create_forest_plot(
        summary=summary,
        endpoint_name=endpoint_name,
        figure_path=figure_path,
    )

    metadata = {
        "endpoint": endpoint_name,
        "patient_count": patient_count,
        "event_count": event_count,
        "concordance_index": float(
            cox.concordance_index_
        ),
        "log_likelihood": float(
            cox.log_likelihood_
        ),
        "penalizer": PENALIZER,
    }

    return summary, metadata


def create_forest_plot(
    summary: pd.DataFrame,
    endpoint_name: str,
    figure_path: Path,
) -> None:
    plot_data = summary.copy()

    label_mapping = {
        "fgsm_z": "FGSM sensitivity",
        "age_z": "Age",
        "advanced_stage": "Stage III or IV",
        "tnbc": "TNBC status",
    }

    plot_data["display_name"] = (
        plot_data["covariate"]
        .map(label_mapping)
        .fillna(plot_data["covariate"])
    )

    plot_data = plot_data.iloc[::-1].reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(plot_data)
    )

    hazard_ratios = plot_data[
        "hazard_ratio"
    ].to_numpy()

    lower_bounds = plot_data[
        "hr_ci95_lower"
    ].to_numpy()

    upper_bounds = plot_data[
        "hr_ci95_upper"
    ].to_numpy()

    errors = np.vstack(
        [
            hazard_ratios - lower_bounds,
            upper_bounds - hazard_ratios,
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7, 4.5)
    )

    axis.errorbar(
        hazard_ratios,
        y_positions,
        xerr=errors,
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_xscale("log")

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        plot_data["display_name"]
    )

    axis.set_xlabel(
        "Hazard ratio with 95% confidence interval"
    )

    axis.set_title(
        f"{endpoint_name} multivariable Cox model"
    )

    figure.tight_layout()

    figure.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_model_results(
    summary: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    print()
    print(metadata["endpoint"])
    print("----------------------------------------")
    print(
        f"Patients: {metadata['patient_count']}"
    )
    print(
        f"Events:   {metadata['event_count']}"
    )
    print(
        "Concordance index: "
        f"{metadata['concordance_index']:.3f}"
    )

    for _, row in summary.iterrows():
        print()
        print(row["covariate"])
        print(
            f"  HR: {row['hazard_ratio']:.3f}"
        )
        print(
            "  95% CI: "
            f"{row['hr_ci95_lower']:.3f} to "
            f"{row['hr_ci95_upper']:.3f}"
        )
        print(
            f"  p: {row['p_value']:.6g}"
        )


def main() -> None:
    configure_logging()

    merged = load_and_merge_data()

    print()
    print("Merged clinical cohort")
    print("----------------------------------------")
    print(
        f"Sensitivity patients: "
        f"{pd.read_csv(SENSITIVITY_FILE)['patient_id'].nunique()}"
    )
    print(
        f"Matched patients:     "
        f"{merged['patient_id'].nunique()}"
    )
    print(
        f"Age available:        "
        f"{merged['age'].notna().sum()}"
    )
    print(
        f"Stage available:      "
        f"{merged['advanced_stage'].notna().sum()}"
    )
    print(
        f"OS available:         "
        f"{merged[['os_time', 'os_event']].dropna().shape[0]}"
    )
    print(
        f"PFI available:        "
        f"{merged[['pfi_time', 'pfi_event']].dropna().shape[0]}"
    )

    all_results = []
    metadata_records = []

    os_results, os_metadata = fit_cox_model(
        merged=merged,
        endpoint_name="Overall survival",
        duration_column="os_time",
        event_column="os_event",
        figure_path=OS_FIGURE,
    )

    all_results.append(
        os_results
    )

    metadata_records.append(
        os_metadata
    )

    pfi_results, pfi_metadata = fit_cox_model(
        merged=merged,
        endpoint_name="Progression-free interval",
        duration_column="pfi_time",
        event_column="pfi_event",
        figure_path=PFI_FIGURE,
    )

    all_results.append(
        pfi_results
    )

    metadata_records.append(
        pfi_metadata
    )

    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            {
                "clinical_file": str(
                    CLINICAL_FILE
                ),
                "sensitivity_file": str(
                    SENSITIVITY_FILE
                ),
                "stage_definition": (
                    "Stage I or II equals 0. "
                    "Stage III or IV equals 1."
                ),
                "continuous_variables": (
                    "FGSM sensitivity and age were "
                    "standardized before model fitting."
                ),
                "models": metadata_records,
            },
            file_handle,
            indent=2,
        )

    print_model_results(
        os_results,
        os_metadata,
    )

    print_model_results(
        pfi_results,
        pfi_metadata,
    )

    print()
    print("Multivariable Cox analysis finished")
    print("----------------------------------------")
    print(f"Merged cohort: {MERGED_CSV}")
    print(f"Results:       {RESULTS_CSV}")
    print(f"OS figure:     {OS_FIGURE}")
    print(f"PFI figure:    {PFI_FIGURE}")
    print(f"Summary:       {SUMMARY_JSON}")
    print(f"Log file:      {LOG_FILE}")


if __name__ == "__main__":
    main()