from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_adversarial_validation_macenko"
)

PATIENT_RESULTS_CSV = INPUT_ROOT / "cptac_patient_results.csv"
TILE_RESULTS_CSV = INPUT_ROOT / "cptac_tile_results.csv"
MODEL_AGREEMENT_CSV = INPUT_ROOT / "cptac_model_agreement.csv"

OUTPUT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_statistical_summary_macenko"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"

MODEL_SUMMARY_CSV = OUTPUT_ROOT / "model_summary.csv"
PATIENT_CLASSIFICATION_CSV = OUTPUT_ROOT / "patient_classification.csv"
CROSS_MODEL_CORRELATIONS_CSV = (
    OUTPUT_ROOT / "cross_model_correlations.csv"
)
FGSM_PGD_SUMMARY_CSV = OUTPUT_ROOT / "fgsm_pgd_summary.csv"
PAIRED_MODEL_TESTS_CSV = OUTPUT_ROOT / "paired_model_tests.csv"
TILE_SUMMARY_CSV = OUTPUT_ROOT / "tile_level_summary.csv"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
LOG_FILE = OUTPUT_ROOT / "cptac_summary.log"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BOOTSTRAP_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
CLASSIFICATION_THRESHOLD = 0.50
RANDOM_SEED = 2026

MODEL_DISPLAY_NAMES = {
    "resnet50_ts": "ResNet50-TS",
    "efficientnet_b0_ts": "EfficientNet-B0-TS",
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_inputs() -> None:
    required_files = [
        PATIENT_RESULTS_CSV,
        TILE_RESULTS_CSV,
        MODEL_AGREEMENT_CSV,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(str(path) for path in missing)

        raise FileNotFoundError(
            f"Required result files are missing:\n{missing_text}"
        )


def validate_patient_results(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "patient_id",
        "model",
        "tnbc_probability_mean",
        "tnbc_probability_median",
        "tnbc_probability_std",
        "tnbc_tile_fraction_at_0_5",
        "fgsm_mean",
        "fgsm_max",
        "fgsm_std",
        "fgsm_p75",
        "fgsm_p90",
        "pgd_mean",
        "pgd_max",
        "pgd_std",
        "pgd_p75",
        "pgd_p90",
        "fgsm_pgd_spearman_rho",
    }

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "Patient-results file is missing columns: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------

def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]

    if clean_values.size == 0:
        return float("nan"), float("nan")

    if clean_values.size == 1:
        value = float(statistic(clean_values))
        return value, value

    rng = np.random.default_rng(seed)

    bootstrap_statistics = np.empty(
        iterations,
        dtype=np.float64,
    )

    sample_size = clean_values.size

    for index in range(iterations):
        bootstrap_sample = rng.choice(
            clean_values,
            size=sample_size,
            replace=True,
        )

        bootstrap_statistics[index] = statistic(
            bootstrap_sample
        )

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_statistics,
        100 * alpha / 2,
    )

    upper = np.percentile(
        bootstrap_statistics,
        100 * (1 - alpha / 2),
    )

    return float(lower), float(upper)


def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    if confidence_level == 0.95:
        z_value = 1.959963984540054
    else:
        raise ValueError(
            "This implementation currently supports 95% intervals."
        )

    proportion = successes / total
    denominator = 1 + (z_value ** 2 / total)

    centre = (
        proportion
        + z_value ** 2 / (2 * total)
    ) / denominator

    margin = (
        z_value
        * np.sqrt(
            (
                proportion * (1 - proportion) / total
                + z_value ** 2 / (4 * total ** 2)
            )
        )
        / denominator
    )

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return float(lower), float(upper)


def safe_spearman(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    valid = np.isfinite(first) & np.isfinite(second)

    first = first[valid]
    second = second[valid]

    if first.size < 3:
        return float("nan"), float("nan")

    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan"), float("nan")

    result = spearmanr(first, second)

    return float(result.statistic), float(result.pvalue)


def paired_wilcoxon(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    valid = np.isfinite(first) & np.isfinite(second)

    first = first[valid]
    second = second[valid]

    if first.size == 0:
        return float("nan"), float("nan")

    differences = first - second

    if np.allclose(differences, 0):
        return 0.0, 1.0

    result = wilcoxon(
        first,
        second,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return float(result.statistic), float(result.pvalue)


def rank_biserial_from_paired_differences(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    differences = np.asarray(first, dtype=float) - np.asarray(
        second,
        dtype=float,
    )

    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]

    if differences.size == 0:
        return 0.0

    absolute_differences = np.abs(differences)
    ranks = pd.Series(absolute_differences).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = ranks[differences > 0].sum()
    negative_rank_sum = ranks[differences < 0].sum()

    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0:
        return 0.0

    return float(
        (positive_rank_sum - negative_rank_sum)
        / denominator
    )


# ---------------------------------------------------------------------
# Model summaries
# ---------------------------------------------------------------------

def summarize_models(
    patient_results: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    continuous_metrics = [
        "tnbc_probability_mean",
        "tnbc_probability_median",
        "tnbc_tile_fraction_at_0_5",
        "fgsm_mean",
        "fgsm_max",
        "fgsm_std",
        "fgsm_p75",
        "fgsm_p90",
        "pgd_mean",
        "pgd_max",
        "pgd_std",
        "pgd_p75",
        "pgd_p90",
        "fgsm_pgd_spearman_rho",
    ]

    for model_name, model_data in patient_results.groupby("model"):
        row: dict[str, Any] = {
            "model": model_name,
            "model_display_name": MODEL_DISPLAY_NAMES.get(
                model_name,
                model_name,
            ),
            "patient_count": int(
                model_data["patient_id"].nunique()
            ),
        }

        for metric_index, metric in enumerate(continuous_metrics):
            values = pd.to_numeric(
                model_data[metric],
                errors="coerce",
            ).dropna().to_numpy()

            lower, upper = bootstrap_confidence_interval(
                values=values,
                statistic=np.mean,
                seed=RANDOM_SEED + metric_index,
            )

            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper

        patient_predictions = (
            model_data["tnbc_probability_mean"]
            >= CLASSIFICATION_THRESHOLD
        )

        positive_patients = int(patient_predictions.sum())
        total_patients = int(len(patient_predictions))

        sensitivity = positive_patients / total_patients

        sensitivity_lower, sensitivity_upper = wilson_interval(
            successes=positive_patients,
            total=total_patients,
        )

        row.update(
            {
                "patient_level_true_positive_count": positive_patients,
                "patient_level_total_count": total_patients,
                "patient_level_sensitivity": sensitivity,
                "patient_level_sensitivity_ci95_lower": (
                    sensitivity_lower
                ),
                "patient_level_sensitivity_ci95_upper": (
                    sensitivity_upper
                ),
            }
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MODEL_SUMMARY_CSV, index=False)

    return summary


# ---------------------------------------------------------------------
# Patient classifications
# ---------------------------------------------------------------------

def create_patient_classification_table(
    patient_results: pd.DataFrame,
) -> pd.DataFrame:
    classification = patient_results[
        [
            "patient_id",
            "model",
            "tnbc_probability_mean",
            "tnbc_probability_median",
            "tnbc_tile_fraction_at_0_5",
            "fgsm_mean",
            "pgd_mean",
        ]
    ].copy()

    classification["predicted_tnbc"] = (
        classification["tnbc_probability_mean"]
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    classification["correct_positive_classification"] = (
        classification["predicted_tnbc"] == 1
    ).astype(int)

    classification.to_csv(
        PATIENT_CLASSIFICATION_CSV,
        index=False,
    )

    return classification


# ---------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------

def calculate_cross_model_correlations(
    patient_results: pd.DataFrame,
) -> pd.DataFrame:
    models = sorted(patient_results["model"].unique())

    if len(models) != 2:
        raise RuntimeError(
            "Exactly two models are required for paired comparison. "
            f"Found: {models}"
        )

    first_model, second_model = models

    paired = patient_results.pivot(
        index="patient_id",
        columns="model",
    )

    metrics = [
        "tnbc_probability_mean",
        "tnbc_tile_fraction_at_0_5",
        "fgsm_mean",
        "fgsm_max",
        "fgsm_std",
        "fgsm_p75",
        "fgsm_p90",
        "pgd_mean",
        "pgd_max",
        "pgd_std",
        "pgd_p75",
        "pgd_p90",
        "fgsm_pgd_spearman_rho",
    ]

    rows: list[dict[str, Any]] = []

    for metric in metrics:
        first_values = paired[(metric, first_model)].to_numpy()
        second_values = paired[(metric, second_model)].to_numpy()

        rho, p_value = safe_spearman(
            first_values,
            second_values,
        )

        rows.append(
            {
                "metric": metric,
                "first_model": first_model,
                "second_model": second_model,
                "patient_count": int(
                    np.sum(
                        np.isfinite(first_values)
                        & np.isfinite(second_values)
                    )
                ),
                "spearman_rho": rho,
                "spearman_p_value": p_value,
            }
        )

    correlation_results = pd.DataFrame(rows)
    correlation_results.to_csv(
        CROSS_MODEL_CORRELATIONS_CSV,
        index=False,
    )

    return correlation_results


# ---------------------------------------------------------------------
# Paired model comparisons
# ---------------------------------------------------------------------

def calculate_paired_model_tests(
    patient_results: pd.DataFrame,
) -> pd.DataFrame:
    models = sorted(patient_results["model"].unique())
    first_model, second_model = models

    paired = patient_results.pivot(
        index="patient_id",
        columns="model",
    )

    metrics = [
        "tnbc_probability_mean",
        "tnbc_tile_fraction_at_0_5",
        "fgsm_mean",
        "fgsm_max",
        "fgsm_std",
        "fgsm_p75",
        "fgsm_p90",
        "pgd_mean",
        "pgd_max",
        "pgd_std",
        "pgd_p75",
        "pgd_p90",
    ]

    rows: list[dict[str, Any]] = []

    for metric in metrics:
        first_values = paired[(metric, first_model)].to_numpy()
        second_values = paired[(metric, second_model)].to_numpy()

        statistic, p_value = paired_wilcoxon(
            first_values,
            second_values,
        )

        effect_size = rank_biserial_from_paired_differences(
            first_values,
            second_values,
        )

        differences = first_values - second_values
        differences = differences[np.isfinite(differences)]

        lower, upper = bootstrap_confidence_interval(
            differences,
            statistic=np.mean,
            seed=RANDOM_SEED + len(rows),
        )

        rows.append(
            {
                "metric": metric,
                "first_model": first_model,
                "second_model": second_model,
                "first_model_mean": float(
                    np.nanmean(first_values)
                ),
                "second_model_mean": float(
                    np.nanmean(second_values)
                ),
                "mean_paired_difference": float(
                    np.nanmean(differences)
                ),
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "rank_biserial_correlation": effect_size,
            }
        )

    tests = pd.DataFrame(rows)
    tests.to_csv(PAIRED_MODEL_TESTS_CSV, index=False)

    return tests


# ---------------------------------------------------------------------
# FGSM versus PGD summary
# ---------------------------------------------------------------------

def summarize_fgsm_pgd_agreement(
    patient_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, model_data in patient_results.groupby("model"):
        values = pd.to_numeric(
            model_data["fgsm_pgd_spearman_rho"],
            errors="coerce",
        ).dropna().to_numpy()

        lower, upper = bootstrap_confidence_interval(
            values=values,
            statistic=np.mean,
            seed=RANDOM_SEED,
        )

        rows.append(
            {
                "model": model_name,
                "patient_count": int(len(values)),
                "fgsm_pgd_rho_mean": float(np.mean(values)),
                "fgsm_pgd_rho_median": float(np.median(values)),
                "fgsm_pgd_rho_std": float(
                    np.std(values, ddof=1)
                ),
                "fgsm_pgd_rho_ci95_lower": lower,
                "fgsm_pgd_rho_ci95_upper": upper,
                "patients_with_positive_correlation": int(
                    np.sum(values > 0)
                ),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(FGSM_PGD_SUMMARY_CSV, index=False)

    return results


# ---------------------------------------------------------------------
# Tile-level summary
# ---------------------------------------------------------------------

def summarize_tiles(
    tile_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, model_data in tile_results.groupby("model"):
        probabilities = pd.to_numeric(
            model_data["tnbc_probability"],
            errors="coerce",
        ).dropna().to_numpy()

        positive_tiles = int(
            np.sum(
                probabilities >= CLASSIFICATION_THRESHOLD
            )
        )

        total_tiles = int(len(probabilities))

        fraction = positive_tiles / total_tiles

        lower, upper = wilson_interval(
            successes=positive_tiles,
            total=total_tiles,
        )

        rows.append(
            {
                "model": model_name,
                "tile_count": total_tiles,
                "mean_tnbc_probability": float(
                    np.mean(probabilities)
                ),
                "median_tnbc_probability": float(
                    np.median(probabilities)
                ),
                "positive_tile_count": positive_tiles,
                "positive_tile_fraction": fraction,
                "positive_tile_fraction_ci95_lower": lower,
                "positive_tile_fraction_ci95_upper": upper,
                "fgsm_mean_across_tiles": float(
                    model_data["fgsm_mean"].mean()
                ),
                "pgd_mean_across_tiles": float(
                    model_data["pgd_mean"].mean()
                ),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(TILE_SUMMARY_CSV, index=False)

    return results


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def create_probability_boxplot(
    patient_results: pd.DataFrame,
) -> None:
    models = sorted(patient_results["model"].unique())

    values = [
        patient_results.loc[
            patient_results["model"] == model,
            "tnbc_probability_mean",
        ].to_numpy()
        for model in models
    ]

    labels = [
        MODEL_DISPLAY_NAMES.get(model, model)
        for model in models
    ]

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
    )

    axis.axhline(
        CLASSIFICATION_THRESHOLD,
        linestyle="--",
        linewidth=1,
    )

    axis.set_ylabel("Mean patient-level TNBC probability")
    axis.set_title("CPTAC-BRCA TNBC probability by model")

    figure.tight_layout()
    figure.savefig(
        FIGURE_ROOT / "patient_tnbc_probability_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_model_agreement_scatter(
    patient_results: pd.DataFrame,
) -> None:
    models = sorted(patient_results["model"].unique())
    first_model, second_model = models

    pivoted = patient_results.pivot(
        index="patient_id",
        columns="model",
        values="fgsm_mean",
    )

    x_values = pivoted[first_model].to_numpy()
    y_values = pivoted[second_model].to_numpy()

    rho, p_value = safe_spearman(
        x_values,
        y_values,
    )

    figure, axis = plt.subplots(figsize=(6, 6))

    axis.scatter(
        x_values,
        y_values,
        alpha=0.8,
    )

    axis.set_xlabel(
        f"{MODEL_DISPLAY_NAMES.get(first_model, first_model)} "
        "FGSM mean"
    )

    axis.set_ylabel(
        f"{MODEL_DISPLAY_NAMES.get(second_model, second_model)} "
        "FGSM mean"
    )

    axis.set_title(
        "Cross-model patient-level FGSM agreement\n"
        f"Spearman rho = {rho:.3f}, p = {p_value:.3g}"
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_ROOT / "fgsm_cross_model_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_probability_patient_plot(
    patient_results: pd.DataFrame,
) -> None:
    pivoted = patient_results.pivot(
        index="patient_id",
        columns="model",
        values="tnbc_probability_mean",
    ).sort_index()

    figure, axis = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(len(pivoted))

    for model_name in pivoted.columns:
        axis.plot(
            x_positions,
            pivoted[model_name].to_numpy(),
            marker="o",
            linewidth=1.5,
            label=MODEL_DISPLAY_NAMES.get(
                model_name,
                model_name,
            ),
        )

    axis.axhline(
        CLASSIFICATION_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label="TNBC threshold",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        pivoted.index,
        rotation=90,
    )

    axis.set_ylabel("Mean patient-level TNBC probability")
    axis.set_xlabel("CPTAC-BRCA patient")
    axis.set_title(
        "Patient-level predictions in the CPTAC-BRCA TNBC cohort"
    )

    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_ROOT / "patient_probability_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_fgsm_pgd_scatter(
    patient_results: pd.DataFrame,
) -> None:
    for model_name, model_data in patient_results.groupby("model"):
        x_values = model_data["fgsm_mean"].to_numpy()
        y_values = model_data["pgd_mean"].to_numpy()

        rho, p_value = safe_spearman(
            x_values,
            y_values,
        )

        figure, axis = plt.subplots(figsize=(6, 6))

        axis.scatter(
            x_values,
            y_values,
            alpha=0.8,
        )

        axis.set_xlabel("Patient-level FGSM mean sensitivity")
        axis.set_ylabel("Patient-level PGD mean sensitivity")

        axis.set_title(
            f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\n"
            f"FGSM-PGD agreement: rho = {rho:.3f}, "
            f"p = {p_value:.3g}"
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_ROOT
            / f"{model_name}_fgsm_pgd_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


# ---------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------

def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    clean_dataframe = dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return json.loads(
        clean_dataframe.to_json(
            orient="records"
        )
    )


def save_summary_json(
    model_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    fgsm_pgd_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
    tile_summary: pd.DataFrame,
) -> None:
    summary = {
        "dataset": "CPTAC-BRCA",
        "cohort": "TNBC-positive patients only",
        "important_limitation": (
            "ROC AUC, specificity and TNBC-versus-non-TNBC "
            "classification performance cannot be estimated because "
            "this validation subset contains no negative-class patients."
        ),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "confidence_level": CONFIDENCE_LEVEL,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_summary": dataframe_to_records(model_summary),
        "cross_model_correlations": dataframe_to_records(
            correlations
        ),
        "fgsm_pgd_summary": dataframe_to_records(
            fgsm_pgd_summary
        ),
        "paired_model_tests": dataframe_to_records(
            paired_tests
        ),
        "tile_summary": dataframe_to_records(tile_summary),
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            summary,
            file_handle,
            indent=2,
        )


# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------

def print_terminal_report(
    model_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    fgsm_pgd_summary: pd.DataFrame,
) -> None:
    print()
    print("CPTAC-BRCA statistical summary")
    print("----------------------------------------")
    print(
        "Important: this validation cohort contains TNBC-positive "
        "patients only. ROC AUC and specificity are not estimable."
    )
    print()

    for _, row in model_summary.iterrows():
        print(row["model_display_name"])
        print(
            "  Patient-level TNBC sensitivity: "
            f"{row['patient_level_sensitivity']:.3f} "
            f"(95% CI "
            f"{row['patient_level_sensitivity_ci95_lower']:.3f}-"
            f"{row['patient_level_sensitivity_ci95_upper']:.3f})"
        )

        print(
            "  Mean patient TNBC probability: "
            f"{row['tnbc_probability_mean_mean']:.3f} "
            f"(bootstrap 95% CI "
            f"{row['tnbc_probability_mean_ci95_lower']:.3f}-"
            f"{row['tnbc_probability_mean_ci95_upper']:.3f})"
        )

        print(
            "  Mean FGSM sensitivity: "
            f"{row['fgsm_mean_mean']:.6f} "
            f"(bootstrap 95% CI "
            f"{row['fgsm_mean_ci95_lower']:.6f}-"
            f"{row['fgsm_mean_ci95_upper']:.6f})"
        )

        print()

    fgsm_correlation = correlations.loc[
        correlations["metric"] == "fgsm_mean"
    ]

    if not fgsm_correlation.empty:
        result = fgsm_correlation.iloc[0]

        print(
            "Cross-model patient-level FGSM correlation: "
            f"rho = {result['spearman_rho']:.3f}, "
            f"p = {result['spearman_p_value']:.3g}"
        )

    for _, row in fgsm_pgd_summary.iterrows():
        print(
            f"{MODEL_DISPLAY_NAMES.get(row['model'], row['model'])} "
            "mean within-patient FGSM-PGD correlation: "
            f"{row['fgsm_pgd_rho_mean']:.3f} "
            f"(bootstrap 95% CI "
            f"{row['fgsm_pgd_rho_ci95_lower']:.3f}-"
            f"{row['fgsm_pgd_rho_ci95_upper']:.3f})"
        )

    print()
    print(f"Model summary:       {MODEL_SUMMARY_CSV}")
    print(f"Patient results:     {PATIENT_CLASSIFICATION_CSV}")
    print(f"Model correlations:  {CROSS_MODEL_CORRELATIONS_CSV}")
    print(f"FGSM-PGD summary:    {FGSM_PGD_SUMMARY_CSV}")
    print(f"Paired tests:        {PAIRED_MODEL_TESTS_CSV}")
    print(f"Tile summary:        {TILE_SUMMARY_CSV}")
    print(f"Figures:             {FIGURE_ROOT}")
    print(f"JSON summary:        {SUMMARY_JSON}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    configure_logging()
    validate_inputs()

    patient_results = pd.read_csv(PATIENT_RESULTS_CSV)
    tile_results = pd.read_csv(TILE_RESULTS_CSV)
    model_agreement = pd.read_csv(MODEL_AGREEMENT_CSV)

    validate_patient_results(patient_results)

    logging.info(
        "Loaded %d patient-model rows.",
        len(patient_results),
    )

    logging.info(
        "Loaded %d tile-model rows.",
        len(tile_results),
    )

    logging.info(
        "Loaded agreement data for %d patients.",
        model_agreement["patient_id"].nunique(),
    )

    expected_patient_model_rows = (
        patient_results["patient_id"].nunique()
        * patient_results["model"].nunique()
    )

    if len(patient_results) != expected_patient_model_rows:
        raise RuntimeError(
            "Patient-results table is incomplete. "
            f"Expected {expected_patient_model_rows} rows but "
            f"found {len(patient_results)}."
        )

    model_summary = summarize_models(patient_results)

    create_patient_classification_table(patient_results)

    correlations = calculate_cross_model_correlations(
        patient_results
    )

    paired_tests = calculate_paired_model_tests(
        patient_results
    )

    fgsm_pgd_summary = summarize_fgsm_pgd_agreement(
        patient_results
    )

    tile_summary = summarize_tiles(tile_results)

    create_probability_boxplot(patient_results)
    create_model_agreement_scatter(patient_results)
    create_probability_patient_plot(patient_results)
    create_fgsm_pgd_scatter(patient_results)

    save_summary_json(
        model_summary=model_summary,
        correlations=correlations,
        fgsm_pgd_summary=fgsm_pgd_summary,
        paired_tests=paired_tests,
        tile_summary=tile_summary,
    )

    print_terminal_report(
        model_summary=model_summary,
        correlations=correlations,
        fgsm_pgd_summary=fgsm_pgd_summary,
    )


if __name__ == "__main__":
    main()