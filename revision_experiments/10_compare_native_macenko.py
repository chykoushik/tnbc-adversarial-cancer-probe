from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


NATIVE_RESULTS = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_adversarial_validation"
    r"\cptac_patient_results.csv"
)

MACENKO_RESULTS = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_adversarial_validation_macenko"
    r"\cptac_patient_results.csv"
)

OUTPUT_DIR = Path(
    r"E:\apply\journal publication\onco-probe\reviewer"
    r"\output\cptac_native_vs_macenko"
)

OUTPUT_CSV = OUTPUT_DIR / "native_vs_macenko_comparison.csv"


METRICS = [
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


def bootstrap_mean_difference(
    differences: np.ndarray,
    iterations: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if differences.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    bootstrap_values = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(iterations):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_values[index] = np.mean(sample)

    lower = np.percentile(
        bootstrap_values,
        2.5,
    )

    upper = np.percentile(
        bootstrap_values,
        97.5,
    )

    return float(lower), float(upper)


def paired_rank_biserial(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    absolute_values = np.abs(differences)

    ranks = pd.Series(
        absolute_values
    ).rank(
        method="average"
    ).to_numpy()

    positive_sum = ranks[
        differences > 0
    ].sum()

    negative_sum = ranks[
        differences < 0
    ].sum()

    denominator = positive_sum + negative_sum

    if denominator == 0:
        return 0.0

    return float(
        (positive_sum - negative_sum)
        / denominator
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    native = pd.read_csv(
        NATIVE_RESULTS
    )

    macenko = pd.read_csv(
        MACENKO_RESULTS
    )

    paired = native.merge(
        macenko,
        on=[
            "patient_id",
            "model",
        ],
        suffixes=(
            "_native",
            "_macenko",
        ),
        validate="one_to_one",
    )

    comparison_rows = []

    for model_name, model_data in paired.groupby(
        "model"
    ):
        for metric_index, metric in enumerate(
            METRICS
        ):
            native_values = pd.to_numeric(
                model_data[
                    f"{metric}_native"
                ],
                errors="coerce",
            ).to_numpy()

            macenko_values = pd.to_numeric(
                model_data[
                    f"{metric}_macenko"
                ],
                errors="coerce",
            ).to_numpy()

            valid = (
                np.isfinite(native_values)
                & np.isfinite(macenko_values)
            )

            native_values = native_values[
                valid
            ]

            macenko_values = macenko_values[
                valid
            ]

            differences = (
                macenko_values
                - native_values
            )

            if len(differences) == 0:
                statistic = float("nan")
                p_value = float("nan")
            elif np.allclose(
                differences,
                0,
            ):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    macenko_values,
                    native_values,
                    alternative="two-sided",
                )

                statistic = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )

            ci_lower, ci_upper = (
                bootstrap_mean_difference(
                    differences,
                    seed=2026 + metric_index,
                )
            )

            comparison_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "patient_count": len(
                        differences
                    ),
                    "native_mean": float(
                        np.mean(native_values)
                    ),
                    "macenko_mean": float(
                        np.mean(macenko_values)
                    ),
                    "mean_difference_macenko_minus_native": float(
                        np.mean(differences)
                    ),
                    "difference_ci95_lower": ci_lower,
                    "difference_ci95_upper": ci_upper,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "rank_biserial_correlation": paired_rank_biserial(
                        differences
                    ),
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("Native versus Macenko comparison")
    print("----------------------------------------")

    important_metrics = comparison[
        comparison["metric"].isin(
            [
                "tnbc_probability_mean",
                "tnbc_tile_fraction_at_0_5",
                "fgsm_mean",
                "pgd_mean",
                "fgsm_pgd_spearman_rho",
            ]
        )
    ]

    for _, row in important_metrics.iterrows():
        print()
        print(
            f"{row['model']} | "
            f"{row['metric']}"
        )

        print(
            f"  Native mean:  "
            f"{row['native_mean']:.6f}"
        )

        print(
            f"  Macenko mean: "
            f"{row['macenko_mean']:.6f}"
        )

        print(
            "  Difference:   "
            f"{row['mean_difference_macenko_minus_native']:.6f}"
        )

        print(
            "  95% CI:       "
            f"{row['difference_ci95_lower']:.6f} to "
            f"{row['difference_ci95_upper']:.6f}"
        )

        print(
            f"  Wilcoxon p:   "
            f"{row['wilcoxon_p_value']:.6g}"
        )

    print()
    print(
        f"Saved comparison: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()