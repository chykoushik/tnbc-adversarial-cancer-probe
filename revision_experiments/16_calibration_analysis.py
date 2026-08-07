from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(
    r"E:\apply\journal publication\onco-probe"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "calibration_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PRIMARY_PRED_FILE = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "fusion_bootstrap_ci"
    / "fusion_loo_predictions.csv"
)

GATED_PRED_FILE = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "multimodal_gated_fusion"
    / "gated_multimodal_loo_predictions.csv"
)

ALT_PRED_FILE = (
    PROJECT_ROOT
    / "reviewer"
    / "output"
    / "reviewer2_remaining_analyses"
    / "alternative_fusion_predictions.csv"
)


METRICS_FILE = (
    OUTPUT_ROOT
    / "calibration_metrics.csv"
)

CURVE_DATA_FILE = (
    OUTPUT_ROOT
    / "calibration_curve_points.csv"
)

FIGURE_FILE = (
    OUTPUT_ROOT
    / "calibration_curves.png"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "calibration_summary.json"
)


N_BINS = 10
EPS = 1e-6


def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        y_prob,
        bin_edges[1:-1],
        right=True,
    )

    ece = 0.0

    for bin_id in range(
        n_bins
    ):
        mask = (
            bin_ids == bin_id
        )

        if not np.any(
            mask
        ):
            continue

        bin_prob = y_prob[
            mask
        ]

        bin_true = y_true[
            mask
        ]

        confidence = float(
            np.mean(
                bin_prob
            )
        )

        accuracy = float(
            np.mean(
                bin_true
            )
        )

        weight = (
            len(
                bin_prob
            )
            / len(
                y_prob
            )
        )

        ece += (
            weight
            * abs(
                confidence
                - accuracy
            )
        )

    return float(
        ece
    )


def calibration_intercept_slope(
    y_true,
    y_prob,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    clipped = np.clip(
        y_prob,
        EPS,
        1.0 - EPS,
    )

    logit = np.log(
        clipped
        / (
            1.0 - clipped
        )
    ).reshape(
        -1,
        1,
    )

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
    )

    try:
        model.fit(
            logit,
            y_true,
        )
    except Exception:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=10000,
        )

        model.fit(
            logit,
            y_true,
        )

    intercept = float(
        model.intercept_[0]
    )

    slope = float(
        model.coef_[0][0]
    )

    return (
        intercept,
        slope,
    )


def load_primary_models():
    frame = pd.read_csv(
        PRIMARY_PRED_FILE
    )

    output = []

    for model_name, group in (
        frame.groupby(
            "model"
        )
    ):
        output.append(
            {
                "model":
                    str(
                        model_name
                    ),
                "source":
                    "Primary RF/XGB fusion",
                "patient_id":
                    group[
                        "patient_id"
                    ].astype(
                        str
                    ).tolist(),
                "y_true":
                    group[
                        "true_label"
                    ].to_numpy(
                        dtype=int
                    ),
                "y_prob":
                    group[
                        "loo_probability"
                    ].to_numpy(
                        dtype=float
                    ),
            }
        )

    return output


def load_gated_models():
    frame = pd.read_csv(
        GATED_PRED_FILE
    )

    output = []

    for model_name, group in (
        frame.groupby(
            "model"
        )
    ):
        output.append(
            {
                "model":
                    str(
                        model_name
                    ),
                "source":
                    "Gated multimodal fusion",
                "patient_id":
                    group[
                        "patient_id"
                    ].astype(
                        str
                    ).tolist(),
                "y_true":
                    group[
                        "true_label"
                    ].to_numpy(
                        dtype=int
                    ),
                "y_prob":
                    group[
                        "loo_probability"
                    ].to_numpy(
                        dtype=float
                    ),
            }
        )

    return output


def load_alternative_models():
    frame = pd.read_csv(
        ALT_PRED_FILE
    )

    output = []

    grouped = frame.groupby(
        [
            "image_features",
            "fusion_model",
        ]
    )

    for (
        image_features,
        fusion_model,
    ), group in grouped:

        model_name = (
            f"{image_features} + "
            f"{fusion_model}"
        )

        output.append(
            {
                "model":
                    model_name,
                "source":
                    "Alternative fusion baseline",
                "patient_id":
                    group[
                        "patient_id"
                    ].astype(
                        str
                    ).tolist(),
                "y_true":
                    group[
                        "true_label"
                    ].to_numpy(
                        dtype=int
                    ),
                "y_prob":
                    group[
                        "loo_probability"
                    ].to_numpy(
                        dtype=float
                    ),
            }
        )

    return output


def compute_metrics(
    model_record,
):
    y_true = model_record[
        "y_true"
    ]

    y_prob = model_record[
        "y_prob"
    ]

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    ece = expected_calibration_error(
        y_true,
        y_prob,
        n_bins=N_BINS,
    )

    intercept, slope = (
        calibration_intercept_slope(
            y_true,
            y_prob,
        )
    )

    return {
        "model":
            model_record[
                "model"
            ],
        "source":
            model_record[
                "source"
            ],
        "patients":
            int(
                len(
                    y_true
                )
            ),
        "tnbc":
            int(
                np.sum(
                    y_true == 1
                )
            ),
        "non_tnbc":
            int(
                np.sum(
                    y_true == 0
                )
            ),
        "auc":
            float(
                auc
            ),
        "brier_score":
            float(
                brier
            ),
        "ece_10_bins":
            float(
                ece
            ),
        "calibration_intercept":
            float(
                intercept
            ),
        "calibration_slope":
            float(
                slope
            ),
    }


def compute_curve_points(
    model_record,
):
    y_true = model_record[
        "y_true"
    ]

    y_prob = model_record[
        "y_prob"
    ]

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            y_prob,
            n_bins=N_BINS,
            strategy="quantile",
        )
    )

    rows = []

    for i, (
        pred,
        observed,
    ) in enumerate(
        zip(
            mean_predicted,
            fraction_positive,
        ),
        start=1,
    ):
        rows.append(
            {
                "model":
                    model_record[
                        "model"
                    ],
                "source":
                    model_record[
                        "source"
                    ],
                "bin":
                    i,
                "mean_predicted_probability":
                    float(
                        pred
                    ),
                "observed_tnbc_fraction":
                    float(
                        observed
                    ),
            }
        )

    return rows


def make_figure(
    models,
):
    plt.figure(
        figsize=(
            8,
            7,
        )
    )

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
        label="Perfect calibration",
    )

    preferred_names = [
        "Adversarial-EfficientNet + RF",
        "Adversarial-EfficientNet + XGB",
        "EfficientNet + LogisticRegression",
        "EfficientNet + MLP",
        "EfficientNet + Gated Multimodal Fusion",
    ]

    for record in models:

        if (
            record[
                "model"
            ]
            not in preferred_names
        ):
            continue

        fraction_positive, mean_predicted = (
            calibration_curve(
                record[
                    "y_true"
                ],
                record[
                    "y_prob"
                ],
                n_bins=N_BINS,
                strategy="quantile",
            )
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=record[
                "model"
            ],
        )

    plt.xlabel(
        "Mean predicted TNBC probability"
    )

    plt.ylabel(
        "Observed TNBC fraction"
    )

    plt.title(
        "Calibration of multimodal fusion models"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    print()
    print(
        "Calibration analysis"
    )
    print(
        "=" * 60
    )

    for path in [
        PRIMARY_PRED_FILE,
        GATED_PRED_FILE,
        ALT_PRED_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    models = []

    models.extend(
        load_primary_models()
    )

    models.extend(
        load_gated_models()
    )

    models.extend(
        load_alternative_models()
    )

    metric_rows = []
    curve_rows = []

    for record in models:

        metrics = compute_metrics(
            record
        )

        metric_rows.append(
            metrics
        )

        curve_rows.extend(
            compute_curve_points(
                record
            )
        )

    metrics_frame = pd.DataFrame(
        metric_rows
    )

    metrics_frame = (
        metrics_frame
        .sort_values(
            "auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    curve_frame = pd.DataFrame(
        curve_rows
    )

    metrics_frame.to_csv(
        METRICS_FILE,
        index=False,
    )

    curve_frame.to_csv(
        CURVE_DATA_FILE,
        index=False,
    )

    make_figure(
        models
    )

    print()
    print(
        "Calibration metrics"
    )
    print(
        "-" * 100
    )

    print(
        metrics_frame.to_string(
            index=False
        )
    )

    summary = {
        "n_models":
            int(
                len(
                    metrics_frame
                )
            ),
        "patients_per_model":
            97,
        "tnbc_per_model":
            11,
        "calibration_metrics": [
            "Brier score",
            "Expected Calibration Error (10 bins)",
            "Calibration intercept",
            "Calibration slope",
        ],
        "figure_models": [
            "Adversarial-EfficientNet + RF",
            "Adversarial-EfficientNet + XGB",
            "EfficientNet + LogisticRegression",
            "EfficientNet + MLP",
            "EfficientNet + Gated Multimodal Fusion",
        ],
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
        f"Metrics: "
        f"{METRICS_FILE}"
    )

    print(
        f"Curve points: "
        f"{CURVE_DATA_FILE}"
    )

    print(
        f"Figure: "
        f"{FIGURE_FILE}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()