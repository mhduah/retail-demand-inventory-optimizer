"""Select an adaptive forecasting baseline using rolling backtesting.

For every product-store series, five simple forecasting rules are
evaluated over four historical 28-day windows. The model with the
lowest backtest MAE is selected before evaluation on the untouched
final 28-day validation period.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_inventory.db"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

BASELINE_OVERALL_FILE = (
    REPORT_DIRECTORY
    / "baseline_metrics_overall.csv"
)

BASELINE_STORE_FILE = (
    REPORT_DIRECTORY
    / "baseline_metrics_by_store.csv"
)

FINAL_HORIZON = 28
BACKTEST_HORIZON = 28
BACKTEST_WINDOWS = 4

EXPECTED_SERIES = 150
EXPECTED_DAYS = 1_941

MODEL_NAMES = [
    "zero_forecast",
    "last_value_naive",
    "weekly_seasonal_naive",
    "mean_7_days",
    "mean_28_days",
]

MODEL_PRIORITY = {
    model_name: priority
    for priority, model_name in enumerate(MODEL_NAMES)
}


def load_sales() -> pd.DataFrame:
    """Load the complete sales panel in chronological order."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DATABASE_FILE}"
        )

    query = """
        SELECT
            s.store_id,
            s.item_id,
            s.day_id,
            c.date,
            s.units_sold
        FROM fact_sales AS s
        INNER JOIN dim_calendar AS c
            ON s.day_id = c.day_id
        ORDER BY
            s.store_id,
            s.item_id,
            c.date;
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        sales = pd.read_sql_query(
            query,
            connection,
            parse_dates=["date"],
        )

    return sales


def validate_sales(sales: pd.DataFrame) -> None:
    """Validate the panel before model selection."""

    required_columns = {
        "store_id",
        "item_id",
        "day_id",
        "date",
        "units_sold",
    }

    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    series_sizes = (
        sales.groupby(["store_id", "item_id"])
        .size()
    )

    if len(series_sizes) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} series, "
            f"but found {len(series_sizes)}."
        )

    if not series_sizes.eq(EXPECTED_DAYS).all():
        raise ValueError(
            f"Every series must contain {EXPECTED_DAYS} days."
        )

    if sales["units_sold"].isna().any():
        raise ValueError("Missing sales observations found.")

    if (sales["units_sold"] < 0).any():
        raise ValueError("Negative sales observations found.")


def generate_forecasts(
    training_values: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Generate forecasts from all candidate baseline models."""

    if len(training_values) < 28:
        raise ValueError(
            "At least 28 training observations are required."
        )

    last_value = float(training_values[-1])

    weekly_pattern = training_values[-7:]

    mean_7 = float(
        np.mean(training_values[-7:])
    )

    mean_28 = float(
        np.mean(training_values[-28:])
    )

    forecasts = {
        "zero_forecast": np.zeros(
            horizon,
            dtype=float,
        ),
        "last_value_naive": np.full(
            horizon,
            last_value,
            dtype=float,
        ),
        "weekly_seasonal_naive": np.resize(
            weekly_pattern,
            horizon,
        ).astype(float),
        "mean_7_days": np.full(
            horizon,
            mean_7,
            dtype=float,
        ),
        "mean_28_days": np.full(
            horizon,
            mean_28,
            dtype=float,
        ),
    }

    return forecasts


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate standard forecast-error metrics."""

    actual = actual.astype(float)
    predicted = predicted.astype(float)

    errors = predicted - actual

    absolute_error = np.abs(errors)

    actual_total = np.sum(
        np.abs(actual)
    )

    wape = (
        np.sum(absolute_error)
        / actual_total
        * 100
        if actual_total > 0
        else np.nan
    )

    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(
            np.sqrt(np.mean(errors**2))
        ),
        "bias": float(np.mean(errors)),
        "wape_percent": float(wape),
    }


def backtest_models(
    values: np.ndarray,
) -> pd.DataFrame:
    """Evaluate candidate models on historical rolling windows."""

    selection_values = values[:-FINAL_HORIZON]

    required_length = (
        BACKTEST_WINDOWS * BACKTEST_HORIZON
    )

    first_validation_index = (
        len(selection_values) - required_length
    )

    if first_validation_index < 28:
        raise ValueError(
            "Insufficient history for rolling backtesting."
        )

    model_actuals: dict[str, list[np.ndarray]] = {
        model_name: []
        for model_name in MODEL_NAMES
    }

    model_predictions: dict[
        str,
        list[np.ndarray],
    ] = {
        model_name: []
        for model_name in MODEL_NAMES
    }

    for window_number in range(BACKTEST_WINDOWS):
        validation_start = (
            first_validation_index
            + window_number * BACKTEST_HORIZON
        )

        validation_end = (
            validation_start
            + BACKTEST_HORIZON
        )

        training = selection_values[
            :validation_start
        ]

        actual = selection_values[
            validation_start:validation_end
        ]

        candidate_forecasts = generate_forecasts(
            training,
            BACKTEST_HORIZON,
        )

        for model_name, predicted in (
            candidate_forecasts.items()
        ):
            model_actuals[model_name].append(actual)
            model_predictions[model_name].append(
                predicted
            )

    score_rows = []

    for model_name in MODEL_NAMES:
        combined_actual = np.concatenate(
            model_actuals[model_name]
        )

        combined_prediction = np.concatenate(
            model_predictions[model_name]
        )

        metrics = calculate_metrics(
            combined_actual,
            combined_prediction,
        )

        score_rows.append(
            {
                "candidate_model": model_name,
                "backtest_mae": metrics["mae"],
                "backtest_rmse": metrics["rmse"],
                "backtest_bias": metrics["bias"],
                "backtest_wape_percent": (
                    metrics["wape_percent"]
                ),
                "model_priority": (
                    MODEL_PRIORITY[model_name]
                ),
            }
        )

    scores = pd.DataFrame(score_rows)

    return scores.sort_values(
        [
            "backtest_mae",
            "backtest_rmse",
            "model_priority",
        ]
    ).reset_index(drop=True)


def evaluate_adaptive_model(
    sales: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Select and evaluate one model for every series."""

    selection_rows = []
    backtest_rows = []
    forecast_rows = []

    grouped = sales.groupby(
        ["store_id", "item_id"],
        sort=True,
    )

    for (
        store_id,
        item_id,
    ), series in grouped:
        series = (
            series.sort_values("date")
            .reset_index(drop=True)
        )

        values = series[
            "units_sold"
        ].to_numpy(dtype=float)

        backtest_scores = backtest_models(values)

        selected_model = backtest_scores.iloc[0][
            "candidate_model"
        ]

        backtest_scores.insert(
            0,
            "item_id",
            item_id,
        )

        backtest_scores.insert(
            0,
            "store_id",
            store_id,
        )

        backtest_rows.append(backtest_scores)

        final_training = values[
            :-FINAL_HORIZON
        ]

        final_actual = values[
            -FINAL_HORIZON:
        ]

        selected_prediction = generate_forecasts(
            final_training,
            FINAL_HORIZON,
        )[selected_model]

        final_metrics = calculate_metrics(
            final_actual,
            selected_prediction,
        )

        selection_rows.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "selected_model": selected_model,
                "backtest_mae": (
                    backtest_scores.iloc[0][
                        "backtest_mae"
                    ]
                ),
                "validation_mae": (
                    final_metrics["mae"]
                ),
                "validation_rmse": (
                    final_metrics["rmse"]
                ),
                "validation_bias": (
                    final_metrics["bias"]
                ),
                "validation_wape_percent": (
                    final_metrics[
                        "wape_percent"
                    ]
                ),
            }
        )

        validation_rows = series.iloc[
            -FINAL_HORIZON:
        ]

        forecast_rows.append(
            pd.DataFrame(
                {
                    "model": "adaptive_baseline",
                    "selected_model": selected_model,
                    "store_id": store_id,
                    "item_id": item_id,
                    "day_id": validation_rows[
                        "day_id"
                    ].to_numpy(),
                    "date": validation_rows[
                        "date"
                    ].to_numpy(),
                    "actual_units": final_actual,
                    "predicted_units": (
                        selected_prediction
                    ),
                }
            )
        )

    selections = pd.DataFrame(selection_rows)

    backtest_scores = pd.concat(
        backtest_rows,
        ignore_index=True,
    )

    forecasts = pd.concat(
        forecast_rows,
        ignore_index=True,
    )

    return selections, backtest_scores, forecasts


def calculate_group_metrics(
    forecasts: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Calculate metrics for requested forecast groups."""

    result_rows = []

    grouped = forecasts.groupby(
        group_columns,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        metrics = calculate_metrics(
            group["actual_units"].to_numpy(),
            group["predicted_units"].to_numpy(),
        )

        row = dict(
            zip(
                group_columns,
                group_key,
                strict=True,
            )
        )

        row.update(metrics)
        result_rows.append(row)

    result = pd.DataFrame(result_rows)

    metric_columns = [
        "mae",
        "rmse",
        "bias",
        "wape_percent",
    ]

    result[metric_columns] = result[
        metric_columns
    ].round(4)

    return result


def create_comparison_tables(
    adaptive_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine adaptive and existing baseline metrics."""

    adaptive_overall = calculate_group_metrics(
        adaptive_forecasts,
        ["model"],
    )

    adaptive_store = calculate_group_metrics(
        adaptive_forecasts,
        ["model", "store_id"],
    )

    baseline_overall = pd.read_csv(
        BASELINE_OVERALL_FILE
    )

    baseline_store = pd.read_csv(
        BASELINE_STORE_FILE
    )

    overall_comparison = pd.concat(
        [
            baseline_overall,
            adaptive_overall,
        ],
        ignore_index=True,
    ).sort_values("mae").reset_index(drop=True)

    store_comparison = pd.concat(
        [
            baseline_store,
            adaptive_store,
        ],
        ignore_index=True,
    ).sort_values(
        ["store_id", "mae"]
    ).reset_index(drop=True)

    return overall_comparison, store_comparison


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save one result table."""

    output_path = (
        REPORT_DIRECTORY / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] {filename:<42} "
        f"rows={len(dataframe):>5,}"
    )


def main() -> None:
    """Run adaptive baseline selection and evaluation."""

    print("=" * 78)
    print("ADAPTIVE BASELINE MODEL SELECTION")
    print("=" * 78)

    sales = load_sales()
    validate_sales(sales)

    (
        selections,
        backtest_scores,
        adaptive_forecasts,
    ) = evaluate_adaptive_model(sales)

    (
        overall_comparison,
        store_comparison,
    ) = create_comparison_tables(
        adaptive_forecasts
    )

    model_distribution = (
        selections.groupby(
            ["store_id", "selected_model"]
        )
        .size()
        .rename("number_of_series")
        .reset_index()
        .sort_values(
            ["store_id", "number_of_series"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    print("\nSelected models")
    print("-" * 78)
    print(
        model_distribution.to_string(
            index=False
        )
    )

    print("\nOverall comparison")
    print("-" * 78)
    print(
        overall_comparison.to_string(
            index=False
        )
    )

    print("\nComparison by store")
    print("-" * 78)
    print(
        store_comparison.to_string(
            index=False
        )
    )

    print("\nSaving results...")
    print("-" * 78)

    save_table(
        selections,
        "adaptive_model_selection_by_series.csv",
    )

    save_table(
        backtest_scores,
        "adaptive_backtest_scores.csv",
    )

    save_table(
        adaptive_forecasts,
        "adaptive_forecasts.csv",
    )

    save_table(
        model_distribution,
        "adaptive_model_distribution.csv",
    )

    save_table(
        overall_comparison,
        "adaptive_metrics_overall.csv",
    )

    save_table(
        store_comparison,
        "adaptive_metrics_by_store.csv",
    )

    print("\nAdaptive baseline evaluation completed.")


if __name__ == "__main__":
    main()