"""Analyse where the baseline forecasting models succeed and fail.

The analysis combines:
- training-period demand characteristics;
- series-level forecast metrics;
- descriptive demand-pattern groups;
- model wins by product-store series.

The demand-pattern labels used here are descriptive project categories,
not universal statistical classifications.
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

FORECAST_FILE = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "baseline_forecasts.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

EXPECTED_MODELS = {
    "last_value_naive",
    "weekly_seasonal_naive",
}


def load_forecasts() -> pd.DataFrame:
    """Load and validate baseline forecasts."""

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast file not found:\n{FORECAST_FILE}"
        )

    forecasts = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["date"],
    )

    required_columns = {
        "model",
        "store_id",
        "item_id",
        "day_id",
        "date",
        "actual_units",
        "predicted_units",
    }

    missing_columns = required_columns - set(
        forecasts.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing forecast columns: "
            + ", ".join(sorted(missing_columns))
        )

    models = set(forecasts["model"].unique())

    if models != EXPECTED_MODELS:
        raise ValueError(
            f"Expected models {EXPECTED_MODELS}, "
            f"but found {models}."
        )

    return forecasts


def load_training_sales(
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Load sales observations preceding the validation period."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DATABASE_FILE}"
        )

    query = """
        SELECT
            s.store_id,
            s.item_id,
            c.date,
            s.units_sold
        FROM fact_sales AS s
        INNER JOIN dim_calendar AS c
            ON s.day_id = c.day_id
        WHERE c.date < ?
        ORDER BY
            s.store_id,
            s.item_id,
            c.date;
    """

    validation_start_string = (
        validation_start.strftime("%Y-%m-%d")
    )

    with sqlite3.connect(DATABASE_FILE) as connection:
        training_sales = pd.read_sql_query(
            query,
            connection,
            params=[validation_start_string],
            parse_dates=["date"],
        )

    return training_sales


def classify_demand_pattern(
    zero_sales_percentage: float,
    mean_daily_demand: float,
) -> str:
    """Assign a descriptive demand-pattern category."""

    if mean_daily_demand == 0:
        return "no_training_demand"

    if zero_sales_percentage >= 80:
        return "highly_intermittent"

    if zero_sales_percentage >= 50:
        return "intermittent"

    if zero_sales_percentage >= 20:
        return "mixed"

    return "regular"


def calculate_training_characteristics(
    training_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate demand characteristics for each series."""

    grouped = training_sales.groupby(
        ["store_id", "item_id"],
        sort=True,
    )["units_sold"]

    characteristics = (
        grouped.agg(
            training_days="size",
            total_training_demand="sum",
            mean_daily_demand="mean",
            demand_standard_deviation="std",
            zero_sales_days=lambda values: (
                values.eq(0).sum()
            ),
        )
        .reset_index()
    )

    characteristics[
        "zero_sales_percentage"
    ] = (
        100
        * characteristics["zero_sales_days"]
        / characteristics["training_days"]
    )

    characteristics[
        "coefficient_of_variation"
    ] = (
        characteristics[
            "demand_standard_deviation"
        ]
        / characteristics["mean_daily_demand"]
    )

    characteristics[
        "coefficient_of_variation"
    ] = characteristics[
        "coefficient_of_variation"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    characteristics[
        "demand_pattern"
    ] = characteristics.apply(
        lambda row: classify_demand_pattern(
            row["zero_sales_percentage"],
            row["mean_daily_demand"],
        ),
        axis=1,
    )

    numeric_columns = [
        "mean_daily_demand",
        "demand_standard_deviation",
        "zero_sales_percentage",
        "coefficient_of_variation",
    ]

    characteristics[
        numeric_columns
    ] = characteristics[
        numeric_columns
    ].round(4)

    return characteristics


def calculate_metrics(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Calculate forecast metrics for one group."""

    actual = dataframe[
        "actual_units"
    ].to_numpy(dtype=float)

    prediction = dataframe[
        "predicted_units"
    ].to_numpy(dtype=float)

    error = prediction - actual

    actual_total = np.sum(
        np.abs(actual)
    )

    wape = (
        np.sum(np.abs(error))
        / actual_total
        * 100
        if actual_total > 0
        else np.nan
    )

    return pd.Series(
        {
            "validation_actual_units": actual.sum(),
            "validation_forecast_units": prediction.sum(),
            "mae": np.mean(np.abs(error)),
            "rmse": np.sqrt(np.mean(error**2)),
            "bias": np.mean(error),
            "wape_percent": wape,
        }
    )


def calculate_series_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics for every model and product-store series."""

    series_metrics = (
        forecasts.groupby(
            [
                "model",
                "store_id",
                "item_id",
            ],
            sort=True,
        )
        .apply(
            calculate_metrics,
            include_groups=False,
        )
        .reset_index()
    )

    numeric_columns = [
        "validation_actual_units",
        "validation_forecast_units",
        "mae",
        "rmse",
        "bias",
        "wape_percent",
    ]

    series_metrics[
        numeric_columns
    ] = series_metrics[
        numeric_columns
    ].round(4)

    return series_metrics


def calculate_pattern_metrics(
    forecasts: pd.DataFrame,
    characteristics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate model performance by demand-pattern group."""

    labelled_forecasts = forecasts.merge(
        characteristics[
            [
                "store_id",
                "item_id",
                "demand_pattern",
            ]
        ],
        on=["store_id", "item_id"],
        how="left",
        validate="many_to_one",
    )

    if labelled_forecasts[
        "demand_pattern"
    ].isna().any():
        raise ValueError(
            "Some forecasts could not be matched "
            "to demand characteristics."
        )

    pattern_metrics = (
        labelled_forecasts.groupby(
            ["model", "demand_pattern"],
            sort=True,
        )
        .apply(
            calculate_metrics,
            include_groups=False,
        )
        .reset_index()
    )

    series_counts = (
        characteristics.groupby(
            "demand_pattern"
        )
        .size()
        .rename("number_of_series")
        .reset_index()
    )

    pattern_metrics = pattern_metrics.merge(
        series_counts,
        on="demand_pattern",
        how="left",
        validate="many_to_one",
    )

    numeric_columns = [
        "validation_actual_units",
        "validation_forecast_units",
        "mae",
        "rmse",
        "bias",
        "wape_percent",
    ]

    pattern_metrics[
        numeric_columns
    ] = pattern_metrics[
        numeric_columns
    ].round(4)

    return pattern_metrics


def determine_model_winners(
    series_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Determine the lowest-MAE model for each series."""

    mae_table = series_metrics.pivot(
        index=["store_id", "item_id"],
        columns="model",
        values="mae",
    )

    missing_models = EXPECTED_MODELS - set(
        mae_table.columns
    )

    if missing_models:
        raise ValueError(
            "Series metrics are missing models: "
            + ", ".join(sorted(missing_models))
        )

    winners = (
        mae_table.idxmin(axis=1)
        .rename("best_model")
        .reset_index()
    )

    winner_summary = (
        winners.groupby(
            ["store_id", "best_model"]
        )
        .size()
        .rename("series_won")
        .reset_index()
        .sort_values(
            ["store_id", "series_won"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return winner_summary


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save one analytical result table."""

    output_path = OUTPUT_DIRECTORY / filename

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] {filename:<42} "
        f"rows={len(dataframe):>4}"
    )


def main() -> None:
    """Run the complete baseline error analysis."""

    print("=" * 78)
    print("BASELINE FORECAST ERROR ANALYSIS")
    print("=" * 78)

    forecasts = load_forecasts()

    validation_start = forecasts["date"].min()

    training_sales = load_training_sales(
        validation_start
    )

    characteristics = (
        calculate_training_characteristics(
            training_sales
        )
    )

    series_metrics = calculate_series_metrics(
        forecasts
    )

    combined_series_metrics = (
        series_metrics.merge(
            characteristics,
            on=["store_id", "item_id"],
            how="left",
            validate="many_to_one",
        )
    )

    pattern_metrics = calculate_pattern_metrics(
        forecasts,
        characteristics,
    )

    model_winners = determine_model_winners(
        series_metrics
    )

    weekly_series = (
        combined_series_metrics.loc[
            combined_series_metrics["model"].eq(
                "weekly_seasonal_naive"
            )
        ]
        .sort_values(
            "mae",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    worst_weekly_series = weekly_series.head(20)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Validation start:       "
        f"{validation_start.date()}"
    )
    print(
        f"Training observations:  "
        f"{len(training_sales):,}"
    )
    print(
        f"Series analysed:        "
        f"{len(characteristics):,}"
    )

    print("\nDemand-pattern distribution")
    print("-" * 78)
    print(
        characteristics[
            "demand_pattern"
        ]
        .value_counts()
        .rename_axis("demand_pattern")
        .reset_index(name="number_of_series")
        .to_string(index=False)
    )

    print("\nModel wins by store")
    print("-" * 78)
    print(
        model_winners.to_string(
            index=False
        )
    )

    print("\nTen most difficult series for the weekly baseline")
    print("-" * 78)
    print(
        worst_weekly_series[
            [
                "store_id",
                "item_id",
                "mae",
                "rmse",
                "bias",
                "wape_percent",
                "mean_daily_demand",
                "zero_sales_percentage",
                "demand_pattern",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaving results...")
    print("-" * 78)

    save_table(
        characteristics,
        "demand_characteristics_by_series.csv",
    )

    save_table(
        combined_series_metrics,
        "baseline_metrics_by_series.csv",
    )

    save_table(
        pattern_metrics,
        "baseline_metrics_by_demand_pattern.csv",
    )

    save_table(
        model_winners,
        "baseline_model_wins_by_store.csv",
    )

    save_table(
        worst_weekly_series,
        "baseline_worst_weekly_series.csv",
    )

    print("\nBaseline error analysis completed successfully.")


if __name__ == "__main__":
    main()