"""Create and evaluate baseline 28-day demand forecasts.

Models
------
1. Last-value naive:
   Repeats the final observed training value for all 28 forecast days.

2. Weekly seasonal naive:
   Repeats the final seven-day training pattern over the 28-day horizon.

The final 28 historical days are held out for time-based validation.
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

FORECAST_HORIZON = 28
SEASON_LENGTH = 7

EXPECTED_ITEMS = 50
EXPECTED_STORES = 3
EXPECTED_DAYS = 1_941
EXPECTED_SERIES = EXPECTED_ITEMS * EXPECTED_STORES


def load_sales_data() -> pd.DataFrame:
    """Load chronologically ordered sales from SQLite."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DATABASE_FILE}"
        )

    query = """
        SELECT
            s.item_id,
            s.store_id,
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


def validate_sales_data(
    sales: pd.DataFrame,
) -> None:
    """Validate the time-series panel before forecasting."""

    required_columns = {
        "item_id",
        "store_id",
        "day_id",
        "date",
        "units_sold",
    }

    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sales.empty:
        raise ValueError("The sales dataset is empty.")

    if sales["units_sold"].isna().any():
        raise ValueError(
            "Missing sales observations were found."
        )

    if (sales["units_sold"] < 0).any():
        raise ValueError(
            "Negative sales observations were found."
        )

    number_of_series = (
        sales[["item_id", "store_id"]]
        .drop_duplicates()
        .shape[0]
    )

    if number_of_series != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} series, "
            f"but found {number_of_series}."
        )

    observations_per_series = (
        sales.groupby(
            ["item_id", "store_id"]
        )
        .size()
    )

    invalid_series = observations_per_series[
        observations_per_series.ne(EXPECTED_DAYS)
    ]

    if not invalid_series.empty:
        raise ValueError(
            "Some product-store series do not contain "
            f"exactly {EXPECTED_DAYS} observations."
        )

    unique_dates = sales["date"].nunique()

    if unique_dates != EXPECTED_DAYS:
        raise ValueError(
            f"Expected {EXPECTED_DAYS} dates, "
            f"but found {unique_dates}."
        )

    if FORECAST_HORIZON % SEASON_LENGTH != 0:
        raise ValueError(
            "Forecast horizon must be divisible by "
            "the seasonal period."
        )


def create_forecasts(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Create last-value and weekly seasonal-naive forecasts."""

    forecast_frames: list[pd.DataFrame] = []

    grouped_sales = sales.groupby(
        ["store_id", "item_id"],
        sort=True,
    )

    for (
        store_id,
        item_id,
    ), series in grouped_sales:
        series = (
            series.sort_values("date")
            .reset_index(drop=True)
        )

        training = series.iloc[
            :-FORECAST_HORIZON
        ]

        validation = series.iloc[
            -FORECAST_HORIZON:
        ].copy()

        if len(training) != (
            EXPECTED_DAYS - FORECAST_HORIZON
        ):
            raise ValueError(
                f"Invalid training length for "
                f"{store_id}-{item_id}."
            )

        if len(validation) != FORECAST_HORIZON:
            raise ValueError(
                f"Invalid validation length for "
                f"{store_id}-{item_id}."
            )

        last_value_prediction = np.repeat(
            training["units_sold"].iloc[-1],
            FORECAST_HORIZON,
        )

        weekly_pattern = (
            training["units_sold"]
            .tail(SEASON_LENGTH)
            .to_numpy()
        )

        seasonal_prediction = np.resize(
            weekly_pattern,
            FORECAST_HORIZON,
        )

        model_predictions = {
            "last_value_naive": last_value_prediction,
            "weekly_seasonal_naive": seasonal_prediction,
        }

        for model_name, prediction in (
            model_predictions.items()
        ):
            forecast = pd.DataFrame(
                {
                    "model": model_name,
                    "store_id": store_id,
                    "item_id": item_id,
                    "day_id": validation[
                        "day_id"
                    ].to_numpy(),
                    "date": validation[
                        "date"
                    ].to_numpy(),
                    "actual_units": validation[
                        "units_sold"
                    ].to_numpy(),
                    "predicted_units": prediction,
                }
            )

            forecast_frames.append(forecast)

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    forecasts["prediction_error"] = (
        forecasts["predicted_units"]
        - forecasts["actual_units"]
    )

    forecasts["absolute_error"] = (
        forecasts["prediction_error"].abs()
    )

    forecasts["squared_error"] = (
        forecasts["prediction_error"] ** 2
    )

    if forecasts["predicted_units"].isna().any():
        raise ValueError(
            "Missing forecast values were generated."
        )

    if (forecasts["predicted_units"] < 0).any():
        raise ValueError(
            "Negative forecast values were generated."
        )

    expected_rows = (
        EXPECTED_SERIES
        * FORECAST_HORIZON
        * len(forecasts["model"].unique())
    )

    if len(forecasts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} forecast rows, "
            f"but created {len(forecasts):,}."
        )

    return forecasts


def calculate_metrics(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Calculate forecasting accuracy metrics."""

    actual = dataframe[
        "actual_units"
    ].to_numpy(dtype=float)

    prediction = dataframe[
        "predicted_units"
    ].to_numpy(dtype=float)

    error = prediction - actual

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error**2)
    )

    bias = np.mean(error)

    actual_total = np.sum(
        np.abs(actual)
    )

    if actual_total == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(np.abs(error))
            / actual_total
            * 100
        )

    return pd.Series(
        {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "wape_percent": wape,
        }
    )


def create_metric_tables(
    forecasts: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create overall and store-level metric tables."""

    overall_metrics = (
        forecasts.groupby(
            "model",
            sort=False,
        )
        .apply(
            calculate_metrics,
            include_groups=False,
        )
        .reset_index()
        .sort_values("mae")
        .reset_index(drop=True)
    )

    store_metrics = (
        forecasts.groupby(
            ["model", "store_id"],
            sort=False,
        )
        .apply(
            calculate_metrics,
            include_groups=False,
        )
        .reset_index()
        .sort_values(
            ["model", "mae"]
        )
        .reset_index(drop=True)
    )

    numeric_columns = [
        "mae",
        "rmse",
        "bias",
        "wape_percent",
    ]

    overall_metrics[
        numeric_columns
    ] = overall_metrics[
        numeric_columns
    ].round(4)

    store_metrics[
        numeric_columns
    ] = store_metrics[
        numeric_columns
    ].round(4)

    return overall_metrics, store_metrics


def save_results(
    forecasts: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    store_metrics: pd.DataFrame,
) -> None:
    """Save predictions and evaluation metrics."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "baseline_forecasts.csv": forecasts,
        "baseline_metrics_overall.csv": (
            overall_metrics
        ),
        "baseline_metrics_by_store.csv": (
            store_metrics
        ),
    }

    for filename, dataframe in (
        output_files.items()
    ):
        output_path = (
            OUTPUT_DIRECTORY
            / filename
        )

        dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"[SAVED] {filename:<32} "
            f"rows={len(dataframe):>6,}"
        )


def main() -> None:
    """Run baseline forecasting and evaluation."""

    print("=" * 76)
    print("28-DAY DEMAND FORECASTING BASELINES")
    print("=" * 76)

    sales = load_sales_data()
    validate_sales_data(sales)

    validation_dates = (
        sales["date"]
        .drop_duplicates()
        .sort_values()
        .tail(FORECAST_HORIZON)
    )

    print(
        f"Series:              {EXPECTED_SERIES:,}"
    )
    print(
        f"Historical days:     {EXPECTED_DAYS:,}"
    )
    print(
        f"Training days:       "
        f"{EXPECTED_DAYS - FORECAST_HORIZON:,}"
    )
    print(
        f"Validation days:     {FORECAST_HORIZON}"
    )
    print(
        f"Validation period:   "
        f"{validation_dates.iloc[0].date()} "
        f"to {validation_dates.iloc[-1].date()}"
    )

    forecasts = create_forecasts(
        sales
    )

    (
        overall_metrics,
        store_metrics,
    ) = create_metric_tables(
        forecasts
    )

    print("\nOverall forecasting metrics")
    print("-" * 76)
    print(
        overall_metrics.to_string(
            index=False
        )
    )

    print("\nMetrics by store")
    print("-" * 76)
    print(
        store_metrics.to_string(
            index=False
        )
    )

    print("\nSaving results...")
    print("-" * 76)

    save_results(
        forecasts,
        overall_metrics,
        store_metrics,
    )

    best_model = overall_metrics.iloc[0]

    print("\n" + "-" * 76)
    print(
        f"Best baseline by MAE: "
        f"{best_model['model']}"
    )
    print(
        f"MAE:  {best_model['mae']:.4f}"
    )
    print(
        f"RMSE: {best_model['rmse']:.4f}"
    )
    print(
        f"Bias: {best_model['bias']:.4f}"
    )


if __name__ == "__main__":
    main()