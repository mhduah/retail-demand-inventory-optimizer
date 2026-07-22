"""Evaluate forecast-driven inventory replenishment policies.

This script compares adaptive-baseline and gradient-boosting forecasts
under two unconstrained 28-day replenishment policies:

1. Point-forecast ordering
2. Newsvendor-adjusted ordering with safety stock

The cost assumptions are expressed in relative cost units rather than
currency. Budget and storage constraints will be introduced later.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm


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

ADAPTIVE_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "adaptive_forecasts.csv"
)

ML_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "ml_forecasts.csv"
)

FORECAST_HORIZON = 28
EXPECTED_SERIES = 150
EXPECTED_FORECAST_ROWS = (
    EXPECTED_SERIES
    * FORECAST_HORIZON
)

UNCERTAINTY_LOOKBACK_DAYS = 365

HOLDING_COST_PER_UNIT = 1.0
SHORTAGE_COST_PER_UNIT = 5.0

CRITICAL_RATIO = (
    SHORTAGE_COST_PER_UNIT
    / (
        SHORTAGE_COST_PER_UNIT
        + HOLDING_COST_PER_UNIT
    )
)

SERVICE_FACTOR = float(
    norm.ppf(CRITICAL_RATIO)
)


def load_forecast_file(
    path: Path,
    expected_model: str,
) -> pd.DataFrame:
    """Load and validate one forecast result file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found:\n{path}"
        )

    forecasts = pd.read_csv(
        path,
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

    missing_columns = (
        required_columns
        - set(forecasts.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if len(forecasts) != EXPECTED_FORECAST_ROWS:
        raise ValueError(
            f"{path.name} contains "
            f"{len(forecasts):,} rows; "
            f"expected {EXPECTED_FORECAST_ROWS:,}."
        )

    models = set(
        forecasts["model"].unique()
    )

    if models != {expected_model}:
        raise ValueError(
            f"Expected model {expected_model}, "
            f"but found {models}."
        )

    if forecasts[
        "predicted_units"
    ].isna().any():
        raise ValueError(
            f"Missing predictions found in {path.name}."
        )

    if (
        forecasts["predicted_units"] < 0
    ).any():
        raise ValueError(
            f"Negative predictions found in {path.name}."
        )

    return forecasts


def aggregate_forecasts(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily forecasts to 28-day series totals."""

    aggregated = (
        forecasts.groupby(
            [
                "model",
                "store_id",
                "item_id",
            ],
            sort=True,
        )
        .agg(
            forecast_total_units=(
                "predicted_units",
                "sum",
            ),
            actual_total_units=(
                "actual_units",
                "sum",
            ),
            forecast_start_date=(
                "date",
                "min",
            ),
            forecast_end_date=(
                "date",
                "max",
            ),
            forecast_days=(
                "day_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    if len(aggregated) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} "
            f"aggregated series, "
            f"but found {len(aggregated)}."
        )

    if not aggregated[
        "forecast_days"
    ].eq(FORECAST_HORIZON).all():
        raise ValueError(
            "Some series do not contain "
            f"{FORECAST_HORIZON} forecast days."
        )

    return aggregated


def validate_matching_actuals(
    adaptive: pd.DataFrame,
    ml: pd.DataFrame,
) -> None:
    """Ensure both model files use the same holdout observations."""

    comparison = adaptive.merge(
        ml,
        on=[
            "store_id",
            "item_id",
        ],
        suffixes=(
            "_adaptive",
            "_ml",
        ),
        validate="one_to_one",
    )

    matching_actuals = np.allclose(
        comparison[
            "actual_total_units_adaptive"
        ],
        comparison[
            "actual_total_units_ml"
        ],
    )

    if not matching_actuals:
        raise ValueError(
            "Adaptive and ML files contain "
            "different actual demand totals."
        )

    matching_start_dates = (
        comparison[
            "forecast_start_date_adaptive"
        ]
        .eq(
            comparison[
                "forecast_start_date_ml"
            ]
        )
        .all()
    )

    matching_end_dates = (
        comparison[
            "forecast_end_date_adaptive"
        ]
        .eq(
            comparison[
                "forecast_end_date_ml"
            ]
        )
        .all()
    )

    if not (
        matching_start_dates
        and matching_end_dates
    ):
        raise ValueError(
            "Forecast periods do not match."
        )


def load_training_sales(
    holdout_start: pd.Timestamp,
) -> pd.DataFrame:
    """Load sales preceding the final evaluation period."""

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

    with sqlite3.connect(
        DATABASE_FILE
    ) as connection:
        training_sales = pd.read_sql_query(
            query,
            connection,
            params=[
                holdout_start.strftime(
                    "%Y-%m-%d"
                )
            ],
            parse_dates=["date"],
        )

    return training_sales


def calculate_demand_uncertainty(
    training_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate variability of 28-day demand totals by series."""

    uncertainty_rows = []

    grouped = training_sales.groupby(
        ["store_id", "item_id"],
        sort=True,
    )

    for (
        store_id,
        item_id,
    ), series in grouped:
        values = (
            series.sort_values("date")[
                "units_sold"
            ]
            .tail(
                UNCERTAINTY_LOOKBACK_DAYS
            )
            .astype(float)
            .reset_index(drop=True)
        )

        if len(values) < FORECAST_HORIZON:
            raise ValueError(
                f"Insufficient history for "
                f"{store_id}-{item_id}."
            )

        rolling_totals = (
            values.rolling(
                window=FORECAST_HORIZON,
                min_periods=FORECAST_HORIZON,
            )
            .sum()
            .dropna()
        )

        uncertainty_std = float(
            rolling_totals.std(
                ddof=1
            )
        )

        if np.isnan(
            uncertainty_std
        ):
            uncertainty_std = 0.0

        uncertainty_rows.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "uncertainty_std_28": (
                    uncertainty_std
                ),
                "uncertainty_windows": (
                    len(rolling_totals)
                ),
            }
        )

    uncertainty = pd.DataFrame(
        uncertainty_rows
    )

    if len(uncertainty) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected uncertainty estimates for "
            f"{EXPECTED_SERIES} series, "
            f"but found {len(uncertainty)}."
        )

    return uncertainty


def create_inventory_results(
    forecasts: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    """Create point-forecast and newsvendor order policies."""

    enriched = forecasts.merge(
        uncertainty,
        on=[
            "store_id",
            "item_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if enriched[
        "uncertainty_std_28"
    ].isna().any():
        raise ValueError(
            "Missing demand uncertainty estimates."
        )

    result_rows = []

    for row in enriched.itertuples(
        index=False
    ):
        point_order = int(
            np.ceil(
                max(
                    0.0,
                    row.forecast_total_units,
                )
            )
        )

        newsvendor_target = (
            row.forecast_total_units
            + SERVICE_FACTOR
            * row.uncertainty_std_28
        )

        newsvendor_order = int(
            np.ceil(
                max(
                    0.0,
                    newsvendor_target,
                )
            )
        )

        policy_orders = {
            "point_forecast_order": (
                point_order
            ),
            "newsvendor_adjusted_order": (
                newsvendor_order
            ),
        }

        for (
            inventory_policy,
            order_quantity,
        ) in policy_orders.items():
            actual_demand = float(
                row.actual_total_units
            )

            served_units = min(
                float(order_quantity),
                actual_demand,
            )

            shortage_units = max(
                actual_demand
                - order_quantity,
                0.0,
            )

            leftover_units = max(
                order_quantity
                - actual_demand,
                0.0,
            )

            holding_cost = (
                leftover_units
                * HOLDING_COST_PER_UNIT
            )

            shortage_cost = (
                shortage_units
                * SHORTAGE_COST_PER_UNIT
            )

            total_cost = (
                holding_cost
                + shortage_cost
            )

            fill_rate = (
                100.0
                if actual_demand == 0
                else (
                    served_units
                    / actual_demand
                    * 100
                )
            )

            safety_stock_units = max(
                order_quantity
                - point_order,
                0,
            )

            result_rows.append(
                {
                    "forecast_model": row.model,
                    "inventory_policy": (
                        inventory_policy
                    ),
                    "store_id": row.store_id,
                    "item_id": row.item_id,
                    "forecast_total_units": (
                        row.forecast_total_units
                    ),
                    "actual_demand_units": (
                        actual_demand
                    ),
                    "uncertainty_std_28": (
                        row.uncertainty_std_28
                    ),
                    "service_factor": (
                        SERVICE_FACTOR
                    ),
                    "safety_stock_units": (
                        safety_stock_units
                    ),
                    "order_quantity": (
                        order_quantity
                    ),
                    "served_units": (
                        served_units
                    ),
                    "shortage_units": (
                        shortage_units
                    ),
                    "leftover_units": (
                        leftover_units
                    ),
                    "fill_rate_percent": (
                        fill_rate
                    ),
                    "stockout_occurred": int(
                        shortage_units > 0
                    ),
                    "holding_cost_units": (
                        holding_cost
                    ),
                    "shortage_cost_units": (
                        shortage_cost
                    ),
                    "total_cost_units": (
                        total_cost
                    ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )

    numeric_columns = [
        "forecast_total_units",
        "actual_demand_units",
        "uncertainty_std_28",
        "service_factor",
        "served_units",
        "shortage_units",
        "leftover_units",
        "fill_rate_percent",
        "holding_cost_units",
        "shortage_cost_units",
        "total_cost_units",
    ]

    results[
        numeric_columns
    ] = results[
        numeric_columns
    ].round(4)

    return results


def summarize_inventory_results(
    results: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    """Summarize inventory performance by model and policy."""

    summary = (
        results.groupby(
            grouping_columns,
            sort=True,
        )
        .agg(
            number_of_series=(
                "item_id",
                "size",
            ),
            forecast_units=(
                "forecast_total_units",
                "sum",
            ),
            actual_demand_units=(
                "actual_demand_units",
                "sum",
            ),
            ordered_units=(
                "order_quantity",
                "sum",
            ),
            served_units=(
                "served_units",
                "sum",
            ),
            shortage_units=(
                "shortage_units",
                "sum",
            ),
            leftover_units=(
                "leftover_units",
                "sum",
            ),
            stockout_series=(
                "stockout_occurred",
                "sum",
            ),
            holding_cost_units=(
                "holding_cost_units",
                "sum",
            ),
            shortage_cost_units=(
                "shortage_cost_units",
                "sum",
            ),
            total_cost_units=(
                "total_cost_units",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "fill_rate_percent"
    ] = np.where(
        summary[
            "actual_demand_units"
        ].eq(0),
        100.0,
        (
            summary["served_units"]
            / summary[
                "actual_demand_units"
            ]
            * 100
        ),
    )

    summary[
        "stockout_series_percent"
    ] = (
        summary["stockout_series"]
        / summary["number_of_series"]
        * 100
    )

    summary[
        "average_cost_per_series"
    ] = (
        summary["total_cost_units"]
        / summary["number_of_series"]
    )

    numeric_columns = [
        "forecast_units",
        "actual_demand_units",
        "ordered_units",
        "served_units",
        "shortage_units",
        "leftover_units",
        "holding_cost_units",
        "shortage_cost_units",
        "total_cost_units",
        "fill_rate_percent",
        "stockout_series_percent",
        "average_cost_per_series",
    ]

    summary[
        numeric_columns
    ] = summary[
        numeric_columns
    ].round(2)

    return summary


def create_assumptions_table() -> pd.DataFrame:
    """Create a tracked description of policy assumptions."""

    return pd.DataFrame(
        {
            "assumption": [
                "forecast_horizon_days",
                "uncertainty_lookback_days",
                "holding_cost_per_leftover_unit",
                "shortage_cost_per_unserved_unit",
                "critical_ratio",
                "normal_service_factor",
                "cost_unit_interpretation",
            ],
            "value": [
                FORECAST_HORIZON,
                UNCERTAINTY_LOOKBACK_DAYS,
                HOLDING_COST_PER_UNIT,
                SHORTAGE_COST_PER_UNIT,
                round(
                    CRITICAL_RATIO,
                    4,
                ),
                round(
                    SERVICE_FACTOR,
                    4,
                ),
                (
                    "Relative cost units, "
                    "not currency"
                ),
            ],
        }
    )


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save one inventory-analysis result."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIRECTORY
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] {filename:<38} "
        f"rows={len(dataframe):>4}"
    )


def main() -> None:
    """Evaluate forecast-driven inventory policies."""

    print("=" * 80)
    print("FORECAST-DRIVEN INVENTORY POLICY EVALUATION")
    print("=" * 80)

    adaptive_daily = load_forecast_file(
        ADAPTIVE_FORECAST_FILE,
        "adaptive_baseline",
    )

    ml_daily = load_forecast_file(
        ML_FORECAST_FILE,
        "hist_gradient_boosting",
    )

    adaptive_totals = aggregate_forecasts(
        adaptive_daily
    )

    ml_totals = aggregate_forecasts(
        ml_daily
    )

    validate_matching_actuals(
        adaptive_totals,
        ml_totals,
    )

    holdout_start = adaptive_daily[
        "date"
    ].min()

    holdout_end = adaptive_daily[
        "date"
    ].max()

    training_sales = load_training_sales(
        holdout_start
    )

    uncertainty = calculate_demand_uncertainty(
        training_sales
    )

    combined_forecasts = pd.concat(
        [
            adaptive_totals,
            ml_totals,
        ],
        ignore_index=True,
    )

    inventory_results = (
        create_inventory_results(
            combined_forecasts,
            uncertainty,
        )
    )

    overall_summary = (
        summarize_inventory_results(
            inventory_results,
            [
                "forecast_model",
                "inventory_policy",
            ],
        )
    )

    store_summary = (
        summarize_inventory_results(
            inventory_results,
            [
                "store_id",
                "forecast_model",
                "inventory_policy",
            ],
        )
    )

    assumptions = (
        create_assumptions_table()
    )

    print(
        f"Evaluation period: "
        f"{holdout_start.date()} "
        f"to {holdout_end.date()}"
    )

    print(
        f"Critical ratio:   "
        f"{CRITICAL_RATIO:.4f}"
    )

    print(
        f"Service factor:   "
        f"{SERVICE_FACTOR:.4f}"
    )

    print("\nOverall inventory-policy comparison")
    print("-" * 80)

    display_columns = [
        "forecast_model",
        "inventory_policy",
        "ordered_units",
        "shortage_units",
        "leftover_units",
        "fill_rate_percent",
        "stockout_series_percent",
        "total_cost_units",
    ]

    print(
        overall_summary[
            display_columns
        ]
        .sort_values(
            "total_cost_units"
        )
        .to_string(
            index=False
        )
    )

    print("\nSaving results...")
    print("-" * 80)

    save_table(
        inventory_results,
        "inventory_policy_by_series.csv",
    )

    save_table(
        overall_summary,
        "inventory_policy_summary.csv",
    )

    save_table(
        store_summary,
        "inventory_policy_by_store.csv",
    )

    save_table(
        assumptions,
        "inventory_policy_assumptions.csv",
    )

    print("\nInventory-policy evaluation completed.")


if __name__ == "__main__":
    main()