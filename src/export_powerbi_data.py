"""Export dashboard-ready tables for Power BI.

The export follows a star-schema structure with shared dimensions for
stores, products, dates, forecast models, policies, constraint
scenarios, and allocation methods.

Large technical intermediate files remain outside the Power BI model.
Only curated analytical tables are exported.
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

POWERBI_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "powerbi"
)

BASELINE_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "baseline_forecasts.csv"
)

ADAPTIVE_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "adaptive_forecasts.csv"
)

ML_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "ml_forecasts.csv"
)

INVENTORY_POLICY_FILE = (
    REPORT_DIRECTORY
    / "inventory_policy_by_series.csv"
)

CONSTRAINED_ALLOCATION_FILE = (
    REPORT_DIRECTORY
    / "stochastic_allocation_by_series.csv"
)

EXPECTED_STORES = 3
EXPECTED_ITEMS = 50
EXPECTED_DATES = 1_941
EXPECTED_SERIES = 150
EXPECTED_HOLDOUT_DAYS = 28
EXPECTED_ACTUAL_DAILY_ROWS = (
    EXPECTED_SERIES
    * EXPECTED_HOLDOUT_DAYS
)


MODEL_METADATA = {
    "last_value_naive": {
        "model_name": "Last-value naïve",
        "model_family": "Statistical baseline",
        "display_order": 1,
    },
    "weekly_seasonal_naive": {
        "model_name": "Weekly seasonal naïve",
        "model_family": "Statistical baseline",
        "display_order": 2,
    },
    "adaptive_baseline": {
        "model_name": "Adaptive baseline",
        "model_family": "Backtest-selected baseline",
        "display_order": 3,
    },
    "hist_gradient_boosting": {
        "model_name": "Histogram gradient boosting",
        "model_family": "Machine learning",
        "display_order": 4,
    },
}


POLICY_METADATA = {
    "point_forecast_order": {
        "policy_name": "Point-forecast order",
        "uses_safety_stock": 0,
        "display_order": 1,
    },
    "newsvendor_adjusted_order": {
        "policy_name": "Newsvendor-adjusted order",
        "uses_safety_stock": 1,
        "display_order": 2,
    },
}


SCENARIO_METADATA = {
    "tight": {
        "scenario_name": "Tight",
        "budget_ratio": 0.65,
        "capacity_ratio": 0.70,
        "display_order": 1,
    },
    "balanced": {
        "scenario_name": "Balanced",
        "budget_ratio": 0.80,
        "capacity_ratio": 0.85,
        "display_order": 2,
    },
    "near_full": {
        "scenario_name": "Near full",
        "budget_ratio": 0.95,
        "capacity_ratio": 0.95,
        "display_order": 3,
    },
}


ALLOCATION_METHOD_METADATA = {
    "proportional_heuristic": {
        "allocation_method_name": (
            "Proportional heuristic"
        ),
        "method_family": "Heuristic",
        "display_order": 1,
    },
    "milp_optimised": {
        "allocation_method_name": (
            "Target-shortfall MILP"
        ),
        "method_family": (
            "Deterministic optimisation"
        ),
        "display_order": 2,
    },
    "stochastic_expected_cost_milp": {
        "allocation_method_name": (
            "Stochastic expected-cost MILP"
        ),
        "method_family": (
            "Stochastic optimisation"
        ),
        "display_order": 3,
    },
}


def require_file(path: Path) -> None:
    """Raise an informative error when an input file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def read_database_table(
    table_name: str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Read one SQLite table."""

    require_file(DATABASE_FILE)

    query = f"SELECT * FROM {table_name};"

    with sqlite3.connect(DATABASE_FILE) as connection:
        dataframe = pd.read_sql_query(
            query,
            connection,
            parse_dates=parse_dates,
        )

    return dataframe


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Validate required columns."""

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    table_name: str,
) -> None:
    """Validate one table's intended primary key."""

    duplicated = dataframe.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = dataframe.loc[
            duplicated,
            key_columns,
        ].head()

        raise ValueError(
            f"{table_name} has duplicate keys for "
            f"{key_columns}:\n"
            f"{duplicate_rows}"
        )


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
    grain: str,
    manifest_rows: list[dict[str, object]],
) -> None:
    """Save one Power BI table and record its metadata."""

    POWERBI_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        POWERBI_DIRECTORY
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )

    manifest_rows.append(
        {
            "file_name": filename,
            "table_name": output_path.stem,
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "grain": grain,
        }
    )

    print(
        f"[SAVED] {filename:<42} "
        f"rows={len(dataframe):>7,} "
        f"columns={len(dataframe.columns):>2}"
    )


def create_store_dimension() -> pd.DataFrame:
    """Create the store dimension."""

    stores = read_database_table(
        "dim_stores"
    )

    require_columns(
        stores,
        {
            "store_id",
            "state_id",
        },
        "dim_stores",
    )

    validate_unique_key(
        stores,
        ["store_id"],
        "dim_store",
    )

    if len(stores) != EXPECTED_STORES:
        raise ValueError(
            f"Expected {EXPECTED_STORES} stores, "
            f"but found {len(stores)}."
        )

    stores = stores.copy()

    stores["store_name"] = (
        stores["store_id"]
        .str.replace("_", " ", regex=False)
    )

    preferred_columns = [
        "store_id",
        "store_name",
        "state_id",
    ]

    remaining_columns = [
        column
        for column in stores.columns
        if column not in preferred_columns
    ]

    return stores[
        preferred_columns
        + remaining_columns
    ]


def create_item_dimension() -> pd.DataFrame:
    """Create the product dimension."""

    items = read_database_table(
        "dim_items"
    )

    require_columns(
        items,
        {"item_id"},
        "dim_items",
    )

    validate_unique_key(
        items,
        ["item_id"],
        "dim_item",
    )

    if len(items) != EXPECTED_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_ITEMS} items, "
            f"but found {len(items)}."
        )

    items = items.copy()

    items["item_name"] = (
        items["item_id"]
        .str.replace("_", " ", regex=False)
    )

    preferred_columns = [
        "item_id",
        "item_name",
    ]

    remaining_columns = [
        column
        for column in items.columns
        if column not in preferred_columns
    ]

    return items[
        preferred_columns
        + remaining_columns
    ]


def create_date_dimension() -> pd.DataFrame:
    """Create the date dimension."""

    dates = read_database_table(
        "dim_calendar",
        parse_dates=["date"],
    )

    require_columns(
        dates,
        {
            "day_id",
            "date",
            "wday",
            "month",
            "year",
        },
        "dim_calendar",
    )

    validate_unique_key(
        dates,
        ["day_id"],
        "dim_date",
    )

    if len(dates) != EXPECTED_DATES:
        raise ValueError(
            f"Expected {EXPECTED_DATES} dates, "
            f"but found {len(dates)}."
        )

    dates = dates.sort_values(
        "date"
    ).reset_index(drop=True)

    dates["date_key"] = (
        dates["date"]
        .dt.strftime("%Y%m%d")
        .astype("int32")
    )

    dates["day_of_month"] = (
        dates["date"].dt.day
    )

    dates["quarter"] = (
        dates["date"].dt.quarter
    )

    dates["month_name"] = (
        dates["date"].dt.month_name()
    )

    dates["weekday_name"] = (
        dates["date"].dt.day_name()
    )

    dates["year_month"] = (
        dates["date"]
        .dt.strftime("%Y-%m")
    )

    dates["is_weekend"] = (
        dates["date"]
        .dt.dayofweek
        .ge(5)
        .astype("int8")
    )

    preferred_columns = [
        "date_key",
        "day_id",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "year_month",
        "day_of_month",
        "wday",
        "weekday_name",
        "is_weekend",
    ]

    remaining_columns = [
        column
        for column in dates.columns
        if column not in preferred_columns
    ]

    return dates[
        preferred_columns
        + remaining_columns
    ]


def metadata_dimension(
    metadata: dict[str, dict[str, object]],
    key_name: str,
) -> pd.DataFrame:
    """Create a dimension from a metadata dictionary."""

    rows = []

    for key_value, attributes in metadata.items():
        rows.append(
            {
                key_name: key_value,
                **attributes,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "display_order"
    ).reset_index(drop=True)


def load_forecast_file(
    path: Path,
) -> pd.DataFrame:
    """Load and standardise one forecast file."""

    require_file(path)

    forecasts = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    require_columns(
        forecasts,
        {
            "model",
            "store_id",
            "item_id",
            "day_id",
            "date",
            "actual_units",
            "predicted_units",
        },
        path.name,
    )

    return forecasts[
        [
            "model",
            "store_id",
            "item_id",
            "day_id",
            "date",
            "actual_units",
            "predicted_units",
        ]
    ].copy()


def create_forecast_facts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create actual-demand and forecast facts."""

    baseline = load_forecast_file(
        BASELINE_FORECAST_FILE
    )

    adaptive = load_forecast_file(
        ADAPTIVE_FORECAST_FILE
    )

    machine_learning = load_forecast_file(
        ML_FORECAST_FILE
    )

    all_forecasts = pd.concat(
        [
            baseline,
            adaptive,
            machine_learning,
        ],
        ignore_index=True,
    )

    all_forecasts = all_forecasts.rename(
        columns={
            "model": "model_id",
        }
    )

    all_forecasts[
        "absolute_error"
    ] = (
        all_forecasts[
            "predicted_units"
        ]
        - all_forecasts[
            "actual_units"
        ]
    ).abs()

    all_forecasts["squared_error"] = (
        all_forecasts[
            "predicted_units"
        ]
        - all_forecasts[
            "actual_units"
        ]
    ) ** 2

    all_forecasts["forecast_error"] = (
        all_forecasts[
            "predicted_units"
        ]
        - all_forecasts[
            "actual_units"
        ]
    )

    all_forecasts["date_key"] = (
        all_forecasts["date"]
        .dt.strftime("%Y%m%d")
        .astype("int32")
    )

    validate_unique_key(
        all_forecasts,
        [
            "model_id",
            "store_id",
            "item_id",
            "day_id",
        ],
        "fact_forecast_daily",
    )

    unknown_models = (
        set(all_forecasts["model_id"])
        - set(MODEL_METADATA)
    )

    if unknown_models:
        raise ValueError(
            "Missing model metadata for: "
            + ", ".join(
                sorted(unknown_models)
            )
        )

    actual_daily = (
        machine_learning[
            [
                "store_id",
                "item_id",
                "day_id",
                "date",
                "actual_units",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "store_id",
                "item_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    actual_daily["date_key"] = (
        actual_daily["date"]
        .dt.strftime("%Y%m%d")
        .astype("int32")
    )

    validate_unique_key(
        actual_daily,
        [
            "store_id",
            "item_id",
            "day_id",
        ],
        "fact_actual_daily",
    )

    if (
        len(actual_daily)
        != EXPECTED_ACTUAL_DAILY_ROWS
    ):
        raise ValueError(
            f"Expected {EXPECTED_ACTUAL_DAILY_ROWS:,} "
            f"actual daily rows, but found "
            f"{len(actual_daily):,}."
        )

    actual_horizon = (
        actual_daily.groupby(
            [
                "store_id",
                "item_id",
            ],
            as_index=False,
        )
        .agg(
            actual_demand_units=(
                "actual_units",
                "sum",
            ),
            holdout_start_date=(
                "date",
                "min",
            ),
            holdout_end_date=(
                "date",
                "max",
            ),
            holdout_days=(
                "day_id",
                "nunique",
            ),
        )
    )

    validate_unique_key(
        actual_horizon,
        [
            "store_id",
            "item_id",
        ],
        "fact_actual_horizon",
    )

    if len(actual_horizon) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} horizon rows, "
            f"but found {len(actual_horizon)}."
        )

    forecast_fact = all_forecasts.drop(
        columns=["actual_units"]
    )

    return (
        actual_daily,
        actual_horizon,
        forecast_fact,
    )


def create_inventory_policy_fact() -> pd.DataFrame:
    """Create the unconstrained inventory-policy fact."""

    require_file(INVENTORY_POLICY_FILE)

    inventory = pd.read_csv(
        INVENTORY_POLICY_FILE
    )

    require_columns(
        inventory,
        {
            "forecast_model",
            "inventory_policy",
            "store_id",
            "item_id",
            "forecast_total_units",
            "uncertainty_std_28",
            "safety_stock_units",
            "order_quantity",
            "served_units",
            "shortage_units",
            "leftover_units",
            "stockout_occurred",
            "holding_cost_units",
            "shortage_cost_units",
            "total_cost_units",
        },
        INVENTORY_POLICY_FILE.name,
    )

    inventory = inventory.rename(
        columns={
            "forecast_model": "model_id",
            "inventory_policy": "policy_id",
        }
    )

    selected_columns = [
        "model_id",
        "policy_id",
        "store_id",
        "item_id",
        "forecast_total_units",
        "uncertainty_std_28",
        "service_factor",
        "safety_stock_units",
        "order_quantity",
        "served_units",
        "shortage_units",
        "leftover_units",
        "stockout_occurred",
        "holding_cost_units",
        "shortage_cost_units",
        "total_cost_units",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in inventory.columns
    ]

    inventory = inventory[
        selected_columns
    ].copy()

    validate_unique_key(
        inventory,
        [
            "model_id",
            "policy_id",
            "store_id",
            "item_id",
        ],
        "fact_inventory_policy",
    )

    unknown_policies = (
        set(inventory["policy_id"])
        - set(POLICY_METADATA)
    )

    if unknown_policies:
        raise ValueError(
            "Missing policy metadata for: "
            + ", ".join(
                sorted(unknown_policies)
            )
        )

    return inventory


def create_constrained_allocation_fact() -> pd.DataFrame:
    """Create the constrained-allocation fact."""

    require_file(CONSTRAINED_ALLOCATION_FILE)

    allocation = pd.read_csv(
        CONSTRAINED_ALLOCATION_FILE
    )

    require_columns(
        allocation,
        {
            "scenario",
            "allocation_method",
            "store_id",
            "item_id",
            "forecast_total_units",
            "target_order_quantity",
            "unit_purchase_cost",
            "allocated_order_quantity",
            "served_units",
            "shortage_units",
            "leftover_units",
            "stockout_occurred",
            "holding_cost_units",
            "shortage_cost_units",
            "operational_cost_units",
            "purchase_spend",
            "budget_limit",
            "capacity_limit",
        },
        CONSTRAINED_ALLOCATION_FILE.name,
    )

    allocation = allocation.rename(
        columns={
            "scenario": "scenario_id",
            "allocation_method": (
                "allocation_method_id"
            ),
        }
    )

    allocation["model_id"] = (
        "hist_gradient_boosting"
    )

    allocation["policy_id"] = (
        "newsvendor_adjusted_order"
    )

    selected_columns = [
        "scenario_id",
        "allocation_method_id",
        "model_id",
        "policy_id",
        "store_id",
        "item_id",
        "forecast_total_units",
        "uncertainty_std_28",
        "target_order_quantity",
        "unit_purchase_cost",
        "allocated_order_quantity",
        "served_units",
        "shortage_units",
        "leftover_units",
        "stockout_occurred",
        "holding_cost_units",
        "shortage_cost_units",
        "operational_cost_units",
        "purchase_spend",
        "budget_limit",
        "capacity_limit",
        "target_coverage_percent",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in allocation.columns
    ]

    allocation = allocation[
        selected_columns
    ].copy()

    validate_unique_key(
        allocation,
        [
            "scenario_id",
            "allocation_method_id",
            "store_id",
            "item_id",
        ],
        "fact_constrained_allocation",
    )

    unknown_scenarios = (
        set(allocation["scenario_id"])
        - set(SCENARIO_METADATA)
    )

    if unknown_scenarios:
        raise ValueError(
            "Missing scenario metadata for: "
            + ", ".join(
                sorted(unknown_scenarios)
            )
        )

    unknown_methods = (
        set(
            allocation[
                "allocation_method_id"
            ]
        )
        - set(ALLOCATION_METHOD_METADATA)
    )

    if unknown_methods:
        raise ValueError(
            "Missing allocation-method metadata for: "
            + ", ".join(
                sorted(unknown_methods)
            )
        )

    return allocation


def validate_foreign_keys(
    stores: pd.DataFrame,
    items: pd.DataFrame,
    dates: pd.DataFrame,
    actual_daily: pd.DataFrame,
    forecast_daily: pd.DataFrame,
    inventory_policy: pd.DataFrame,
    constrained_allocation: pd.DataFrame,
) -> None:
    """Validate fact-to-dimension key coverage."""

    store_ids = set(
        stores["store_id"]
    )

    item_ids = set(
        items["item_id"]
    )

    day_ids = set(
        dates["day_id"]
    )

    fact_tables = {
        "fact_actual_daily": actual_daily,
        "fact_forecast_daily": forecast_daily,
        "fact_inventory_policy": (
            inventory_policy
        ),
        "fact_constrained_allocation": (
            constrained_allocation
        ),
    }

    for table_name, dataframe in fact_tables.items():
        unknown_stores = (
            set(dataframe["store_id"])
            - store_ids
        )

        unknown_items = (
            set(dataframe["item_id"])
            - item_ids
        )

        if unknown_stores:
            raise ValueError(
                f"{table_name} contains unknown stores: "
                f"{sorted(unknown_stores)}"
            )

        if unknown_items:
            raise ValueError(
                f"{table_name} contains unknown items: "
                f"{sorted(unknown_items)}"
            )

    for table_name, dataframe in {
        "fact_actual_daily": actual_daily,
        "fact_forecast_daily": forecast_daily,
    }.items():
        unknown_days = (
            set(dataframe["day_id"])
            - day_ids
        )

        if unknown_days:
            raise ValueError(
                f"{table_name} contains unknown days: "
                f"{sorted(unknown_days)[:10]}"
            )


def main() -> None:
    """Export the complete Power BI star schema."""

    print("=" * 82)
    print("EXPORT POWER BI STAR SCHEMA")
    print("=" * 82)

    manifest_rows: list[
        dict[str, object]
    ] = []

    stores = create_store_dimension()
    items = create_item_dimension()
    dates = create_date_dimension()

    models = metadata_dimension(
        MODEL_METADATA,
        "model_id",
    )

    policies = metadata_dimension(
        POLICY_METADATA,
        "policy_id",
    )

    scenarios = metadata_dimension(
        SCENARIO_METADATA,
        "scenario_id",
    )

    allocation_methods = (
        metadata_dimension(
            ALLOCATION_METHOD_METADATA,
            "allocation_method_id",
        )
    )

    (
        actual_daily,
        actual_horizon,
        forecast_daily,
    ) = create_forecast_facts()

    inventory_policy = (
        create_inventory_policy_fact()
    )

    constrained_allocation = (
        create_constrained_allocation_fact()
    )

    validate_foreign_keys(
        stores,
        items,
        dates,
        actual_daily,
        forecast_daily,
        inventory_policy,
        constrained_allocation,
    )

    print("\nSaving dimensions...")
    print("-" * 82)

    save_table(
        stores,
        "dim_store.csv",
        "One row per store",
        manifest_rows,
    )

    save_table(
        items,
        "dim_item.csv",
        "One row per product",
        manifest_rows,
    )

    save_table(
        dates,
        "dim_date.csv",
        "One row per calendar day",
        manifest_rows,
    )

    save_table(
        models,
        "dim_model.csv",
        "One row per forecast model",
        manifest_rows,
    )

    save_table(
        policies,
        "dim_inventory_policy.csv",
        "One row per inventory policy",
        manifest_rows,
    )

    save_table(
        scenarios,
        "dim_constraint_scenario.csv",
        "One row per resource-constraint scenario",
        manifest_rows,
    )

    save_table(
        allocation_methods,
        "dim_allocation_method.csv",
        "One row per inventory-allocation method",
        manifest_rows,
    )

    print("\nSaving fact tables...")
    print("-" * 82)

    save_table(
        actual_daily,
        "fact_actual_daily.csv",
        (
            "One row per store, product, "
            "and holdout day"
        ),
        manifest_rows,
    )

    save_table(
        actual_horizon,
        "fact_actual_horizon.csv",
        (
            "One row per store and product "
            "for the 28-day holdout"
        ),
        manifest_rows,
    )

    save_table(
        forecast_daily,
        "fact_forecast_daily.csv",
        (
            "One row per model, store, "
            "product, and holdout day"
        ),
        manifest_rows,
    )

    save_table(
        inventory_policy,
        "fact_inventory_policy.csv",
        (
            "One row per model, policy, "
            "store, and product"
        ),
        manifest_rows,
    )

    save_table(
        constrained_allocation,
        "fact_constrained_allocation.csv",
        (
            "One row per constraint scenario, "
            "allocation method, store, and product"
        ),
        manifest_rows,
    )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        POWERBI_DIRECTORY
        / "powerbi_export_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] "
        f"{manifest_path.name:<42} "
        f"rows={len(manifest):>7,} "
        f"columns={len(manifest.columns):>2}"
    )

    print("\nPower BI export summary")
    print("-" * 82)

    print(
        manifest.to_string(
            index=False
        )
    )

    print(
        "\nPower BI star-schema export "
        "completed successfully."
    )


if __name__ == "__main__":
    main()