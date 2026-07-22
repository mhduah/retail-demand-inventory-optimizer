"""Optimise inventory allocation under budget and capacity constraints.

The unconstrained target is the histogram-gradient-boosting newsvendor
order quantity. A mixed-integer linear programme allocates limited
inventory across products and stores.

Actual holdout demand is used only after optimisation for evaluation.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


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

POLICY_FILE = (
    REPORT_DIRECTORY
    / "inventory_policy_by_series.csv"
)

ML_FORECAST_FILE = (
    REPORT_DIRECTORY
    / "ml_forecasts.csv"
)

SOURCE_MODEL = "hist_gradient_boosting"
SOURCE_POLICY = "newsvendor_adjusted_order"

EXPECTED_STORES = {
    "CA_1",
    "TX_1",
    "WI_1",
}

EXPECTED_SERIES = 150

# Simulated procurement cost assumption.
PROCUREMENT_COST_SHARE_OF_SELL_PRICE = 0.60

# Relative operating-cost assumptions.
HOLDING_COST_PER_UNIT = 1.0
SHORTAGE_COST_PER_UNIT = 5.0

# Every product receives at least this fraction of its target.
MINIMUM_TARGET_COVERAGE = 0.40

# Ratios are relative to the unconstrained ML-newsvendor plan.
SCENARIOS = {
    "tight": {
        "budget_ratio": 0.65,
        "capacity_ratio": 0.70,
    },
    "balanced": {
        "budget_ratio": 0.80,
        "capacity_ratio": 0.85,
    },
    "near_full": {
        "budget_ratio": 0.95,
        "capacity_ratio": 0.95,
    },
}


def load_target_plan() -> pd.DataFrame:
    """Load the selected unconstrained inventory target."""

    if not POLICY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory policy file not found:\n{POLICY_FILE}"
        )

    policies = pd.read_csv(POLICY_FILE)

    required_columns = {
        "forecast_model",
        "inventory_policy",
        "store_id",
        "item_id",
        "forecast_total_units",
        "actual_demand_units",
        "uncertainty_std_28",
        "order_quantity",
    }

    missing_columns = required_columns - set(
        policies.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing policy columns: "
            + ", ".join(sorted(missing_columns))
        )

    targets = policies.loc[
        policies["forecast_model"].eq(SOURCE_MODEL)
        & policies["inventory_policy"].eq(SOURCE_POLICY)
    ].copy()

    if len(targets) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} target rows, "
            f"but found {len(targets)}."
        )

    if targets.duplicated(
        subset=["store_id", "item_id"]
    ).any():
        raise ValueError(
            "Duplicate product-store targets were found."
        )

    if set(targets["store_id"]) != EXPECTED_STORES:
        raise ValueError(
            "Unexpected store coverage in target plan."
        )

    targets["target_order_quantity"] = (
        pd.to_numeric(
            targets["order_quantity"],
            errors="raise",
        )
        .round()
        .astype("int32")
    )

    if (
        targets["target_order_quantity"] < 0
    ).any():
        raise ValueError(
            "Negative target quantities were found."
        )

    return targets


def load_holdout_dates() -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """Obtain the evaluation period from the ML forecasts."""

    if not ML_FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"ML forecast file not found:\n{ML_FORECAST_FILE}"
        )

    forecasts = pd.read_csv(
        ML_FORECAST_FILE,
        usecols=["date"],
        parse_dates=["date"],
    )

    return (
        forecasts["date"].min(),
        forecasts["date"].max(),
    )


def load_average_prices(
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load holdout-period prices with a historical fallback."""

    query = """
        WITH holdout_weeks AS (
            SELECT DISTINCT
                wm_yr_wk
            FROM dim_calendar
            WHERE date BETWEEN ? AND ?
        )
        SELECT
            p.store_id,
            p.item_id,
            AVG(
                CASE
                    WHEN p.wm_yr_wk IN (
                        SELECT wm_yr_wk
                        FROM holdout_weeks
                    )
                    THEN p.sell_price
                END
            ) AS holdout_average_price,
            AVG(p.sell_price) AS historical_average_price
        FROM fact_prices AS p
        GROUP BY
            p.store_id,
            p.item_id;
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        prices = pd.read_sql_query(
            query,
            connection,
            params=[
                holdout_start.strftime("%Y-%m-%d"),
                holdout_end.strftime("%Y-%m-%d"),
            ],
        )

    prices["average_sell_price"] = (
        prices["holdout_average_price"]
        .fillna(
            prices["historical_average_price"]
        )
    )

    if prices[
        "average_sell_price"
    ].isna().any():
        raise ValueError(
            "Some products have no usable selling price."
        )

    if (
        prices["average_sell_price"] <= 0
    ).any():
        raise ValueError(
            "Non-positive prices were found."
        )

    return prices[
        [
            "store_id",
            "item_id",
            "average_sell_price",
        ]
    ]


def prepare_optimisation_data(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Merge costs and calculate priority weights."""

    data = targets.merge(
        prices,
        on=["store_id", "item_id"],
        how="left",
        validate="one_to_one",
    )

    if data["average_sell_price"].isna().any():
        raise ValueError(
            "Some target rows could not be matched to prices."
        )

    data["unit_purchase_cost"] = (
        data["average_sell_price"]
        * PROCUREMENT_COST_SHARE_OF_SELL_PRICE
    )

    maximum_forecast = (
        data.groupby("store_id")[
            "forecast_total_units"
        ]
        .transform("max")
        .replace(0, 1)
    )

    maximum_uncertainty = (
        data.groupby("store_id")[
            "uncertainty_std_28"
        ]
        .transform("max")
        .replace(0, 1)
    )

    demand_score = (
        data["forecast_total_units"]
        / maximum_forecast
    )

    uncertainty_score = (
        data["uncertainty_std_28"]
        / maximum_uncertainty
    )

    # Priority ranges approximately from 1 to 2.
    data["priority_weight"] = (
        1.0
        + 0.5 * demand_score
        + 0.5 * uncertainty_score
    )

    return data.sort_values(
        ["store_id", "item_id"]
    ).reset_index(drop=True)


def calculate_limits(
    store_data: pd.DataFrame,
    scenario: dict[str, float],
) -> dict[str, float]:
    """Calculate store-level budget and capacity limits."""

    target = store_data[
        "target_order_quantity"
    ].to_numpy(dtype=float)

    unit_cost = store_data[
        "unit_purchase_cost"
    ].to_numpy(dtype=float)

    unconstrained_units = int(
        target.sum()
    )

    unconstrained_spend = float(
        np.dot(
            target,
            unit_cost,
        )
    )

    capacity_limit = int(
        np.floor(
            scenario["capacity_ratio"]
            * unconstrained_units
        )
    )

    budget_limit = float(
        scenario["budget_ratio"]
        * unconstrained_spend
    )

    return {
        "unconstrained_units": (
            unconstrained_units
        ),
        "unconstrained_spend": (
            unconstrained_spend
        ),
        "capacity_limit": (
            capacity_limit
        ),
        "budget_limit": (
            budget_limit
        ),
    }


def solve_milp_allocation(
    store_data: pd.DataFrame,
    limits: dict[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the constrained integer allocation problem."""

    target = store_data[
        "target_order_quantity"
    ].to_numpy(dtype=float)

    unit_cost = store_data[
        "unit_purchase_cost"
    ].to_numpy(dtype=float)

    priority = store_data[
        "priority_weight"
    ].to_numpy(dtype=float)

    number_of_products = len(store_data)

    minimum_orders = np.floor(
        MINIMUM_TARGET_COVERAGE
        * target
    )

    minimum_spend = float(
        np.dot(
            minimum_orders,
            unit_cost,
        )
    )

    minimum_units = int(
        minimum_orders.sum()
    )

    if (
        minimum_spend
        > limits["budget_limit"] + 1e-8
    ):
        raise ValueError(
            "Minimum allocations exceed the budget."
        )

    if (
        minimum_units
        > limits["capacity_limit"]
    ):
        raise ValueError(
            "Minimum allocations exceed capacity."
        )

    # Variables:
    # q_1,...,q_n = integer allocated quantities
    # s_1,...,s_n = continuous target shortfalls
    number_of_variables = (
        2 * number_of_products
    )

    objective = np.concatenate(
        [
            np.zeros(number_of_products),
            priority,
        ]
    )

    integrality = np.concatenate(
        [
            np.ones(
                number_of_products,
                dtype=int,
            ),
            np.zeros(
                number_of_products,
                dtype=int,
            ),
        ]
    )

    lower_bounds = np.concatenate(
        [
            minimum_orders,
            np.zeros(number_of_products),
        ]
    )

    upper_bounds = np.concatenate(
        [
            target,
            target,
        ]
    )

    constraint_matrix = np.zeros(
        (
            number_of_products + 2,
            number_of_variables,
        ),
        dtype=float,
    )

    row_indices = np.arange(
        number_of_products
    )

    # q_i + s_i = target_i
    constraint_matrix[
        row_indices,
        row_indices,
    ] = 1.0

    constraint_matrix[
        row_indices,
        number_of_products + row_indices,
    ] = 1.0

    # Purchasing budget.
    constraint_matrix[
        number_of_products,
        :number_of_products,
    ] = unit_cost

    # Unit-based storage capacity.
    constraint_matrix[
        number_of_products + 1,
        :number_of_products,
    ] = 1.0

    lower_constraints = np.concatenate(
        [
            target,
            [-np.inf, -np.inf],
        ]
    )

    upper_constraints = np.concatenate(
        [
            target,
            [
                limits["budget_limit"],
                limits["capacity_limit"],
            ],
        ]
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(
            lower_bounds,
            upper_bounds,
        ),
        constraints=LinearConstraint(
            constraint_matrix,
            lower_constraints,
            upper_constraints,
        ),
        options={
            "disp": False,
            "time_limit": 60.0,
            "mip_rel_gap": 0.0,
        },
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            "MILP optimisation failed: "
            f"{result.message}"
        )

    allocation = np.rint(
        result.x[:number_of_products]
    ).astype("int32")

    allocated_spend = float(
        np.dot(
            allocation,
            unit_cost,
        )
    )

    if (
        allocated_spend
        > limits["budget_limit"] + 1e-5
    ):
        raise ValueError(
            "MILP allocation violates the budget."
        )

    if (
        allocation.sum()
        > limits["capacity_limit"]
    ):
        raise ValueError(
            "MILP allocation violates capacity."
        )

    status = {
        "solver_success": bool(
            result.success
        ),
        "solver_status": int(
            result.status
        ),
        "solver_message": str(
            result.message
        ),
        "objective_value": float(
            result.fun
        ),
    }

    return allocation, status


def create_proportional_allocation(
    store_data: pd.DataFrame,
    limits: dict[str, float],
) -> np.ndarray:
    """Create a feasible proportional-allocation benchmark."""

    target = store_data[
        "target_order_quantity"
    ].to_numpy(dtype=int)

    unit_cost = store_data[
        "unit_purchase_cost"
    ].to_numpy(dtype=float)

    priority = store_data[
        "priority_weight"
    ].to_numpy(dtype=float)

    allocation = np.floor(
        MINIMUM_TARGET_COVERAGE
        * target
    ).astype(int)

    remaining_target = (
        target - allocation
    )

    remaining_budget = (
        limits["budget_limit"]
        - np.dot(
            allocation,
            unit_cost,
        )
    )

    remaining_capacity = (
        limits["capacity_limit"]
        - allocation.sum()
    )

    desired_spend = float(
        np.dot(
            remaining_target,
            unit_cost,
        )
    )

    desired_units = int(
        remaining_target.sum()
    )

    scale_candidates = [1.0]

    if desired_spend > 0:
        scale_candidates.append(
            remaining_budget
            / desired_spend
        )

    if desired_units > 0:
        scale_candidates.append(
            remaining_capacity
            / desired_units
        )

    scale = max(
        0.0,
        min(scale_candidates),
    )

    allocation += np.floor(
        scale * remaining_target
    ).astype(int)

    # Use remaining budget created by integer rounding.
    priority_per_cost = (
        priority
        / unit_cost
    )

    allocation_order = np.argsort(
        -priority_per_cost
    )

    while True:
        unit_added = False

        current_units = int(
            allocation.sum()
        )

        current_spend = float(
            np.dot(
                allocation,
                unit_cost,
            )
        )

        for index in allocation_order:
            if allocation[index] >= target[index]:
                continue

            if (
                current_units + 1
                > limits["capacity_limit"]
            ):
                continue

            if (
                current_spend
                + unit_cost[index]
                > limits["budget_limit"] + 1e-8
            ):
                continue

            allocation[index] += 1
            current_units += 1
            current_spend += (
                unit_cost[index]
            )
            unit_added = True

        if not unit_added:
            break

    return allocation.astype("int32")


def evaluate_allocation(
    store_data: pd.DataFrame,
    allocation: np.ndarray,
    scenario_name: str,
    allocation_method: str,
    limits: dict[str, float],
) -> pd.DataFrame:
    """Evaluate an allocation using the untouched actual demand."""

    result = store_data.copy()

    result["scenario"] = scenario_name

    result["allocation_method"] = (
        allocation_method
    )

    result["allocated_order_quantity"] = (
        allocation
    )

    actual = result[
        "actual_demand_units"
    ].to_numpy(dtype=float)

    allocated = allocation.astype(float)

    served = np.minimum(
        allocated,
        actual,
    )

    shortage = np.maximum(
        actual - allocated,
        0.0,
    )

    leftover = np.maximum(
        allocated - actual,
        0.0,
    )

    result["served_units"] = served
    result["shortage_units"] = shortage
    result["leftover_units"] = leftover

    result["stockout_occurred"] = (
        shortage > 0
    ).astype("int8")

    result["fill_rate_percent"] = np.where(
        actual == 0,
        100.0,
        served / actual * 100,
    )

    result["holding_cost_units"] = (
        leftover
        * HOLDING_COST_PER_UNIT
    )

    result["shortage_cost_units"] = (
        shortage
        * SHORTAGE_COST_PER_UNIT
    )

    result["operational_cost_units"] = (
        result["holding_cost_units"]
        + result["shortage_cost_units"]
    )

    result["purchase_spend"] = (
        result["allocated_order_quantity"]
        * result["unit_purchase_cost"]
    )

    target = result[
        "target_order_quantity"
    ].to_numpy(dtype=float)

    result["target_coverage_percent"] = (
        np.where(
            target == 0,
            100.0,
            allocated / target * 100,
        )
    )

    result["budget_limit"] = (
        limits["budget_limit"]
    )

    result["capacity_limit"] = (
        limits["capacity_limit"]
    )

    result["unconstrained_spend"] = (
        limits["unconstrained_spend"]
    )

    result["unconstrained_units"] = (
        limits["unconstrained_units"]
    )

    return result


def create_store_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise constrained results by store."""

    summary = (
        results.groupby(
            [
                "scenario",
                "allocation_method",
                "store_id",
            ],
            sort=True,
        )
        .agg(
            number_of_series=(
                "item_id",
                "size",
            ),
            target_order_units=(
                "target_order_quantity",
                "sum",
            ),
            allocated_order_units=(
                "allocated_order_quantity",
                "sum",
            ),
            actual_demand_units=(
                "actual_demand_units",
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
            purchase_spend=(
                "purchase_spend",
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
            operational_cost_units=(
                "operational_cost_units",
                "sum",
            ),
            budget_limit=(
                "budget_limit",
                "first",
            ),
            capacity_limit=(
                "capacity_limit",
                "first",
            ),
            unconstrained_spend=(
                "unconstrained_spend",
                "first",
            ),
            unconstrained_units=(
                "unconstrained_units",
                "first",
            ),
        )
        .reset_index()
    )

    return add_summary_metrics(summary)


def add_summary_metrics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived performance measures."""

    summary = summary.copy()

    summary["fill_rate_percent"] = np.where(
        summary["actual_demand_units"].eq(0),
        100.0,
        (
            summary["served_units"]
            / summary["actual_demand_units"]
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
        "target_coverage_percent"
    ] = (
        summary["allocated_order_units"]
        / summary["target_order_units"]
        * 100
    )

    summary[
        "budget_utilisation_percent"
    ] = (
        summary["purchase_spend"]
        / summary["budget_limit"]
        * 100
    )

    summary[
        "capacity_utilisation_percent"
    ] = (
        summary["allocated_order_units"]
        / summary["capacity_limit"]
        * 100
    )

    summary[
        "average_cost_per_series"
    ] = (
        summary["operational_cost_units"]
        / summary["number_of_series"]
    )

    numeric_columns = summary.select_dtypes(
        include=[np.number]
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(2)
    )

    return summary


def create_overall_summary(
    store_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate store summaries into scenario summaries."""

    overall = (
        store_summary.groupby(
            [
                "scenario",
                "allocation_method",
            ],
            sort=True,
        )
        .agg(
            number_of_series=(
                "number_of_series",
                "sum",
            ),
            target_order_units=(
                "target_order_units",
                "sum",
            ),
            allocated_order_units=(
                "allocated_order_units",
                "sum",
            ),
            actual_demand_units=(
                "actual_demand_units",
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
                "stockout_series",
                "sum",
            ),
            purchase_spend=(
                "purchase_spend",
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
            operational_cost_units=(
                "operational_cost_units",
                "sum",
            ),
            budget_limit=(
                "budget_limit",
                "sum",
            ),
            capacity_limit=(
                "capacity_limit",
                "sum",
            ),
            unconstrained_spend=(
                "unconstrained_spend",
                "sum",
            ),
            unconstrained_units=(
                "unconstrained_units",
                "sum",
            ),
        )
        .reset_index()
    )

    return add_summary_metrics(overall)


def create_assumptions_table() -> pd.DataFrame:
    """Create tracked optimisation assumptions."""

    rows = [
        {
            "assumption": "source_forecast_model",
            "value": SOURCE_MODEL,
        },
        {
            "assumption": "source_inventory_policy",
            "value": SOURCE_POLICY,
        },
        {
            "assumption": (
                "procurement_cost_share_of_sell_price"
            ),
            "value": (
                PROCUREMENT_COST_SHARE_OF_SELL_PRICE
            ),
        },
        {
            "assumption": "minimum_target_coverage",
            "value": MINIMUM_TARGET_COVERAGE,
        },
        {
            "assumption": "holding_cost_per_unit",
            "value": HOLDING_COST_PER_UNIT,
        },
        {
            "assumption": "shortage_cost_per_unit",
            "value": SHORTAGE_COST_PER_UNIT,
        },
        {
            "assumption": "capacity_interpretation",
            "value": (
                "Unit-count proxy because product "
                "volume data is unavailable"
            ),
        },
        {
            "assumption": "cost_interpretation",
            "value": (
                "Procurement spend is simulated; "
                "operational costs are relative units"
            ),
        },
    ]

    for scenario_name, scenario in (
        SCENARIOS.items()
    ):
        rows.extend(
            [
                {
                    "assumption": (
                        f"{scenario_name}_budget_ratio"
                    ),
                    "value": (
                        scenario["budget_ratio"]
                    ),
                },
                {
                    "assumption": (
                        f"{scenario_name}_capacity_ratio"
                    ),
                    "value": (
                        scenario["capacity_ratio"]
                    ),
                },
            ]
        )

    return pd.DataFrame(rows)


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save one tracked result table."""

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
        f"[SAVED] {filename:<43} "
        f"rows={len(dataframe):>5,}"
    )


def main() -> None:
    """Run all constrained-allocation scenarios."""

    print("=" * 82)
    print("CONSTRAINED INVENTORY ALLOCATION")
    print("=" * 82)

    targets = load_target_plan()

    (
        holdout_start,
        holdout_end,
    ) = load_holdout_dates()

    prices = load_average_prices(
        holdout_start,
        holdout_end,
    )

    optimisation_data = (
        prepare_optimisation_data(
            targets,
            prices,
        )
    )

    result_frames = []
    solver_status_rows = []

    for scenario_name, scenario in (
        SCENARIOS.items()
    ):
        print(
            f"\nScenario: {scenario_name}"
        )
        print("-" * 82)

        for store_id in sorted(
            EXPECTED_STORES
        ):
            store_data = (
                optimisation_data.loc[
                    optimisation_data[
                        "store_id"
                    ].eq(store_id)
                ]
                .copy()
                .reset_index(drop=True)
            )

            limits = calculate_limits(
                store_data,
                scenario,
            )

            milp_allocation, status = (
                solve_milp_allocation(
                    store_data,
                    limits,
                )
            )

            proportional_allocation = (
                create_proportional_allocation(
                    store_data,
                    limits,
                )
            )

            result_frames.append(
                evaluate_allocation(
                    store_data,
                    milp_allocation,
                    scenario_name,
                    "milp_optimised",
                    limits,
                )
            )

            result_frames.append(
                evaluate_allocation(
                    store_data,
                    proportional_allocation,
                    scenario_name,
                    "proportional_heuristic",
                    limits,
                )
            )

            solver_status_rows.append(
                {
                    "scenario": scenario_name,
                    "store_id": store_id,
                    **status,
                }
            )

            print(
                f"[SOLVED] {store_id} "
                f"budget={limits['budget_limit']:.2f} "
                f"capacity={limits['capacity_limit']}"
            )

    allocation_results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    store_summary = create_store_summary(
        allocation_results
    )

    overall_summary = create_overall_summary(
        store_summary
    )

    assumptions = (
        create_assumptions_table()
    )

    solver_status = pd.DataFrame(
        solver_status_rows
    )

    display_columns = [
        "scenario",
        "allocation_method",
        "allocated_order_units",
        "shortage_units",
        "leftover_units",
        "fill_rate_percent",
        "stockout_series_percent",
        "operational_cost_units",
        "budget_utilisation_percent",
        "capacity_utilisation_percent",
    ]

    print("\nOverall scenario comparison")
    print("-" * 82)

    print(
        overall_summary[
            display_columns
        ]
        .sort_values(
            [
                "scenario",
                "operational_cost_units",
            ]
        )
        .to_string(
            index=False
        )
    )

    print("\nSaving results...")
    print("-" * 82)

    save_table(
        allocation_results,
        "constrained_allocation_by_series.csv",
    )

    save_table(
        overall_summary,
        "constrained_allocation_summary.csv",
    )

    save_table(
        store_summary,
        "constrained_allocation_by_store.csv",
    )

    save_table(
        assumptions,
        "constrained_optimization_assumptions.csv",
    )

    save_table(
        solver_status,
        "constrained_solver_status.csv",
    )

    print(
        "\nConstrained inventory optimisation completed."
    )


if __name__ == "__main__":
    main()