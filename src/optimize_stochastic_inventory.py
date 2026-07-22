"""Optimise constrained inventory using expected operating cost.

The stochastic MILP minimises expected shortage and leftover costs
across simulated 28-day demand scenarios.

Actual holdout demand is excluded from the optimisation and is used
only after solving to evaluate the resulting allocation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

BENCHMARK_FILE = (
    REPORT_DIRECTORY
    / "constrained_allocation_by_series.csv"
)

EXPECTED_STORES = {
    "CA_1",
    "TX_1",
    "WI_1",
}

EXPECTED_SERIES = 150

REFERENCE_METHOD = "proportional_heuristic"
STOCHASTIC_METHOD = "stochastic_expected_cost_milp"

HOLDING_COST_PER_UNIT = 1.0
SHORTAGE_COST_PER_UNIT = 5.0

MINIMUM_TARGET_COVERAGE = 0.40

# Symmetric discrete approximation to forecast uncertainty.
SCENARIO_DEFINITIONS = [
    {
        "scenario_id": "very_low",
        "z_score": -1.2816,
        "probability": 0.10,
    },
    {
        "scenario_id": "low",
        "z_score": -0.5244,
        "probability": 0.20,
    },
    {
        "scenario_id": "central",
        "z_score": 0.0000,
        "probability": 0.40,
    },
    {
        "scenario_id": "high",
        "z_score": 0.5244,
        "probability": 0.20,
    },
    {
        "scenario_id": "very_high",
        "z_score": 1.2816,
        "probability": 0.10,
    },
]


def load_benchmark_results() -> pd.DataFrame:
    """Load and validate the earlier allocation results."""

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark allocation file not found:\n"
            f"{BENCHMARK_FILE}"
        )

    results = pd.read_csv(
        BENCHMARK_FILE
    )

    required_columns = {
        "scenario",
        "allocation_method",
        "store_id",
        "item_id",
        "forecast_total_units",
        "actual_demand_units",
        "uncertainty_std_28",
        "target_order_quantity",
        "unit_purchase_cost",
        "allocated_order_quantity",
        "budget_limit",
        "capacity_limit",
        "unconstrained_spend",
        "unconstrained_units",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing benchmark columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if results.duplicated(
        subset=[
            "scenario",
            "allocation_method",
            "store_id",
            "item_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate benchmark allocation rows were found."
        )

    if set(results["store_id"]) != EXPECTED_STORES:
        raise ValueError(
            "Unexpected store coverage."
        )

    return results


def create_demand_scenarios(
    store_data: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Create simulated demand scenarios for each product."""

    scenario_rows = []

    for row in store_data.itertuples(
        index=False
    ):
        forecast_mean = float(
            row.forecast_total_units
        )

        uncertainty = float(
            row.uncertainty_std_28
        )

        for definition in SCENARIO_DEFINITIONS:
            simulated_demand = max(
                0.0,
                forecast_mean
                + definition["z_score"]
                * uncertainty,
            )

            scenario_rows.append(
                {
                    "constraint_scenario": (
                        scenario_name
                    ),
                    "store_id": row.store_id,
                    "item_id": row.item_id,
                    "demand_scenario": (
                        definition[
                            "scenario_id"
                        ]
                    ),
                    "z_score": (
                        definition["z_score"]
                    ),
                    "probability": (
                        definition[
                            "probability"
                        ]
                    ),
                    "forecast_mean_units": (
                        forecast_mean
                    ),
                    "uncertainty_std_28": (
                        uncertainty
                    ),
                    "simulated_demand_units": (
                        simulated_demand
                    ),
                }
            )

    scenarios = pd.DataFrame(
        scenario_rows
    )

    probability_check = (
        scenarios.groupby(
            [
                "constraint_scenario",
                "store_id",
                "item_id",
            ]
        )["probability"]
        .sum()
    )

    if not np.allclose(
        probability_check.to_numpy(),
        1.0,
    ):
        raise ValueError(
            "Demand-scenario probabilities do not sum to one."
        )

    return scenarios


def solve_stochastic_milp(
    store_data: pd.DataFrame,
    demand_scenarios: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the expected-operating-cost stochastic MILP."""

    store_data = (
        store_data.sort_values("item_id")
        .reset_index(drop=True)
    )

    item_ids = store_data[
        "item_id"
    ].tolist()

    target = store_data[
        "target_order_quantity"
    ].to_numpy(dtype=float)

    unit_cost = store_data[
        "unit_purchase_cost"
    ].to_numpy(dtype=float)

    minimum_order = np.floor(
        MINIMUM_TARGET_COVERAGE
        * target
    )

    budget_limit = float(
        store_data[
            "budget_limit"
        ].iloc[0]
    )

    capacity_limit = float(
        store_data[
            "capacity_limit"
        ].iloc[0]
    )

    if np.dot(
        minimum_order,
        unit_cost,
    ) > budget_limit + 1e-8:
        raise ValueError(
            "Minimum order quantities exceed the budget."
        )

    if (
        minimum_order.sum()
        > capacity_limit
    ):
        raise ValueError(
            "Minimum order quantities exceed capacity."
        )

    scenario_names = [
        definition["scenario_id"]
        for definition in SCENARIO_DEFINITIONS
    ]

    scenario_probabilities = np.asarray(
        [
            definition["probability"]
            for definition in SCENARIO_DEFINITIONS
        ],
        dtype=float,
    )

    number_of_items = len(item_ids)
    number_of_scenarios = len(
        scenario_names
    )

    demand_matrix = np.zeros(
        (
            number_of_items,
            number_of_scenarios,
        ),
        dtype=float,
    )

    for item_index, item_id in enumerate(
        item_ids
    ):
        item_scenarios = (
            demand_scenarios.loc[
                demand_scenarios[
                    "item_id"
                ].eq(item_id)
            ]
            .set_index(
                "demand_scenario"
            )
        )

        for scenario_index, scenario_id in enumerate(
            scenario_names
        ):
            demand_matrix[
                item_index,
                scenario_index,
            ] = item_scenarios.loc[
                scenario_id,
                "simulated_demand_units",
            ]

    # Variables:
    # q_i: integer order quantities
    # u_ik: continuous shortage quantities
    # o_ik: continuous leftover quantities
    number_of_recourse_variables = (
        number_of_items
        * number_of_scenarios
    )

    number_of_variables = (
        number_of_items
        + 2
        * number_of_recourse_variables
    )

    order_start = 0

    shortage_start = (
        number_of_items
    )

    leftover_start = (
        shortage_start
        + number_of_recourse_variables
    )

    objective = np.zeros(
        number_of_variables,
        dtype=float,
    )

    for item_index in range(
        number_of_items
    ):
        for scenario_index in range(
            number_of_scenarios
        ):
            recourse_index = (
                item_index
                * number_of_scenarios
                + scenario_index
            )

            probability = (
                scenario_probabilities[
                    scenario_index
                ]
            )

            objective[
                shortage_start
                + recourse_index
            ] = (
                probability
                * SHORTAGE_COST_PER_UNIT
            )

            objective[
                leftover_start
                + recourse_index
            ] = (
                probability
                * HOLDING_COST_PER_UNIT
            )

    integrality = np.zeros(
        number_of_variables,
        dtype=int,
    )

    integrality[
        order_start:number_of_items
    ] = 1

    lower_bounds = np.zeros(
        number_of_variables,
        dtype=float,
    )

    upper_bounds = np.full(
        number_of_variables,
        np.inf,
        dtype=float,
    )

    lower_bounds[
        order_start:number_of_items
    ] = minimum_order

    upper_bounds[
        order_start:number_of_items
    ] = target

    # Two recourse constraints for every item-scenario pair:
    #
    # q_i + u_ik >= d_ik
    # q_i - o_ik <= d_ik
    number_of_recourse_constraints = (
        2
        * number_of_recourse_variables
    )

    number_of_constraints = (
        number_of_recourse_constraints
        + 2
    )

    constraint_matrix = np.zeros(
        (
            number_of_constraints,
            number_of_variables,
        ),
        dtype=float,
    )

    constraint_lower = np.full(
        number_of_constraints,
        -np.inf,
        dtype=float,
    )

    constraint_upper = np.full(
        number_of_constraints,
        np.inf,
        dtype=float,
    )

    constraint_row = 0

    for item_index in range(
        number_of_items
    ):
        for scenario_index in range(
            number_of_scenarios
        ):
            recourse_index = (
                item_index
                * number_of_scenarios
                + scenario_index
            )

            simulated_demand = (
                demand_matrix[
                    item_index,
                    scenario_index,
                ]
            )

            # q_i + u_ik >= demand
            constraint_matrix[
                constraint_row,
                order_start + item_index,
            ] = 1.0

            constraint_matrix[
                constraint_row,
                shortage_start
                + recourse_index,
            ] = 1.0

            constraint_lower[
                constraint_row
            ] = simulated_demand

            constraint_row += 1

            # q_i - o_ik <= demand
            constraint_matrix[
                constraint_row,
                order_start + item_index,
            ] = 1.0

            constraint_matrix[
                constraint_row,
                leftover_start
                + recourse_index,
            ] = -1.0

            constraint_upper[
                constraint_row
            ] = simulated_demand

            constraint_row += 1

    # Budget constraint.
    constraint_matrix[
        constraint_row,
        order_start:number_of_items,
    ] = unit_cost

    constraint_upper[
        constraint_row
    ] = budget_limit

    constraint_row += 1

    # Capacity constraint.
    constraint_matrix[
        constraint_row,
        order_start:number_of_items,
    ] = 1.0

    constraint_upper[
        constraint_row
    ] = capacity_limit

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(
            lower_bounds,
            upper_bounds,
        ),
        constraints=LinearConstraint(
            constraint_matrix,
            constraint_lower,
            constraint_upper,
        ),
        options={
            "disp": False,
            "time_limit": 120.0,
            "mip_rel_gap": 0.0,
        },
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            "Stochastic MILP failed: "
            f"{result.message}"
        )

    allocation = np.rint(
        result.x[
            order_start:number_of_items
        ]
    ).astype("int32")

    allocated_spend = float(
        np.dot(
            allocation,
            unit_cost,
        )
    )

    if (
        allocated_spend
        > budget_limit + 1e-5
    ):
        raise ValueError(
            "Stochastic allocation violates budget."
        )

    if (
        allocation.sum()
        > capacity_limit + 1e-5
    ):
        raise ValueError(
            "Stochastic allocation violates capacity."
        )

    solver_information = {
        "solver_success": bool(
            result.success
        ),
        "solver_status": int(
            result.status
        ),
        "solver_message": str(
            result.message
        ),
        "expected_objective_cost": float(
            result.fun
        ),
        "allocated_units": int(
            allocation.sum()
        ),
        "allocated_spend": (
            allocated_spend
        ),
        "budget_limit": budget_limit,
        "capacity_limit": capacity_limit,
    }

    return (
        allocation,
        solver_information,
    )


def evaluate_actual_demand(
    store_data: pd.DataFrame,
    allocation: np.ndarray,
    scenario_name: str,
) -> pd.DataFrame:
    """Evaluate the optimised allocation on untouched actual demand."""

    result = (
        store_data.sort_values("item_id")
        .reset_index(drop=True)
        .copy()
    )

    result["scenario"] = (
        scenario_name
    )

    result["allocation_method"] = (
        STOCHASTIC_METHOD
    )

    result[
        "allocated_order_quantity"
    ] = allocation

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

    fill_rate_ratio = np.ones_like(
        actual,
        dtype=float,
    )

    np.divide(
        served,
        actual,
        out=fill_rate_ratio,
        where=actual > 0,
    )

    result["served_units"] = served
    result["shortage_units"] = shortage
    result["leftover_units"] = leftover

    result[
        "fill_rate_percent"
    ] = (
        100.0
        * fill_rate_ratio
    )

    result[
        "stockout_occurred"
    ] = (
        shortage > 0
    ).astype("int8")

    result["holding_cost_units"] = (
        leftover
        * HOLDING_COST_PER_UNIT
    )

    result["shortage_cost_units"] = (
        shortage
        * SHORTAGE_COST_PER_UNIT
    )

    result[
        "operational_cost_units"
    ] = (
        result[
            "holding_cost_units"
        ]
        + result[
            "shortage_cost_units"
        ]
    )

    result["purchase_spend"] = (
        result[
            "allocated_order_quantity"
        ]
        * result[
            "unit_purchase_cost"
        ]
    )

    target = result[
        "target_order_quantity"
    ].to_numpy(dtype=float)

    target_coverage = np.ones_like(
        target,
        dtype=float,
    )

    np.divide(
        allocated,
        target,
        out=target_coverage,
        where=target > 0,
    )

    result[
        "target_coverage_percent"
    ] = (
        100.0
        * target_coverage
    )

    expected_cost = (
        shortage
        * SHORTAGE_COST_PER_UNIT
        + leftover
        * HOLDING_COST_PER_UNIT
    )

    if not np.allclose(
        result[
            "operational_cost_units"
        ].to_numpy(dtype=float),
        expected_cost,
    ):
        raise ValueError(
            "Actual operating-cost identity failed."
        )

    return result


def add_summary_metrics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived performance measures."""

    summary = summary.copy()

    summary["fill_rate_percent"] = np.where(
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
        summary[
            "operational_cost_units"
        ]
        / summary[
            "number_of_series"
        ]
    )

    numeric_columns = (
        summary.select_dtypes(
            include=[np.number]
        ).columns
    )

    summary[
        numeric_columns
    ] = summary[
        numeric_columns
    ].round(2)

    return summary


def summarize_results(
    results: pd.DataFrame,
    grouping_columns: list[str],
    limit_aggregation: str,
) -> pd.DataFrame:
    """Summarise allocation performance."""

    if limit_aggregation not in {"first", "sum"}:
        raise ValueError(
            "limit_aggregation must be 'first' or 'sum'."
        )

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
                 limit_aggregation,
            ),
            capacity_limit=(
                "capacity_limit",
                limit_aggregation,
            ),
        )
        .reset_index()
    )

    return add_summary_metrics(
        summary
    )

def summarize_overall_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise results while counting each store limit once."""

    grouping_columns = [
        "scenario",
        "allocation_method",
    ]

    performance_summary = (
        results.groupby(
            grouping_columns,
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
        )
        .reset_index()
    )

    limit_columns = [
        "scenario",
        "allocation_method",
        "store_id",
        "budget_limit",
        "capacity_limit",
    ]

    limit_rows = results[
        limit_columns
    ].drop_duplicates()

    key_columns = [
        "scenario",
        "allocation_method",
        "store_id",
    ]

    inconsistent_limits = (
        limit_rows.groupby(
            key_columns
        )[
            [
                "budget_limit",
                "capacity_limit",
            ]
        ]
        .nunique()
        .gt(1)
        .any(axis=1)
    )

    if inconsistent_limits.any():
        raise ValueError(
            "Multiple budget or capacity limits were found "
            "for the same scenario, method, and store."
        )

    limit_rows = (
        limit_rows.drop_duplicates(
            subset=key_columns
        )
    )

    overall_limits = (
        limit_rows.groupby(
            grouping_columns,
            sort=True,
        )
        .agg(
            budget_limit=(
                "budget_limit",
                "sum",
            ),
            capacity_limit=(
                "capacity_limit",
                "sum",
            ),
        )
        .reset_index()
    )

    summary = performance_summary.merge(
        overall_limits,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    if summary[
        [
            "budget_limit",
            "capacity_limit",
        ]
    ].isna().any().any():
        raise ValueError(
            "Overall budget or capacity limits are missing."
        )

    return add_summary_metrics(
        summary
    )

def create_assumptions_table() -> pd.DataFrame:
    """Create tracked stochastic-model assumptions."""

    rows = [
        {
            "assumption": (
                "stochastic_objective"
            ),
            "value": (
                "Expected shortage cost "
                "plus expected holding cost"
            ),
        },
        {
            "assumption": (
                "shortage_cost_per_unit"
            ),
            "value": (
                SHORTAGE_COST_PER_UNIT
            ),
        },
        {
            "assumption": (
                "holding_cost_per_unit"
            ),
            "value": (
                HOLDING_COST_PER_UNIT
            ),
        },
        {
            "assumption": (
                "minimum_target_coverage"
            ),
            "value": (
                MINIMUM_TARGET_COVERAGE
            ),
        },
        {
            "assumption": (
                "actual_holdout_usage"
            ),
            "value": (
                "Evaluation only; excluded "
                "from optimisation"
            ),
        },
    ]

    for definition in (
        SCENARIO_DEFINITIONS
    ):
        rows.append(
            {
                "assumption": (
                    "demand_scenario_"
                    + definition[
                        "scenario_id"
                    ]
                ),
                "value": (
                    f"z={definition['z_score']}, "
                    f"probability="
                    f"{definition['probability']}"
                ),
            }
        )

    return pd.DataFrame(rows)


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save one tracked output table."""

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
        f"[SAVED] {filename:<42} "
        f"rows={len(dataframe):>5,}"
    )


def main() -> None:
    """Solve and evaluate all stochastic scenarios."""

    print("=" * 84)
    print(
        "STOCHASTIC EXPECTED-COST INVENTORY OPTIMISATION"
    )
    print("=" * 84)

    benchmark_results = (
        load_benchmark_results()
    )

    reference_rows = (
        benchmark_results.loc[
            benchmark_results[
                "allocation_method"
            ].eq(REFERENCE_METHOD)
        ]
        .copy()
    )

    constraint_scenarios = sorted(
        reference_rows[
            "scenario"
        ].unique()
    )

    stochastic_results = []
    demand_scenario_frames = []
    solver_rows = []

    for constraint_scenario in (
        constraint_scenarios
    ):
        print(
            f"\nConstraint scenario: "
            f"{constraint_scenario}"
        )
        print("-" * 84)

        for store_id in sorted(
            EXPECTED_STORES
        ):
            store_data = (
                reference_rows.loc[
                    reference_rows[
                        "scenario"
                    ].eq(
                        constraint_scenario
                    )
                    & reference_rows[
                        "store_id"
                    ].eq(store_id)
                ]
                .copy()
                .sort_values("item_id")
                .reset_index(drop=True)
            )

            if len(store_data) != 50:
                raise ValueError(
                    f"Expected 50 products for "
                    f"{constraint_scenario}-{store_id}, "
                    f"but found {len(store_data)}."
                )

            demand_scenarios = (
                create_demand_scenarios(
                    store_data,
                    constraint_scenario,
                )
            )

            allocation, status = (
                solve_stochastic_milp(
                    store_data,
                    demand_scenarios,
                )
            )

            evaluated = (
                evaluate_actual_demand(
                    store_data,
                    allocation,
                    constraint_scenario,
                )
            )

            stochastic_results.append(
                evaluated
            )

            demand_scenario_frames.append(
                demand_scenarios
            )

            solver_rows.append(
                {
                    "constraint_scenario": (
                        constraint_scenario
                    ),
                    "store_id": store_id,
                    **status,
                }
            )

            print(
                f"[SOLVED] {store_id} "
                f"allocated="
                f"{allocation.sum():,} "
                f"objective="
                f"{status['expected_objective_cost']:.2f}"
            )

    stochastic_results = pd.concat(
        stochastic_results,
        ignore_index=True,
    )

    demand_scenario_table = (
        pd.concat(
            demand_scenario_frames,
            ignore_index=True,
        )
    )

    # Compare the new model with both earlier methods.
    combined_results = pd.concat(
        [
            benchmark_results,
            stochastic_results,
        ],
        ignore_index=True,
        sort=False,
    )

    store_summary = summarize_results(
        combined_results,
        [
            "scenario",
            "allocation_method",
            "store_id",
        ],
        limit_aggregation="first",
    )

    overall_summary = summarize_overall_results(
        combined_results
    )
    

    solver_status = pd.DataFrame(
        solver_rows
    )

    assumptions = (
        create_assumptions_table()
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

    print("\nActual holdout comparison")
    print("-" * 84)

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
    print("-" * 84)

    save_table(
        combined_results,
        "stochastic_allocation_by_series.csv",
    )

    save_table(
        overall_summary,
        "stochastic_allocation_summary.csv",
    )

    save_table(
        store_summary,
        "stochastic_allocation_by_store.csv",
    )

    save_table(
        demand_scenario_table,
        "stochastic_demand_scenarios.csv",
    )

    save_table(
        solver_status,
        "stochastic_solver_status.csv",
    )

    save_table(
        assumptions,
        "stochastic_optimization_assumptions.csv",
    )

    print(
        "\nStochastic optimisation completed."
    )


if __name__ == "__main__":
    main()