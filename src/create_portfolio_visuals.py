"""Create portfolio-ready figures and key project findings.

The figures summarise:
- forecasting-model performance;
- unconstrained inventory-policy performance;
- constrained allocation costs;
- service-cost trade-offs.

All figures are generated from tracked report tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FORECAST_METRICS_FILE = (
    TABLE_DIRECTORY
    / "ml_metrics_overall.csv"
)

INVENTORY_POLICY_FILE = (
    TABLE_DIRECTORY
    / "inventory_policy_summary.csv"
)

STOCHASTIC_SUMMARY_FILE = (
    TABLE_DIRECTORY
    / "stochastic_allocation_summary.csv"
)

KEY_FINDINGS_FILE = (
    TABLE_DIRECTORY
    / "portfolio_key_findings.csv"
)


MODEL_LABELS = {
    "last_value_naive": "Last value",
    "weekly_seasonal_naive": "Weekly seasonal",
    "adaptive_baseline": "Adaptive baseline",
    "hist_gradient_boosting": "Gradient boosting",
}

POLICY_LABELS = {
    (
        "adaptive_baseline",
        "point_forecast_order",
    ): "Adaptive + point",
    (
        "adaptive_baseline",
        "newsvendor_adjusted_order",
    ): "Adaptive + newsvendor",
    (
        "hist_gradient_boosting",
        "point_forecast_order",
    ): "ML + point",
    (
        "hist_gradient_boosting",
        "newsvendor_adjusted_order",
    ): "ML + newsvendor",
}

ALLOCATION_LABELS = {
    "milp_optimised": "Target-shortfall MILP",
    "proportional_heuristic": "Proportional heuristic",
    "stochastic_expected_cost_milp": "Stochastic cost MILP",
}

SCENARIO_ORDER = [
    "tight",
    "balanced",
    "near_full",
]

METHOD_ORDER = [
    "proportional_heuristic",
    "milp_optimised",
    "stochastic_expected_cost_milp",
]


def load_table(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    """Load and validate one report table."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required report table not found:\n{path}"
        )

    dataframe = pd.read_csv(path)

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return dataframe


def save_figure(
    filename: str,
) -> None:
    """Save and close the current figure."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIRECTORY
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[SAVED] {filename}"
    )


def create_forecast_mae_chart(
    forecast_metrics: pd.DataFrame,
) -> None:
    """Plot overall MAE for all forecasting models."""

    chart_data = (
        forecast_metrics[
            ["model", "mae"]
        ]
        .drop_duplicates(
            subset=["model"]
        )
        .sort_values("mae")
        .reset_index(drop=True)
    )

    labels = (
        chart_data["model"]
        .map(MODEL_LABELS)
        .fillna(
            chart_data["model"]
        )
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        labels,
        chart_data["mae"],
    )

    plt.ylabel(
        "Mean absolute error"
    )

    plt.title(
        "28-Day Demand Forecasting Performance"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    maximum_value = float(
        chart_data["mae"].max()
    )

    plt.ylim(
        0,
        maximum_value * 1.20,
    )

    for bar, value in zip(
        bars,
        chart_data["mae"],
        strict=True,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + maximum_value * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure(
        "forecast_mae_comparison.png"
    )


def create_forecast_rmse_chart(
    forecast_metrics: pd.DataFrame,
) -> None:
    """Plot overall RMSE for all forecasting models."""

    chart_data = (
        forecast_metrics[
            ["model", "rmse"]
        ]
        .drop_duplicates(
            subset=["model"]
        )
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    labels = (
        chart_data["model"]
        .map(MODEL_LABELS)
        .fillna(
            chart_data["model"]
        )
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        labels,
        chart_data["rmse"],
    )

    plt.ylabel(
        "Root mean squared error"
    )

    plt.title(
        "Large-Error Performance of Forecasting Models"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    maximum_value = float(
        chart_data["rmse"].max()
    )

    plt.ylim(
        0,
        maximum_value * 1.20,
    )

    for bar, value in zip(
        bars,
        chart_data["rmse"],
        strict=True,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + maximum_value * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure(
        "forecast_rmse_comparison.png"
    )


def create_inventory_policy_chart(
    policy_results: pd.DataFrame,
) -> None:
    """Plot unconstrained inventory-policy operating costs."""

    chart_data = (
        policy_results.copy()
    )

    chart_data["policy_label"] = [
        POLICY_LABELS.get(
            (
                forecast_model,
                inventory_policy,
            ),
            (
                f"{forecast_model} + "
                f"{inventory_policy}"
            ),
        )
        for (
            forecast_model,
            inventory_policy,
        ) in zip(
            chart_data[
                "forecast_model"
            ],
            chart_data[
                "inventory_policy"
            ],
            strict=True,
        )
    ]

    chart_data = (
        chart_data.sort_values(
            "total_cost_units"
        )
        .reset_index(drop=True)
    )

    plt.figure(
        figsize=(10, 5.5)
    )

    bars = plt.bar(
        chart_data["policy_label"],
        chart_data["total_cost_units"],
    )

    plt.ylabel(
        "Operating cost units"
    )

    plt.title(
        "Forecast-Driven Inventory Policy Comparison"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    maximum_value = float(
        chart_data[
            "total_cost_units"
        ].max()
    )

    plt.ylim(
        0,
        maximum_value * 1.18,
    )

    for bar, value in zip(
        bars,
        chart_data[
            "total_cost_units"
        ],
        strict=True,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + maximum_value * 0.025,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )

    save_figure(
        "inventory_policy_cost_comparison.png"
    )


def create_constrained_cost_chart(
    stochastic_summary: pd.DataFrame,
) -> None:
    """Plot operating cost by constraint scenario and method."""

    pivot = (
        stochastic_summary.pivot(
            index="scenario",
            columns="allocation_method",
            values="operational_cost_units",
        )
        .reindex(SCENARIO_ORDER)
    )

    available_methods = [
        method
        for method in METHOD_ORDER
        if method in pivot.columns
    ]

    x_positions = np.arange(
        len(pivot.index)
    )

    group_width = 0.80

    bar_width = (
        group_width
        / len(available_methods)
    )

    plt.figure(
        figsize=(11, 6)
    )

    for method_index, method in enumerate(
        available_methods
    ):
        positions = (
            x_positions
            - group_width / 2
            + bar_width / 2
            + method_index * bar_width
        )

        bars = plt.bar(
            positions,
            pivot[method],
            width=bar_width,
            label=ALLOCATION_LABELS.get(
                method,
                method,
            ),
        )

        for bar, value in zip(
            bars,
            pivot[method],
            strict=True,
        ):
            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(
        x_positions,
        [
            scenario.replace(
                "_",
                " ",
            ).title()
            for scenario in pivot.index
        ],
    )

    plt.ylabel(
        "Operating cost units"
    )

    plt.title(
        "Inventory Allocation Under Resource Constraints"
    )

    plt.legend()

    maximum_value = float(
        pivot[
            available_methods
        ]
        .to_numpy()
        .max()
    )

    plt.ylim(
        0,
        maximum_value * 1.18,
    )

    save_figure(
        "constrained_allocation_costs.png"
    )


def create_service_cost_chart(
    stochastic_summary: pd.DataFrame,
) -> None:
    """Plot the service-cost trade-off for all allocations."""

    plt.figure(
        figsize=(10, 6)
    )

    for method in METHOD_ORDER:
        method_data = (
            stochastic_summary.loc[
                stochastic_summary[
                    "allocation_method"
                ].eq(method)
            ]
            .copy()
        )

        if method_data.empty:
            continue

        plt.scatter(
            method_data[
                "operational_cost_units"
            ],
            method_data[
                "fill_rate_percent"
            ],
            s=80,
            label=ALLOCATION_LABELS.get(
                method,
                method,
            ),
        )

        for row in method_data.itertuples(
            index=False
        ):
            plt.annotate(
                row.scenario.replace(
                    "_",
                    " ",
                ).title(),
                (
                    row.operational_cost_units,
                    row.fill_rate_percent,
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel(
        "Operating cost units"
    )

    plt.ylabel(
        "Fill rate (%)"
    )

    plt.title(
        "Service-Level and Operating-Cost Trade-Off"
    )

    plt.legend()

    save_figure(
        "service_cost_tradeoff.png"
    )


def percentage_improvement(
    old_value: float,
    new_value: float,
) -> float:
    """Calculate percentage reduction from old to new."""

    if old_value == 0:
        return np.nan

    return (
        (old_value - new_value)
        / old_value
        * 100
    )


def extract_value(
    dataframe: pd.DataFrame,
    filters: dict[str, str],
    value_column: str,
) -> float:
    """Extract one uniquely identified numeric result."""

    mask = pd.Series(
        True,
        index=dataframe.index,
    )

    for column, value in filters.items():
        mask &= dataframe[column].eq(value)

    matches = dataframe.loc[
        mask,
        value_column,
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one match for {filters}, "
            f"but found {len(matches)}."
        )

    return float(
        matches.iloc[0]
    )


def create_key_findings(
    forecast_metrics: pd.DataFrame,
    policy_results: pd.DataFrame,
    stochastic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise table of portfolio headline results."""

    adaptive_mae = extract_value(
        forecast_metrics,
        {
            "model": (
                "adaptive_baseline"
            )
        },
        "mae",
    )

    ml_mae = extract_value(
        forecast_metrics,
        {
            "model": (
                "hist_gradient_boosting"
            )
        },
        "mae",
    )

    adaptive_rmse = extract_value(
        forecast_metrics,
        {
            "model": (
                "adaptive_baseline"
            )
        },
        "rmse",
    )

    ml_rmse = extract_value(
        forecast_metrics,
        {
            "model": (
                "hist_gradient_boosting"
            )
        },
        "rmse",
    )

    ml_point_cost = extract_value(
        policy_results,
        {
            "forecast_model": (
                "hist_gradient_boosting"
            ),
            "inventory_policy": (
                "point_forecast_order"
            ),
        },
        "total_cost_units",
    )

    ml_newsvendor_cost = extract_value(
        policy_results,
        {
            "forecast_model": (
                "hist_gradient_boosting"
            ),
            "inventory_policy": (
                "newsvendor_adjusted_order"
            ),
        },
        "total_cost_units",
    )

    tight_proportional_cost = extract_value(
        stochastic_summary,
        {
            "scenario": "tight",
            "allocation_method": (
                "proportional_heuristic"
            ),
        },
        "operational_cost_units",
    )

    tight_stochastic_cost = extract_value(
        stochastic_summary,
        {
            "scenario": "tight",
            "allocation_method": (
                "stochastic_expected_cost_milp"
            ),
        },
        "operational_cost_units",
    )

    balanced_proportional_cost = extract_value(
        stochastic_summary,
        {
            "scenario": "balanced",
            "allocation_method": (
                "proportional_heuristic"
            ),
        },
        "operational_cost_units",
    )

    balanced_stochastic_cost = extract_value(
        stochastic_summary,
        {
            "scenario": "balanced",
            "allocation_method": (
                "stochastic_expected_cost_milp"
            ),
        },
        "operational_cost_units",
    )

    return pd.DataFrame(
        {
            "finding": [
                (
                    "Gradient boosting RMSE reduction "
                    "versus adaptive baseline"
                ),
                (
                    "Gradient boosting MAE change "
                    "versus adaptive baseline"
                ),
                (
                    "ML newsvendor cost reduction "
                    "versus ML point ordering"
                ),
                (
                    "Stochastic MILP cost reduction "
                    "under tight constraints"
                ),
                (
                    "Stochastic MILP cost reduction "
                    "under balanced constraints"
                ),
            ],
            "value_percent": [
                percentage_improvement(
                    adaptive_rmse,
                    ml_rmse,
                ),
                percentage_improvement(
                    adaptive_mae,
                    ml_mae,
                ),
                percentage_improvement(
                    ml_point_cost,
                    ml_newsvendor_cost,
                ),
                percentage_improvement(
                    tight_proportional_cost,
                    tight_stochastic_cost,
                ),
                percentage_improvement(
                    balanced_proportional_cost,
                    balanced_stochastic_cost,
                ),
            ],
            "interpretation": [
                (
                    "The pooled model reduced large "
                    "forecasting errors."
                ),
                (
                    "Negative value means the adaptive "
                    "baseline retained slightly lower MAE."
                ),
                (
                    "Safety-stock optimisation improved "
                    "inventory operating cost."
                ),
                (
                    "Optimisation added the most value "
                    "when resources were scarce."
                ),
                (
                    "Optimisation provided a smaller but "
                    "positive gain under balanced limits."
                ),
            ],
        }
    ).round(
        {
            "value_percent": 2,
        }
    )


def main() -> None:
    """Create all portfolio figures and headline findings."""

    print("=" * 76)
    print("CREATE PORTFOLIO VISUALISATIONS")
    print("=" * 76)

    forecast_metrics = load_table(
        FORECAST_METRICS_FILE,
        {
            "model",
            "mae",
            "rmse",
            "bias",
            "wape_percent",
        },
    )

    policy_results = load_table(
        INVENTORY_POLICY_FILE,
        {
            "forecast_model",
            "inventory_policy",
            "fill_rate_percent",
            "total_cost_units",
        },
    )

    stochastic_summary = load_table(
        STOCHASTIC_SUMMARY_FILE,
        {
            "scenario",
            "allocation_method",
            "fill_rate_percent",
            "operational_cost_units",
        },
    )

    create_forecast_mae_chart(
        forecast_metrics
    )

    create_forecast_rmse_chart(
        forecast_metrics
    )

    create_inventory_policy_chart(
        policy_results
    )

    create_constrained_cost_chart(
        stochastic_summary
    )

    create_service_cost_chart(
        stochastic_summary
    )

    key_findings = create_key_findings(
        forecast_metrics,
        policy_results,
        stochastic_summary,
    )

    key_findings.to_csv(
        KEY_FINDINGS_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] {KEY_FINDINGS_FILE.name}"
    )

    print("\nHeadline findings")
    print("-" * 76)

    print(
        key_findings.to_string(
            index=False
        )
    )

    print("\nPortfolio visualisations completed.")


if __name__ == "__main__":
    main()