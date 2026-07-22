"""Train and evaluate a pooled gradient-boosting demand model.

The model is trained on all 150 product-store series simultaneously.
The final 28 days are forecast recursively so that future actual sales
never enter lag or rolling features.
"""

from pathlib import Path
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_training_features.csv"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_inventory.db"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hist_gradient_boosting_model.joblib"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

ADAPTIVE_OVERALL_FILE = (
    REPORT_DIRECTORY
    / "adaptive_metrics_overall.csv"
)

ADAPTIVE_STORE_FILE = (
    REPORT_DIRECTORY
    / "adaptive_metrics_by_store.csv"
)

FORECAST_HORIZON = 28
EXPECTED_SERIES = 150

CATEGORICAL_COLUMNS = [
    "store_id",
    "item_id",
]

NUMERIC_COLUMNS = [
    "sell_price",
    "price_available",
    "event_flag",
    "snap_active",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_zero_rate_7",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_zero_rate_28",
    "demand_trend_7_vs_28",
]

FEATURE_COLUMNS = (
    CATEGORICAL_COLUMNS
    + NUMERIC_COLUMNS
)


def load_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    dict[str, list[str]],
    int,
]:
    """Load features and prepare categorical columns."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Training feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    training = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    required_columns = (
        set(FEATURE_COLUMNS)
        | {"units_sold"}
    )

    missing_columns = (
        required_columns
        - set(training.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing training columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    category_levels: dict[
        str,
        list[str],
    ] = {}

    for column in CATEGORICAL_COLUMNS:
        levels = sorted(
            training[column]
            .astype(str)
            .unique()
            .tolist()
        )

        category_levels[column] = levels

        training[column] = pd.Categorical(
            training[column].astype(str),
            categories=levels,
        )

    feature_data = training[
        FEATURE_COLUMNS
    ].copy()

    target = training[
        "units_sold"
    ].astype("float64")

    if feature_data.isna().any().any():
        raise ValueError(
            "Missing values were found in training features."
        )

    if target.isna().any():
        raise ValueError(
            "Missing target values were found."
        )

    if (target < 0).any():
        raise ValueError(
            "Negative target values were found."
        )

    number_of_series = (
        training[
            ["store_id", "item_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if number_of_series != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} series, "
            f"but found {number_of_series}."
        )

    return (
        feature_data,
        target,
        category_levels,
        len(training),
    )


def load_full_panel() -> pd.DataFrame:
    """Load the full panel for recursive holdout forecasting."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"{DATABASE_FILE}"
        )

    query = """
        SELECT
            s.store_id,
            st.state_id,
            s.item_id,
            s.day_id,
            c.date,
            c.wday,
            c.month,
            c.event_name_1,
            c.event_name_2,
            c.snap_CA,
            c.snap_TX,
            c.snap_WI,
            p.sell_price,
            s.units_sold
        FROM fact_sales AS s
        INNER JOIN dim_stores AS st
            ON s.store_id = st.store_id
        INNER JOIN dim_calendar AS c
            ON s.day_id = c.day_id
        LEFT JOIN fact_prices AS p
            ON s.store_id = p.store_id
            AND s.item_id = p.item_id
            AND c.wm_yr_wk = p.wm_yr_wk
        ORDER BY
            s.store_id,
            s.item_id,
            c.date;
    """

    with sqlite3.connect(
        DATABASE_FILE
    ) as connection:
        panel = pd.read_sql_query(
            query,
            connection,
            parse_dates=["date"],
        )

    panel["event_flag"] = (
        panel["event_name_1"].notna()
        | panel["event_name_2"].notna()
    ).astype("int8")

    panel["snap_active"] = np.select(
        [
            panel["state_id"].eq("CA"),
            panel["state_id"].eq("TX"),
            panel["state_id"].eq("WI"),
        ],
        [
            panel["snap_CA"],
            panel["snap_TX"],
            panel["snap_WI"],
        ],
        default=0,
    ).astype("int8")

    panel["price_available"] = (
        panel["sell_price"].notna()
    ).astype("int8")

    panel["sell_price"] = (
        panel["sell_price"]
        .fillna(0.0)
        .astype("float32")
    )

    panel["weekday_sin"] = np.sin(
        2 * np.pi * panel["wday"] / 7
    )

    panel["weekday_cos"] = np.cos(
        2 * np.pi * panel["wday"] / 7
    )

    panel["month_sin"] = np.sin(
        2 * np.pi * panel["month"] / 12
    )

    panel["month_cos"] = np.cos(
        2 * np.pi * panel["month"] / 12
    )

    return panel


def fit_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> HistGradientBoostingRegressor:
    """Fit the pooled gradient-boosting model."""

    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=42,
        verbose=1,
    )

    model.fit(
        features,
        target,
    )

    return model


def build_feature_record(
    row: object,
    history: list[float],
) -> dict[str, object]:
    """Create one recursive prediction row."""

    if len(history) < 28:
        raise ValueError(
            "Insufficient history for recursive forecasting."
        )

    last_7 = np.asarray(
        history[-7:],
        dtype=float,
    )

    last_28 = np.asarray(
        history[-28:],
        dtype=float,
    )

    rolling_mean_7 = float(
        np.mean(last_7)
    )

    rolling_mean_28 = float(
        np.mean(last_28)
    )

    return {
        "store_id": row.store_id,
        "item_id": row.item_id,
        "sell_price": row.sell_price,
        "price_available": row.price_available,
        "event_flag": row.event_flag,
        "snap_active": row.snap_active,
        "weekday_sin": row.weekday_sin,
        "weekday_cos": row.weekday_cos,
        "month_sin": row.month_sin,
        "month_cos": row.month_cos,
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": float(
            np.std(
                last_7,
                ddof=1,
            )
        ),
        "rolling_zero_rate_7": float(
            np.mean(last_7 == 0)
        ),
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_28": float(
            np.std(
                last_28,
                ddof=1,
            )
        ),
        "rolling_zero_rate_28": float(
            np.mean(last_28 == 0)
        ),
        "demand_trend_7_vs_28": (
            rolling_mean_7
            - rolling_mean_28
        ),
    }


def cast_categories(
    features: pd.DataFrame,
    category_levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply the same categorical levels used during training."""

    features = features.copy()

    for column in CATEGORICAL_COLUMNS:
        features[column] = pd.Categorical(
            features[column].astype(str),
            categories=category_levels[column],
        )

        if features[column].isna().any():
            raise ValueError(
                f"Unknown category found in {column}."
            )

    return features


def create_recursive_forecasts(
    model: HistGradientBoostingRegressor,
    panel: pd.DataFrame,
    category_levels: dict[str, list[str]],
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
    pd.Timestamp,
]:
    """Forecast the final 28 days recursively."""

    dates = (
        panel["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    holdout_dates = dates.tail(
        FORECAST_HORIZON
    )

    holdout_start = holdout_dates.iloc[0]
    holdout_end = holdout_dates.iloc[-1]

    historical_panel = panel.loc[
        panel["date"] < holdout_start
    ]

    history: dict[
        tuple[str, str],
        list[float],
    ] = {}

    for (
        store_id,
        item_id,
    ), series in historical_panel.groupby(
        ["store_id", "item_id"],
        sort=True,
    ):
        history[
            (store_id, item_id)
        ] = (
            series.sort_values("date")[
                "units_sold"
            ]
            .astype(float)
            .tolist()
        )

    if len(history) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected history for {EXPECTED_SERIES} series, "
            f"but found {len(history)}."
        )

    forecast_frames: list[
        pd.DataFrame
    ] = []

    for forecast_step, forecast_date in enumerate(
        holdout_dates,
        start=1,
    ):
        day_rows = (
            panel.loc[
                panel["date"].eq(
                    forecast_date
                )
            ]
            .sort_values(
                ["store_id", "item_id"]
            )
            .reset_index(drop=True)
        )

        if len(day_rows) != EXPECTED_SERIES:
            raise ValueError(
                f"Expected {EXPECTED_SERIES} rows for "
                f"{forecast_date.date()}, "
                f"but found {len(day_rows)}."
            )

        feature_records = []
        series_keys = []

        for row in day_rows.itertuples(
            index=False
        ):
            series_key = (
                row.store_id,
                row.item_id,
            )

            feature_records.append(
                build_feature_record(
                    row,
                    history[series_key],
                )
            )

            series_keys.append(
                series_key
            )

        prediction_features = pd.DataFrame(
            feature_records
        )[FEATURE_COLUMNS]

        prediction_features = cast_categories(
            prediction_features,
            category_levels,
        )

        predicted_units = model.predict(
            prediction_features
        )

        predicted_units = np.clip(
            predicted_units,
            a_min=0.0,
            a_max=None,
        )

        for series_key, prediction in zip(
            series_keys,
            predicted_units,
            strict=True,
        ):
            history[series_key].append(
                float(prediction)
            )

        forecast_frame = pd.DataFrame(
            {
                "model": (
                    "hist_gradient_boosting"
                ),
                "forecast_step": forecast_step,
                "store_id": day_rows[
                    "store_id"
                ],
                "item_id": day_rows[
                    "item_id"
                ],
                "day_id": day_rows[
                    "day_id"
                ],
                "date": day_rows[
                    "date"
                ],
                "actual_units": day_rows[
                    "units_sold"
                ].astype(float),
                "predicted_units": (
                    predicted_units
                ),
            }
        )

        forecast_frames.append(
            forecast_frame
        )

        print(
            f"\rForecasted step "
            f"{forecast_step:>2}/{FORECAST_HORIZON}",
            end="",
            flush=True,
        )

    print()

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_SERIES
        * FORECAST_HORIZON
    )

    if len(forecasts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} forecasts, "
            f"but created {len(forecasts):,}."
        )

    if forecasts[
        "predicted_units"
    ].isna().any():
        raise ValueError(
            "Missing predictions were generated."
        )

    return (
        forecasts,
        holdout_start,
        holdout_end,
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate consistent evaluation metrics."""

    actual = actual.astype(float)
    predicted = predicted.astype(float)

    error = predicted - actual

    absolute_error = np.abs(error)

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
        "mae": float(
            np.mean(absolute_error)
        ),
        "rmse": float(
            np.sqrt(
                np.mean(error**2)
            )
        ),
        "bias": float(
            np.mean(error)
        ),
        "wape_percent": float(
            wape
        ),
    }


def calculate_group_metrics(
    forecasts: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Calculate metrics by requested groups."""

    result_rows = []

    for group_key, group in forecasts.groupby(
        group_columns,
        sort=True,
    ):
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        row = dict(
            zip(
                group_columns,
                group_key,
                strict=True,
            )
        )

        row.update(
            calculate_metrics(
                group[
                    "actual_units"
                ].to_numpy(),
                group[
                    "predicted_units"
                ].to_numpy(),
            )
        )

        result_rows.append(row)

    result = pd.DataFrame(
        result_rows
    )

    metric_columns = [
        "mae",
        "rmse",
        "bias",
        "wape_percent",
    ]

    result[metric_columns] = (
        result[metric_columns]
        .round(4)
    )

    return result


def create_comparison_tables(
    forecasts: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compare ML results with all previous baselines."""

    if not ADAPTIVE_OVERALL_FILE.exists():
        raise FileNotFoundError(
            f"Missing comparison file:\n"
            f"{ADAPTIVE_OVERALL_FILE}"
        )

    if not ADAPTIVE_STORE_FILE.exists():
        raise FileNotFoundError(
            f"Missing comparison file:\n"
            f"{ADAPTIVE_STORE_FILE}"
        )

    ml_overall = calculate_group_metrics(
        forecasts,
        ["model"],
    )

    ml_by_store = calculate_group_metrics(
        forecasts,
        ["model", "store_id"],
    )

    previous_overall = pd.read_csv(
        ADAPTIVE_OVERALL_FILE
    )

    previous_by_store = pd.read_csv(
        ADAPTIVE_STORE_FILE
    )

    overall_comparison = (
        pd.concat(
            [
                previous_overall,
                ml_overall,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["model"],
            keep="last",
        )
        .sort_values("mae")
        .reset_index(drop=True)
    )

    store_comparison = (
        pd.concat(
            [
                previous_by_store,
                ml_by_store,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "model",
                "store_id",
            ],
            keep="last",
        )
        .sort_values(
            ["store_id", "mae"]
        )
        .reset_index(drop=True)
    )

    return (
        overall_comparison,
        store_comparison,
    )


def create_model_summary(
    model: HistGradientBoostingRegressor,
    training_rows: int,
    training_seconds: float,
    forecast_seconds: float,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    overall_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create a tracked model configuration summary."""

    ml_metrics = overall_comparison.loc[
        overall_comparison["model"].eq(
            "hist_gradient_boosting"
        )
    ].iloc[0]

    adaptive_metrics = overall_comparison.loc[
        overall_comparison["model"].eq(
            "adaptive_baseline"
        )
    ].iloc[0]

    mae_improvement = (
        (
            adaptive_metrics["mae"]
            - ml_metrics["mae"]
        )
        / adaptive_metrics["mae"]
        * 100
    )

    return pd.DataFrame(
        {
            "metric": [
                "model",
                "loss",
                "training_rows",
                "feature_columns",
                "categorical_features",
                "maximum_iterations",
                "completed_iterations",
                "training_seconds",
                "recursive_forecast_seconds",
                "holdout_start",
                "holdout_end",
                "holdout_days",
                "mae",
                "rmse",
                "bias",
                "wape_percent",
                "mae_improvement_vs_adaptive_percent",
            ],
            "value": [
                "HistGradientBoostingRegressor",
                "poisson",
                training_rows,
                len(FEATURE_COLUMNS),
                len(CATEGORICAL_COLUMNS),
                model.max_iter,
                model.n_iter_,
                round(training_seconds, 2),
                round(forecast_seconds, 2),
                holdout_start.date(),
                holdout_end.date(),
                FORECAST_HORIZON,
                ml_metrics["mae"],
                ml_metrics["rmse"],
                ml_metrics["bias"],
                ml_metrics["wape_percent"],
                round(mae_improvement, 4),
            ],
        }
    )


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
        date_format="%Y-%m-%d",
    )

    print(
        f"[SAVED] {filename:<34} "
        f"rows={len(dataframe):>5,}"
    )


def main() -> None:
    """Train, recursively forecast, and evaluate the ML model."""

    print("=" * 78)
    print("POOLED HISTOGRAM GRADIENT-BOOSTING FORECAST")
    print("=" * 78)

    (
        training_features,
        target,
        category_levels,
        training_rows,
    ) = load_training_data()

    print(
        f"Training rows:       "
        f"{training_rows:,}"
    )
    print(
        f"Feature columns:     "
        f"{len(FEATURE_COLUMNS)}"
    )
    print(
        f"Categorical columns: "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    print("\nTraining model...")
    print("-" * 78)

    training_start_time = (
        time.perf_counter()
    )

    model = fit_model(
        training_features,
        target,
    )

    training_seconds = (
        time.perf_counter()
        - training_start_time
    )

    print(
        f"Training completed in "
        f"{training_seconds:.2f} seconds."
    )

    panel = load_full_panel()

    print("\nGenerating recursive forecasts...")
    print("-" * 78)

    forecast_start_time = (
        time.perf_counter()
    )

    (
        forecasts,
        holdout_start,
        holdout_end,
    ) = create_recursive_forecasts(
        model,
        panel,
        category_levels,
    )

    forecast_seconds = (
        time.perf_counter()
        - forecast_start_time
    )

    (
        overall_comparison,
        store_comparison,
    ) = create_comparison_tables(
        forecasts
    )

    model_summary = create_model_summary(
        model,
        training_rows,
        training_seconds,
        forecast_seconds,
        holdout_start,
        holdout_end,
        overall_comparison,
    )

    print("\nOverall model comparison")
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

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "category_levels": category_levels,
        },
        MODEL_FILE,
    )

    print("\nSaving results...")
    print("-" * 78)

    save_table(
        forecasts,
        "ml_forecasts.csv",
    )

    save_table(
        overall_comparison,
        "ml_metrics_overall.csv",
    )

    save_table(
        store_comparison,
        "ml_metrics_by_store.csv",
    )

    save_table(
        model_summary,
        "ml_model_summary.csv",
    )

    print("\n" + "-" * 78)
    print("Machine-learning evaluation completed.")
    print(f"Saved local model: {MODEL_FILE}")


if __name__ == "__main__":
    main()