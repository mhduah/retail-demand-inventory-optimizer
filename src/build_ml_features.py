"""Build leakage-safe features for pooled demand forecasting.

The final 28 historical days remain untouched for final evaluation.
Lag and rolling features use only information available before each
target observation.
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

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_training_features.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "ml_feature_summary.csv"
)

HOLDOUT_DAYS = 28
EXPECTED_DAYS = 1_941
EXPECTED_SERIES = 150

LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)


def load_panel() -> pd.DataFrame:
    """Load sales, calendar, price, and store information."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DATABASE_FILE}"
        )

    query = """
        SELECT
            s.store_id,
            st.state_id,
            s.item_id,
            s.day_id,
            c.date,
            c.wm_yr_wk,
            c.wday,
            c.month,
            c.year,
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

    with sqlite3.connect(DATABASE_FILE) as connection:
        panel = pd.read_sql_query(
            query,
            connection,
            parse_dates=["date"],
        )

    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    """Validate the source panel before feature creation."""

    required_columns = {
        "store_id",
        "state_id",
        "item_id",
        "day_id",
        "date",
        "wday",
        "month",
        "year",
        "sell_price",
        "units_sold",
    }

    missing_columns = required_columns - set(panel.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    series_sizes = (
        panel.groupby(["store_id", "item_id"])
        .size()
    )

    if len(series_sizes) != EXPECTED_SERIES:
        raise ValueError(
            f"Expected {EXPECTED_SERIES} series, "
            f"but found {len(series_sizes)}."
        )

    if not series_sizes.eq(EXPECTED_DAYS).all():
        raise ValueError(
            f"Every series must contain {EXPECTED_DAYS} observations."
        )

    if panel["units_sold"].isna().any():
        raise ValueError("Missing sales values were found.")

    if (panel["units_sold"] < 0).any():
        raise ValueError("Negative sales values were found.")


def add_calendar_and_price_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Create known-in-advance calendar and price features."""

    panel = panel.copy()

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


def add_historical_demand_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Create lag and rolling features without target leakage."""

    panel = panel.copy()

    grouped = panel.groupby(
        ["store_id", "item_id"],
        sort=False,
    )

    for lag in LAG_DAYS:
        panel[f"lag_{lag}"] = (
            grouped["units_sold"]
            .shift(lag)
            .astype("float32")
        )

    for window in ROLLING_WINDOWS:
        panel[f"rolling_mean_{window}"] = (
            grouped["units_sold"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
            .astype("float32")
        )

        panel[f"rolling_std_{window}"] = (
            grouped["units_sold"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
                )
            )
            .astype("float32")
        )

        panel[f"rolling_zero_rate_{window}"] = (
            grouped["units_sold"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .eq(0)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
            .astype("float32")
        )

    panel["demand_trend_7_vs_28"] = (
        panel["rolling_mean_7"]
        - panel["rolling_mean_28"]
    ).astype("float32")

    return panel


def create_training_table(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Remove the final holdout period and incomplete warm-up rows."""

    unique_dates = (
        panel["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) != EXPECTED_DAYS:
        raise ValueError(
            f"Expected {EXPECTED_DAYS} unique dates, "
            f"but found {len(unique_dates)}."
        )

    holdout_start = unique_dates.iloc[-HOLDOUT_DAYS]

    training = panel.loc[
        panel["date"] < holdout_start
    ].copy()

    feature_columns = [
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

    training = (
        training.dropna(
            subset=feature_columns
        )
        .reset_index(drop=True)
    )

    expected_training_days = (
        EXPECTED_DAYS
        - HOLDOUT_DAYS
        - max(LAG_DAYS)
    )

    expected_rows = (
        EXPECTED_SERIES
        * expected_training_days
    )

    if len(training) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} training rows, "
            f"but created {len(training):,}."
        )

    if training[feature_columns].isna().any().any():
        raise ValueError(
            "Missing values remain in the feature columns."
        )

    if training["date"].max() >= holdout_start:
        raise ValueError(
            "Holdout-period observations entered the training table."
        )

    return training, holdout_start


def create_summary(
    training: pd.DataFrame,
    holdout_start: pd.Timestamp,
) -> pd.DataFrame:
    """Create a small tracked summary of the feature dataset."""

    feature_columns = [
        column
        for column in training.columns
        if column.startswith(
            (
                "lag_",
                "rolling_",
                "weekday_",
                "month_",
                "demand_trend",
            )
        )
        or column
        in {
            "sell_price",
            "price_available",
            "event_flag",
            "snap_active",
        }
    ]

    summary = pd.DataFrame(
        {
            "metric": [
                "training_rows",
                "product_store_series",
                "training_start_date",
                "training_end_date",
                "holdout_start_date",
                "feature_columns",
                "missing_feature_values",
            ],
            "value": [
                len(training),
                (
                    training[
                        ["store_id", "item_id"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                training["date"].min().date(),
                training["date"].max().date(),
                holdout_start.date(),
                len(feature_columns),
                int(
                    training[
                        feature_columns
                    ]
                    .isna()
                    .sum()
                    .sum()
                ),
            ],
        }
    )

    return summary


def main() -> None:
    """Build and save the machine-learning feature table."""

    print("=" * 78)
    print("BUILD MACHINE-LEARNING FEATURES")
    print("=" * 78)

    panel = load_panel()
    validate_panel(panel)

    panel = add_calendar_and_price_features(
        panel
    )

    panel = add_historical_demand_features(
        panel
    )

    training, holdout_start = create_training_table(
        panel
    )

    summary = create_summary(
        training,
        holdout_start,
    )

    FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training.to_csv(
        FEATURE_FILE,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    print(summary.to_string(index=False))

    print("\n" + "-" * 78)
    print("Feature table created successfully.")
    print(f"Training data: {FEATURE_FILE}")
    print(f"Summary:       {SUMMARY_FILE}")


if __name__ == "__main__":
    main()