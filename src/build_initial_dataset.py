"""Build a manageable, reproducible subset of the M5 dataset.

The initial project scope contains:
- FOODS_3 department
- Three stores: CA_1, TX_1 and WI_1
- Fifty products available in all three stores
- All available historical sales days
- Relevant weekly selling prices

The resulting files are written to data/interim in a structure suitable
for later SQL loading and forecasting.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

SALES_FILE = RAW_DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"

TARGET_DEPARTMENT = "FOODS_3"
TARGET_STORES = ("CA_1", "TX_1", "WI_1")
NUMBER_OF_ITEMS = 50

SALES_ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def load_filtered_sales() -> tuple[pd.DataFrame, list[str]]:
    """Load only the target department and stores from the sales data."""

    print("Reading sales-file structure...")

    header = pd.read_csv(SALES_FILE, nrows=0)
    day_columns = [
        column
        for column in header.columns
        if column.startswith("d_")
    ]

    if not day_columns:
        raise ValueError("No daily sales columns were found.")

    use_columns = SALES_ID_COLUMNS + day_columns
    filtered_chunks: list[pd.DataFrame] = []

    print(
        f"Filtering department {TARGET_DEPARTMENT} "
        f"for stores {', '.join(TARGET_STORES)}..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            SALES_FILE,
            usecols=use_columns,
            chunksize=2_000,
            low_memory=False,
        ),
        start=1,
    ):
        mask = (
            chunk["dept_id"].eq(TARGET_DEPARTMENT)
            & chunk["store_id"].isin(TARGET_STORES)
        )

        filtered_chunk = chunk.loc[mask]

        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)

        print(
            f"\rProcessed sales chunk {chunk_number}",
            end="",
            flush=True,
        )

    print()

    if not filtered_chunks:
        raise ValueError(
            "No sales records matched the selected department and stores."
        )

    filtered_sales = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

    return filtered_sales, day_columns


def select_common_items(filtered_sales: pd.DataFrame) -> list[str]:
    """Select products represented in every target store."""

    store_counts = (
        filtered_sales.groupby("item_id")["store_id"]
        .nunique()
    )

    common_items = sorted(
        store_counts[
            store_counts.eq(len(TARGET_STORES))
        ].index
    )

    if len(common_items) < NUMBER_OF_ITEMS:
        raise ValueError(
            f"Only {len(common_items)} common products were found, "
            f"but {NUMBER_OF_ITEMS} are required."
        )

    selected_items = common_items[:NUMBER_OF_ITEMS]

    print(
        f"Selected {len(selected_items)} products "
        f"available in all {len(TARGET_STORES)} stores."
    )

    return selected_items


def create_sales_tables(
    filtered_sales: pd.DataFrame,
    selected_items: list[str],
    day_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create item, store and long-format sales tables."""

    sales_subset = filtered_sales.loc[
        filtered_sales["item_id"].isin(selected_items)
    ].copy()

    expected_series = NUMBER_OF_ITEMS * len(TARGET_STORES)

    if len(sales_subset) != expected_series:
        raise ValueError(
            f"Expected {expected_series} product-store series, "
            f"but found {len(sales_subset)}."
        )

    duplicate_series = sales_subset.duplicated(
        subset=["item_id", "store_id"]
    )

    if duplicate_series.any():
        raise ValueError(
            "Duplicate product-store sales series were found."
        )

    item_dimension = (
        sales_subset[
            [
                "item_id",
                "dept_id",
                "cat_id",
            ]
        ]
        .drop_duplicates()
        .sort_values("item_id")
        .reset_index(drop=True)
    )

    store_dimension = (
        sales_subset[
            [
                "store_id",
                "state_id",
            ]
        ]
        .drop_duplicates()
        .sort_values("store_id")
        .reset_index(drop=True)
    )

    print("Converting sales from wide format to long format...")

    sales_fact = sales_subset.melt(
        id_vars=["item_id", "store_id"],
        value_vars=day_columns,
        var_name="day_id",
        value_name="units_sold",
    )

    sales_fact["units_sold"] = pd.to_numeric(
        sales_fact["units_sold"],
        downcast="integer",
    )

    sales_fact = (
        sales_fact.sort_values(
            ["store_id", "item_id", "day_id"]
        )
        .reset_index(drop=True)
    )

    return item_dimension, store_dimension, sales_fact


def create_calendar_dimension(
    day_columns: list[str],
) -> pd.DataFrame:
    """Create a calendar table matching the available sales days."""

    calendar = pd.read_csv(CALENDAR_FILE)

    calendar_subset = (
        calendar.loc[
            calendar["d"].isin(day_columns)
        ]
        .copy()
        .rename(columns={"d": "day_id"})
    )

    calendar_subset["date"] = pd.to_datetime(
        calendar_subset["date"],
        errors="raise",
    )

    if len(calendar_subset) != len(day_columns):
        raise ValueError(
            "Calendar rows do not match the number of sales days."
        )

    return calendar_subset.sort_values("date").reset_index(
        drop=True
    )


def create_price_fact(
    selected_items: list[str],
) -> pd.DataFrame:
    """Load prices for the selected products and stores."""

    filtered_chunks: list[pd.DataFrame] = []

    print("Filtering weekly selling prices...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            PRICES_FILE,
            usecols=[
                "store_id",
                "item_id",
                "wm_yr_wk",
                "sell_price",
            ],
            chunksize=250_000,
        ),
        start=1,
    ):
        mask = (
            chunk["store_id"].isin(TARGET_STORES)
            & chunk["item_id"].isin(selected_items)
        )

        filtered_chunk = chunk.loc[mask]

        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)

        print(
            f"\rProcessed price chunk {chunk_number}",
            end="",
            flush=True,
        )

    print()

    if not filtered_chunks:
        raise ValueError(
            "No price records matched the selected products and stores."
        )

    price_fact = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

    duplicate_prices = price_fact.duplicated(
        subset=["store_id", "item_id", "wm_yr_wk"]
    )

    if duplicate_prices.any():
        raise ValueError(
            "Duplicate weekly price records were found."
        )

    return (
        price_fact.sort_values(
            ["store_id", "item_id", "wm_yr_wk"]
        )
        .reset_index(drop=True)
    )


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save a table and print its dimensions."""

    output_path = INTERIM_DATA_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"[SAVED] {filename:<25} "
        f"rows={len(dataframe):>8,} "
        f"columns={len(dataframe.columns):>3}"
    )


def main() -> None:
    """Build and save the initial analytical dataset."""

    print("=" * 72)
    print("BUILD INITIAL M5 ANALYTICAL DATASET")
    print("=" * 72)

    INTERIM_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered_sales, day_columns = load_filtered_sales()

    selected_items = select_common_items(filtered_sales)

    (
        item_dimension,
        store_dimension,
        sales_fact,
    ) = create_sales_tables(
        filtered_sales,
        selected_items,
        day_columns,
    )

    calendar_dimension = create_calendar_dimension(
        day_columns
    )

    price_fact = create_price_fact(
        selected_items
    )

    print("\nSaving analytical tables...")
    print("-" * 72)

    save_table(
        item_dimension,
        "dim_items.csv",
    )
    save_table(
        store_dimension,
        "dim_stores.csv",
    )
    save_table(
        calendar_dimension,
        "dim_calendar.csv",
    )
    save_table(
        sales_fact,
        "fact_sales.csv",
    )
    save_table(
        price_fact,
        "fact_prices.csv",
    )

    expected_sales_rows = (
        NUMBER_OF_ITEMS
        * len(TARGET_STORES)
        * len(day_columns)
    )

    if len(sales_fact) != expected_sales_rows:
        raise ValueError(
            f"Expected {expected_sales_rows:,} sales rows, "
            f"but created {len(sales_fact):,}."
        )

    print("-" * 72)
    print("Initial analytical dataset built successfully.")
    print(
        f"Products:             {len(item_dimension):,}"
    )
    print(
        f"Stores:               {len(store_dimension):,}"
    )
    print(
        f"Historical days:      {len(calendar_dimension):,}"
    )
    print(
        f"Product-store series: "
        f"{NUMBER_OF_ITEMS * len(TARGET_STORES):,}"
    )
    print(
        f"Daily sales records:  {len(sales_fact):,}"
    )


if __name__ == "__main__":
    main()