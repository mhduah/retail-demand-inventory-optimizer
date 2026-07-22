"""Data-quality tests for the initial M5 analytical dataset."""

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

EXPECTED_ITEMS = 50
EXPECTED_STORES = {"CA_1", "TX_1", "WI_1"}
EXPECTED_DAYS = 1_941
EXPECTED_SERIES = EXPECTED_ITEMS * len(EXPECTED_STORES)
EXPECTED_SALES_ROWS = EXPECTED_SERIES * EXPECTED_DAYS

TABLE_PATHS = {
    "items": INTERIM_DATA_DIR / "dim_items.csv",
    "stores": INTERIM_DATA_DIR / "dim_stores.csv",
    "calendar": INTERIM_DATA_DIR / "dim_calendar.csv",
    "sales": INTERIM_DATA_DIR / "fact_sales.csv",
    "prices": INTERIM_DATA_DIR / "fact_prices.csv",
}


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    """Load all generated analytical tables once per test session."""

    missing_files = [
        str(path)
        for path in TABLE_PATHS.values()
        if not path.exists()
    ]

    assert not missing_files, (
        "Required interim files are missing:\n"
        + "\n".join(missing_files)
    )

    return {
        name: pd.read_csv(path)
        for name, path in TABLE_PATHS.items()
    }


def test_dimension_sizes(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Check the expected number of products, stores and dates."""

    assert len(tables["items"]) == EXPECTED_ITEMS
    assert len(tables["stores"]) == len(EXPECTED_STORES)
    assert len(tables["calendar"]) == EXPECTED_DAYS

    assert set(tables["stores"]["store_id"]) == EXPECTED_STORES


def test_dimension_primary_keys_are_unique(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Ensure dimension-table identifiers are unique."""

    assert tables["items"]["item_id"].is_unique
    assert tables["stores"]["store_id"].is_unique
    assert tables["calendar"]["day_id"].is_unique


def test_sales_table_shape(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Check the expected number of sales records."""

    sales = tables["sales"]

    assert len(sales) == EXPECTED_SALES_ROWS

    series_count = (
        sales[["item_id", "store_id"]]
        .drop_duplicates()
        .shape[0]
    )

    assert series_count == EXPECTED_SERIES


def test_sales_composite_key_is_unique(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Ensure each product-store-day record occurs once."""

    sales = tables["sales"]

    duplicates = sales.duplicated(
        subset=["item_id", "store_id", "day_id"]
    )

    assert not duplicates.any()


def test_sales_values_are_valid(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Check that sales are present, nonnegative integers."""

    units = tables["sales"]["units_sold"]

    assert units.notna().all()
    assert (units >= 0).all()
    assert ((units % 1) == 0).all()


def test_every_series_contains_all_days(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Ensure every product-store series contains d_1 through d_1941."""

    sales = tables["sales"]

    expected_days = {
        f"d_{day_number}"
        for day_number in range(1, EXPECTED_DAYS + 1)
    }

    grouped = sales.groupby(
        ["item_id", "store_id"],
        sort=False,
    )

    for series_key, group in grouped:
        actual_days = set(group["day_id"])

        assert actual_days == expected_days, (
            f"Incomplete day coverage for series {series_key}"
        )


def test_sales_are_chronologically_ordered(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Ensure every series is ordered from d_1 through d_1941."""

    sales = tables["sales"]

    expected_order = [
        f"d_{day_number}"
        for day_number in range(1, EXPECTED_DAYS + 1)
    ]

    grouped = sales.groupby(
        ["item_id", "store_id"],
        sort=False,
    )

    for series_key, group in grouped:
        actual_order = group["day_id"].tolist()

        assert actual_order == expected_order, (
            f"Incorrect chronological order for series {series_key}. "
            f"First days found: {actual_order[:10]}"
        )


def test_sales_foreign_keys(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Check that sales identifiers exist in their dimensions."""

    sales = tables["sales"]

    assert set(sales["item_id"]).issubset(
        set(tables["items"]["item_id"])
    )

    assert set(sales["store_id"]).issubset(
        set(tables["stores"]["store_id"])
    )

    assert set(sales["day_id"]).issubset(
        set(tables["calendar"]["day_id"])
    )


def test_price_records_are_valid(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Validate weekly price records."""

    prices = tables["prices"]

    duplicates = prices.duplicated(
        subset=["store_id", "item_id", "wm_yr_wk"]
    )

    assert not duplicates.any()
    assert prices["sell_price"].notna().all()
    assert (prices["sell_price"] > 0).all()

    assert set(prices["item_id"]).issubset(
        set(tables["items"]["item_id"])
    )

    assert set(prices["store_id"]).issubset(
        set(tables["stores"]["store_id"])
    )