"""Validate the raw M5 dataset files.

This script checks that the required source files exist, verifies their
essential columns, and prints a small summary without loading the complete
sales dataset into memory.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "calendar": RAW_DATA_DIR / "calendar.csv",
    "prices": RAW_DATA_DIR / "sell_prices.csv",
    "sales": RAW_DATA_DIR / "sales_train_evaluation.csv",
}

REQUIRED_COLUMNS = {
    "calendar": {
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
    },
    "prices": {
        "store_id",
        "item_id",
        "wm_yr_wk",
        "sell_price",
    },
    "sales": {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    },
}


def file_size_mb(path: Path) -> float:
    """Return a file size in megabytes."""

    return path.stat().st_size / (1024**2)


def validate_file_exists(name: str, path: Path) -> None:
    """Raise an error when a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file '{name}' was not found at:\n{path}"
        )

    if not path.is_file():
        raise ValueError(f"Expected a file, but found something else: {path}")


def validate_columns(
    name: str,
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Check that all required columns are present."""

    available_columns = set(dataframe.columns)
    missing_columns = required_columns - available_columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def validate_sales_day_columns(dataframe: pd.DataFrame) -> list[str]:
    """Validate and return the daily sales columns."""

    day_columns = [
        column
        for column in dataframe.columns
        if column.startswith("d_")
    ]

    if not day_columns:
        raise ValueError(
            "The sales file does not contain any daily columns such as d_1."
        )

    if "d_1" not in day_columns:
        raise ValueError("The sales file does not contain the expected d_1 column.")

    return day_columns


def main() -> int:
    """Run all validation checks."""

    print("=" * 70)
    print("M5 RAW DATA VALIDATION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}\n")

    try:
        for name, path in FILES.items():
            validate_file_exists(name, path)
            print(
                f"[OK] {path.name:<30} "
                f"{file_size_mb(path):>10.2f} MB"
            )

        print("\nReading file headers and small samples...")

        calendar = pd.read_csv(FILES["calendar"], nrows=5)
        prices = pd.read_csv(FILES["prices"], nrows=5)
        sales = pd.read_csv(FILES["sales"], nrows=5)

        validate_columns(
            "calendar.csv",
            calendar,
            REQUIRED_COLUMNS["calendar"],
        )
        validate_columns(
            "sell_prices.csv",
            prices,
            REQUIRED_COLUMNS["prices"],
        )
        validate_columns(
            "sales_train_evaluation.csv",
            sales,
            REQUIRED_COLUMNS["sales"],
        )

        day_columns = validate_sales_day_columns(sales)

        print("\nColumn checks")
        print("-" * 70)
        print(f"[OK] Calendar columns: {len(calendar.columns)}")
        print(f"[OK] Price columns:    {len(prices.columns)}")
        print(f"[OK] Sales columns:    {len(sales.columns)}")
        print(f"[OK] Daily columns:    {len(day_columns)}")
        print(f"[OK] First sales day:  {day_columns[0]}")
        print(f"[OK] Last sales day:   {day_columns[-1]}")

        print("\nSample identifiers")
        print("-" * 70)
        print(
            sales[
                [
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                ]
            ].head()
        )

        print("\nValidation completed successfully.")
        return 0

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nVALIDATION FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())