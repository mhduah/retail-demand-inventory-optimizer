"""Load the interim M5 analytical tables into a SQLite database."""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SCHEMA_FILE = PROJECT_ROOT / "sql" / "schema.sql"
DATABASE_FILE = PROCESSED_DATA_DIR / "retail_inventory.db"

TABLE_FILES = {
    "dim_items": INTERIM_DATA_DIR / "dim_items.csv",
    "dim_stores": INTERIM_DATA_DIR / "dim_stores.csv",
    "dim_calendar": INTERIM_DATA_DIR / "dim_calendar.csv",
    "fact_sales": INTERIM_DATA_DIR / "fact_sales.csv",
    "fact_prices": INTERIM_DATA_DIR / "fact_prices.csv",
}

EXPECTED_ROW_COUNTS = {
    "dim_items": 50,
    "dim_stores": 3,
    "dim_calendar": 1_941,
    "fact_sales": 291_150,
    "fact_prices": 31_207,
}


def validate_input_files() -> None:
    """Confirm that the schema and source tables exist."""

    required_files = [SCHEMA_FILE, *TABLE_FILES.values()]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        formatted_paths = "\n".join(
            str(path)
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following required files are missing:\n"
            f"{formatted_paths}"
        )


def create_database_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create all database tables, constraints, and indexes."""

    schema_sql = SCHEMA_FILE.read_text(
        encoding="utf-8"
    )

    connection.executescript(schema_sql)


def load_table(
    connection: sqlite3.Connection,
    table_name: str,
    csv_path: Path,
) -> None:
    """Load one CSV table into the SQLite database."""

    dataframe = pd.read_csv(csv_path)

    dataframe.to_sql(
        table_name,
        connection,
        if_exists="append",
        index=False,
        chunksize=10_000,
        
    )

    print(
        f"[LOADED] {table_name:<15} "
        f"rows={len(dataframe):>8,}"
    )


def get_database_row_count(
    connection: sqlite3.Connection,
    table_name: str,
) -> int:
    """Return the number of records stored in a database table."""

    query = f"SELECT COUNT(*) FROM {table_name};"

    result = connection.execute(query).fetchone()

    if result is None:
        raise RuntimeError(
            f"Could not obtain a row count for {table_name}."
        )

    return int(result[0])


def validate_database(
    connection: sqlite3.Connection,
) -> None:
    """Validate row counts and foreign-key integrity."""

    print("\nValidating database...")
    print("-" * 72)

    for table_name, expected_count in EXPECTED_ROW_COUNTS.items():
        actual_count = get_database_row_count(
            connection,
            table_name,
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{table_name} contains {actual_count:,} rows; "
                f"expected {expected_count:,}."
            )

        print(
            f"[OK] {table_name:<15} "
            f"rows={actual_count:>8,}"
        )

    foreign_key_errors = connection.execute(
        "PRAGMA foreign_key_check;"
    ).fetchall()

    if foreign_key_errors:
        raise ValueError(
            "Foreign-key integrity errors were found:\n"
            f"{foreign_key_errors[:10]}"
        )

    print("[OK] Foreign-key integrity check passed")


def print_business_summary(
    connection: sqlite3.Connection,
) -> None:
    """Print a small business-oriented database summary."""

    query = """
        SELECT
            s.store_id,
            SUM(s.units_sold) AS total_units_sold,
            ROUND(AVG(s.units_sold), 2) AS average_daily_units
        FROM fact_sales AS s
        GROUP BY s.store_id
        ORDER BY total_units_sold DESC;
    """

    summary = pd.read_sql_query(
        query,
        connection,
    )

    print("\nSales summary by store")
    print("-" * 72)
    print(summary.to_string(index=False))


def main() -> None:
    """Build and validate the SQLite analytical database."""

    print("=" * 72)
    print("LOAD M5 ANALYTICAL DATA INTO SQLITE")
    print("=" * 72)

    validate_input_files()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        create_database_schema(connection)

        for table_name, csv_path in TABLE_FILES.items():
            load_table(
                connection,
                table_name,
                csv_path,
            )

        validate_database(connection)
        print_business_summary(connection)

    database_size_mb = (
        DATABASE_FILE.stat().st_size / (1024**2)
    )

    print("\n" + "-" * 72)
    print("SQLite database created successfully.")
    print(f"Database: {DATABASE_FILE}")
    print(f"Size:     {database_size_mb:.2f} MB")


if __name__ == "__main__":
    main()