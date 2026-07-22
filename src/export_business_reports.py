"""Execute the business SQL queries and export analysis tables."""

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_inventory.db"
)

SQL_FILE = (
    PROJECT_ROOT
    / "sql"
    / "business_queries.sql"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

QUERY_OUTPUTS = [
    (
        "dataset_coverage.csv",
        "Overall dataset coverage",
    ),
    (
        "sales_by_store.csv",
        "Sales performance by store",
    ),
    (
        "revenue_by_store.csv",
        "Estimated revenue by store",
    ),
    (
        "top_products_by_demand.csv",
        "Top products by demand",
    ),
    (
        "top_products_by_revenue.csv",
        "Top products by estimated revenue",
    ),
    (
        "monthly_demand_trend.csv",
        "Monthly demand trend",
    ),
    (
        "demand_by_weekday.csv",
        "Demand by weekday",
    ),
    (
        "price_data_coverage.csv",
        "Price-data coverage",
    ),
]


def validate_required_files() -> None:
    """Verify that the database and SQL file exist."""

    missing_files = [
        path
        for path in [DATABASE_FILE, SQL_FILE]
        if not path.exists()
    ]

    if missing_files:
        formatted = "\n".join(
            str(path)
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{formatted}"
        )


def read_sql_queries() -> list[str]:
    """Read and separate the SQL statements."""

    sql_text = SQL_FILE.read_text(
        encoding="utf-8-sig"
    )

    statements = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]

    expected_query_count = len(QUERY_OUTPUTS)

    if len(statements) != expected_query_count:
        raise ValueError(
            f"Expected {expected_query_count} SQL queries, "
            f"but found {len(statements)}."
        )

    return statements


def execute_and_export_queries(
    connection: sqlite3.Connection,
    queries: list[str],
) -> None:
    """Execute every query and export its result as CSV."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nExecuting business queries...")
    print("-" * 76)

    for query_number, (
        query,
        output_definition,
    ) in enumerate(
        zip(
            queries,
            QUERY_OUTPUTS,
            strict=True,
        ),
        start=1,
    ):
        filename, title = output_definition
        output_path = OUTPUT_DIRECTORY / filename

        try:
            result = pd.read_sql_query(
                query,
                connection,
            )
        except Exception as error:
            query_preview = " ".join(
                query.split()
            )[:180]

            raise RuntimeError(
                f"Query {query_number} failed: {title}\n"
                f"Query preview: {query_preview}"
            ) from error

        result.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"[EXPORTED] Query {query_number}: "
            f"{filename:<35} "
            f"rows={len(result):>4}"
        )


def print_key_results() -> None:
    """Print selected employer-facing results."""

    store_sales = pd.read_csv(
        OUTPUT_DIRECTORY / "sales_by_store.csv"
    )

    revenue = pd.read_csv(
        OUTPUT_DIRECTORY / "revenue_by_store.csv"
    )

    top_products = pd.read_csv(
        OUTPUT_DIRECTORY
        / "top_products_by_demand.csv"
    )

    print("\nSales performance by store")
    print("-" * 76)
    print(store_sales.to_string(index=False))

    print("\nEstimated revenue by store")
    print("-" * 76)
    print(revenue.to_string(index=False))

    print("\nFive highest-demand products")
    print("-" * 76)
    print(
        top_products.head(5).to_string(
            index=False
        )
    )


def main() -> None:
    """Run and export the complete business analysis."""

    print("=" * 76)
    print("EXPORT RETAIL BUSINESS ANALYSIS")
    print("=" * 76)

    validate_required_files()
    queries = read_sql_queries()

    with sqlite3.connect(
        DATABASE_FILE
    ) as connection:
        execute_and_export_queries(
            connection,
            queries,
        )

    print_key_results()

    print("\n" + "-" * 76)
    print(
        f"All reports exported to:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()