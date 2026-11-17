
from pathlib import Path
import duckdb

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "Combined_Flights_2022.parquet"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    con = duckdb.connect()

    try:
        print("=" * 60)
        print("FLIGHTWISE AI - DATA INSPECTION")
        print("=" * 60)

        # 1. Inspect column names and data types
        print("\n1. DATASET SCHEMA")
        schema = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(DATA_PATH)]
        ).fetchdf()
        print(schema.to_string(index=False))

        # 2. Count records
        print("\n2. TOTAL RECORDS")
        count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(DATA_PATH)]
        ).fetchone()[0]
        print(f"Total records: {count:,}")

        # 3. Display sample records
        print("\n3. FIRST FIVE RECORDS")
        sample = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT 5",
            [str(DATA_PATH)]
        ).fetchdf()
        print(sample.to_string(index=False))

        print("\nData inspection completed successfully!")

    finally:
        con.close()


if __name__ == "__main__":
    main()
