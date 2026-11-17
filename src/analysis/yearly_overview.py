
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

con = duckdb.connect()

try:
    for year in range(2018, 2023):
        file_path = DATA_DIR / f"Combined_Flights_{year}.parquet"

        if not file_path.exists():
            print(f"\n{year}: FILE NOT FOUND")
            continue

        query = """
            SELECT
                COUNT(*) AS total_flights,
                MIN(FlightDate) AS first_date,
                MAX(FlightDate) AS last_date,
                COUNT(*) FILTER (
                    WHERE Cancelled = TRUE
                ) AS cancelled,
                COUNT(*) FILTER (
                    WHERE Diverted = TRUE
                ) AS diverted,
                COUNT(*) FILTER (
                    WHERE NOT Cancelled
                      AND NOT Diverted
                      AND ArrDelayMinutes >= 15
                ) AS delayed_15min
            FROM read_parquet(?)
        """

        result = con.execute(
            query, [str(file_path)]
        ).fetchone()

        print(f"\n{'=' * 45}")
        print(f"YEAR: {year}")
        print(f"Total flights: {result[0]:,}")
        print(f"First date: {result[1]}")
        print(f"Last date: {result[2]}")
        print(f"Cancelled: {result[3]:,}")
        print(f"Diverted: {result[4]:,}")
        print(f"Delayed by 15+ minutes: {result[5]:,}")

finally:
    con.close()
