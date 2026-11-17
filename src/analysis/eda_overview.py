
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Read all yearly Parquet files together
PARQUET_GLOB = (DATA_DIR / "Combined_Flights_*.parquet").as_posix()


def main():
    con = duckdb.connect()

    try:
        source = f"read_parquet('{PARQUET_GLOB}', union_by_name=true)"

        # Use only completed flights with a known arrival delay
        eligible = f"""
            SELECT *
            FROM {source}
            WHERE Cancelled = FALSE
              AND Diverted = FALSE
              AND ArrDelayMinutes IS NOT NULL
        """

        print("=" * 70)
        print("FLIGHTWISE AI - EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        # 1. Delay rate by year
        print("\n1. DELAY RATE BY YEAR")

        query = f"""
            SELECT
                Year,
                COUNT(*) AS completed_flights,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) AS delayed_flights,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE ArrDelayMinutes >= 15
                    ) / COUNT(*),
                    2
                ) AS delay_rate_pct
            FROM ({eligible})
            GROUP BY Year
            ORDER BY Year
        """

        print(con.execute(query).fetchdf().to_string(index=False))

        # 2. Delay rate by month
        print("\n2. DELAY RATE BY MONTH")

        query = f"""
            SELECT
                Month,
                COUNT(*) AS completed_flights,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) AS delayed_flights,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE ArrDelayMinutes >= 15
                    ) / COUNT(*),
                    2
                ) AS delay_rate_pct
            FROM ({eligible})
            GROUP BY Month
            ORDER BY Month
        """

        print(con.execute(query).fetchdf().to_string(index=False))

        # 3. Airlines with the highest delay rates
        print("\n3. AIRLINE DELAY RATES")
        print("(Airlines with at least 10,000 eligible flights)")

        query = f"""
            SELECT
                Airline,
                COUNT(*) AS completed_flights,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) AS delayed_flights,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE ArrDelayMinutes >= 15
                    ) / COUNT(*),
                    2
                ) AS delay_rate_pct
            FROM ({eligible})
            GROUP BY Airline
            HAVING COUNT(*) >= 10000
            ORDER BY delay_rate_pct DESC
            LIMIT 15
        """

        print(con.execute(query).fetchdf().to_string(index=False))

        print("\nEDA overview completed successfully!")

    finally:
        con.close()


if __name__ == "__main__":
    main()
