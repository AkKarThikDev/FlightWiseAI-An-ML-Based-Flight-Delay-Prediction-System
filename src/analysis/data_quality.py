
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Combined_Flights_2022.parquet"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    con = duckdb.connect()

    try:
        file_path = str(DATA_PATH)

        print("=" * 65)
        print("FLIGHTWISE AI - DATA QUALITY ANALYSIS")
        print("=" * 65)

        # 1. Overall dataset statistics
        print("\n1. OVERALL DATA QUALITY")

        query = """
            SELECT
                COUNT(*) AS total_flights,
                COUNT(*) FILTER (
                    WHERE Cancelled = TRUE
                ) AS cancelled_flights,
                COUNT(*) FILTER (
                    WHERE Diverted = TRUE
                ) AS diverted_flights,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes IS NULL
                ) AS missing_arrival_delay,
                MIN(FlightDate) AS first_flight_date,
                MAX(FlightDate) AS last_flight_date
            FROM read_parquet(?)
        """

        result = con.execute(query, [file_path]).fetchdf()
        print(result.to_string(index=False))

        # 2. Check target availability among eligible flights
        print("\n2. COMPLETED FLIGHTS AND TARGET AVAILABILITY")

        query = """
            SELECT
                COUNT(*) AS eligible_flights,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes IS NULL
                ) AS missing_target,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) AS delayed_15min_or_more,
                COUNT(*) FILTER (
                    WHERE ArrDelayMinutes < 15
                ) AS below_15min_delay
            FROM read_parquet(?)
            WHERE Cancelled = FALSE
              AND Diverted = FALSE
        """

        result = con.execute(query, [file_path]).fetchdf()
        print(result.to_string(index=False))

        # 3. Check missing values in important columns
        print("\n3. MISSING VALUES IN IMPORTANT COLUMNS")

        query = """
            SELECT
                COUNT(*) FILTER (
                    WHERE FlightDate IS NULL
                ) AS missing_flight_date,
                COUNT(*) FILTER (
                    WHERE Airline IS NULL
                ) AS missing_airline,
                COUNT(*) FILTER (
                    WHERE Origin IS NULL
                ) AS missing_origin,
                COUNT(*) FILTER (
                    WHERE Dest IS NULL
                ) AS missing_destination,
                COUNT(*) FILTER (
                    WHERE CRSDepTime IS NULL
                ) AS missing_scheduled_departure,
                COUNT(*) FILTER (
                    WHERE CRSArrTime IS NULL
                ) AS missing_scheduled_arrival,
                COUNT(*) FILTER (
                    WHERE Distance IS NULL
                ) AS missing_distance
            FROM read_parquet(?)
        """

        result = con.execute(query, [file_path]).fetchdf()
        print(result.to_string(index=False))

        # 4. Calculate target distribution after excluding
        # cancelled, diverted, and missing-target records
        print("\n4. TARGET CLASS DISTRIBUTION")

        query = """
            SELECT
                CASE
                    WHEN ArrDelayMinutes >= 15 THEN 'Delayed'
                    ELSE 'Not delayed'
                END AS flight_status,
                COUNT(*) AS flight_count,
                ROUND(
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage
            FROM read_parquet(?)
            WHERE Cancelled = FALSE
              AND Diverted = FALSE
              AND ArrDelayMinutes IS NOT NULL
            GROUP BY flight_status
            ORDER BY flight_status
        """

        result = con.execute(query, [file_path]).fetchdf()
        print(result.to_string(index=False))

        print("\nData quality analysis completed!")

    finally:
        con.close()


if __name__ == "__main__":
    main()
