
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

PARQUET_GLOB = (
    DATA_DIR / "Combined_Flights_*.parquet"
).as_posix()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main():
    con = duckdb.connect()

    try:
        source = f"read_parquet('{PARQUET_GLOB}', union_by_name=true)"

        eligible = f"""
            SELECT *
            FROM {source}
            WHERE Cancelled = FALSE
              AND Diverted = FALSE
              AND ArrDelayMinutes IS NOT NULL
        """

        # Chart 1: Yearly delay rate
        yearly = con.execute(f"""
            SELECT
                Year,
                100.0 * COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) / COUNT(*) AS delay_rate
            FROM ({eligible})
            GROUP BY Year
            ORDER BY Year
        """).fetchdf()

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=yearly,
            x="Year",
            y="delay_rate"
        )
        ax.set(
            title="Flight Delay Rate by Year",
            xlabel="Year",
            ylabel="Flights delayed by 15+ minutes (%)"
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3)

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "yearly_delay_rate.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # Chart 2: Monthly delay rate
        monthly = con.execute(f"""
            SELECT
                Month,
                100.0 * COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) / COUNT(*) AS delay_rate
            FROM ({eligible})
            GROUP BY Month
            ORDER BY Month
        """).fetchdf()

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=monthly,
            x="Month",
            y="delay_rate",
            marker="o"
        )
        ax.set(
            title="Flight Delay Rate by Month",
            xlabel="Month",
            ylabel="Flights delayed by 15+ minutes (%)"
        )
        ax.set_xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "monthly_delay_rate.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        # Chart 3: Ten airlines with the highest delay rates
        airlines = con.execute(f"""
            SELECT
                Airline,
                COUNT(*) AS completed_flights,
                100.0 * COUNT(*) FILTER (
                    WHERE ArrDelayMinutes >= 15
                ) / COUNT(*) AS delay_rate
            FROM ({eligible})
            GROUP BY Airline
            HAVING COUNT(*) >= 10000
            ORDER BY delay_rate DESC
            LIMIT 10
        """).fetchdf()

        airlines = airlines.sort_values("delay_rate")

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            data=airlines,
            x="delay_rate",
            y="Airline"
        )
        ax.set(
            title="Airlines with Highest Delay Rates",
            xlabel="Flights delayed by 15+ minutes (%)",
            ylabel="Airline"
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3)

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "airline_delay_rates.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        print("EDA visualizations created successfully!")
        print(f"Saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print("- yearly_delay_rate.png")
        print("- monthly_delay_rate.png")
        print("- airline_delay_rates.png")

    finally:
        con.close()


if __name__ == "__main__":
    main()
