from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/interim/cleaned/nifty_continuous_futures_from_chunks.parquet")
SUMMARY_OUTPUT = Path("data/interim/cleaned/nifty_day_completeness_summary.parquet")
FILTERED_OUTPUT = Path("data/interim/cleaned/nifty_continuous_futures_complete_days.parquet")

EXPECTED_BARS_PER_DAY = 25
MIN_REQUIRED_BARS = 20


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    day_summary = (
        df.groupby("trade_date")
        .agg(
            bars_in_day=("timestamp_ist", "count"),
            first_bar=("timestamp_ist", "min"),
            last_bar=("timestamp_ist", "max"),
        )
        .reset_index()
    )

    day_summary["expected_bars"] = EXPECTED_BARS_PER_DAY
    day_summary["completeness_ratio"] = day_summary["bars_in_day"] / EXPECTED_BARS_PER_DAY
    day_summary["is_complete_day"] = (day_summary["bars_in_day"] >= MIN_REQUIRED_BARS).astype(int)

    print("Day completeness summary:")
    print(day_summary.head(15).to_string(index=False))

    print("\nCompleteness distribution:")
    print(day_summary["is_complete_day"].value_counts(dropna=False).to_string())

    print("\nDays with insufficient bars:")
    print(
        day_summary.loc[day_summary["is_complete_day"] == 0, ["trade_date", "bars_in_day", "completeness_ratio"]]
        .head(20)
        .to_string(index=False)
    )

    # Save summary
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    day_summary.to_parquet(SUMMARY_OUTPUT, index=False)

    # Keep only complete-enough trading days
    valid_days = set(day_summary.loc[day_summary["is_complete_day"] == 1, "trade_date"])
    filtered_df = df[df["trade_date"].isin(valid_days)].copy()

    filtered_df.to_parquet(FILTERED_OUTPUT, index=False)

    print(f"\nSaved day summary to: {SUMMARY_OUTPUT}")
    print(f"Saved filtered complete-days dataset to: {FILTERED_OUTPUT}")
    print("\nFiltered dataset shape:", filtered_df.shape)


if __name__ == "__main__":
    main()
