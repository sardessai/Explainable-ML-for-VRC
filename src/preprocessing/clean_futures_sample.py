from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/interim/merged/nifty_fut_66691_sample.parquet")
OUTPUT_PATH = Path("data/interim/cleaned/nifty_fut_66691_sample_cleaned.parquet")


def convert_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp_ist"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(subset=["timestamp_ist"]).sort_values("timestamp_ist")
    df = df.reset_index(drop=True)

    df["trade_date"] = df["timestamp_ist"].dt.date
    df["trade_time"] = df["timestamp_ist"].dt.strftime("%H:%M:%S")

    return df


def validate_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    market_open = "09:15:00"
    market_close = "15:30:00"

    df["in_market_hours"] = (
        (df["trade_time"] >= market_open) &
        (df["trade_time"] <= market_close)
    )

    return df


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("Original shape:", df.shape)
    print(df.head().to_string(index=False))

    df = convert_to_ist(df)
    df = basic_cleaning(df)
    df = validate_session(df)

    print("\nCleaned shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nHead after IST conversion:")
    print(df.head().to_string(index=False))

    print("\nMarket-hours value counts:")
    print(df["in_market_hours"].value_counts(dropna=False).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved cleaned file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
