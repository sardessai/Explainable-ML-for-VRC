from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/interim/merged/nifty_futures_master.parquet")
OUTPUT_PATH = Path("data/interim/cleaned/nifty_continuous_futures.parquet")


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    # Ensure correct datetime types
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

    # Days to expiry
    df["days_to_expiry"] = (df["expiry_date"] - df["timestamp_ist"].dt.tz_localize(None)).dt.days

    # Keep only contracts that have not yet expired
    df = df[df["days_to_expiry"] >= 0].copy()

    # Sort so nearest expiry comes first
    df = df.sort_values(["timestamp_ist", "days_to_expiry", "expiry_date", "security_id"])

    # Pick nearest-expiry contract at each timestamp
    continuous = df.groupby("timestamp_ist", as_index=False).first()

    # Reorder columns a bit
    preferred_cols = [
        "timestamp",
        "timestamp_ist",
        "trade_date",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "security_id",
        "symbol_name",
        "expiry_date",
        "days_to_expiry",
        "source_file",
    ]
    existing_cols = [c for c in preferred_cols if c in continuous.columns]
    remaining_cols = [c for c in continuous.columns if c not in existing_cols]
    continuous = continuous[existing_cols + remaining_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    continuous.to_parquet(OUTPUT_PATH, index=False)

    print("Continuous futures shape:", continuous.shape)
    print("\nColumns:", continuous.columns.tolist())
    print("\nSample rows:")
    print(continuous.head(10).to_string(index=False))
    print("\nContracts used:")
    print(
        continuous[["security_id", "symbol_name", "expiry_date"]]
        .drop_duplicates()
        .sort_values(["expiry_date", "security_id"])
        .to_string(index=False)
    )
    print(f"\nSaved continuous futures dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
