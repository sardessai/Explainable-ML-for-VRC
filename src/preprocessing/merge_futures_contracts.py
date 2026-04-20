from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/interim/merged/futures_parquet")
OUTPUT_PATH = Path("data/interim/merged/nifty_futures_master.parquet")


def main() -> None:
    parquet_files = sorted(INPUT_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    frames = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df["source_file"] = file.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Convert timestamp to UTC then IST
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce", utc=True)
    merged["timestamp_ist"] = merged["timestamp"].dt.tz_convert("Asia/Kolkata")

    merged["trade_date"] = merged["timestamp_ist"].dt.date
    merged["trade_time"] = merged["timestamp_ist"].dt.strftime("%H:%M:%S")

    merged = merged.sort_values(["timestamp_ist", "expiry_date", "security_id"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print("Merged shape:", merged.shape)
    print("\nColumns:", merged.columns.tolist())
    print("\nContracts in merged file:")
    print(merged[["security_id", "symbol_name", "expiry_date"]].drop_duplicates().to_string(index=False))
    print(f"\nSaved merged futures dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
