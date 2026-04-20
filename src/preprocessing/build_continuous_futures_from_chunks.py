from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/interim/merged/futures_chunks_parquet")
MERGED_OUTPUT = Path("data/interim/merged/nifty_futures_chunks_master.parquet")
CONTINUOUS_OUTPUT = Path("data/interim/cleaned/nifty_continuous_futures_from_chunks.parquet")


def load_and_merge_parquets(input_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    frames = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        if not df.empty:
            df["source_file"] = file.name
            frames.append(df)

    if not frames:
        raise ValueError("All parquet files are empty.")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def clean_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp_ist"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

    # Drop exact duplicates across chunks
    dedupe_cols = ["timestamp", "security_id", "symbol_name", "expiry_date"]
    df = df.drop_duplicates(subset=dedupe_cols).copy()

    df["trade_date"] = df["timestamp_ist"].dt.date
    df["trade_time"] = df["timestamp_ist"].dt.strftime("%H:%M:%S")

    # days_to_expiry using naive comparison
    df["days_to_expiry"] = (
        df["expiry_date"] - df["timestamp_ist"].dt.tz_localize(None)
    ).dt.days

    # Keep non-expired rows only
    df = df[df["days_to_expiry"] >= 0].copy()

    df = df.sort_values(["timestamp_ist", "days_to_expiry", "expiry_date", "security_id"]).reset_index(drop=True)
    return df


def build_continuous_series(df: pd.DataFrame) -> pd.DataFrame:
    continuous = df.groupby("timestamp_ist", as_index=False).first()

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
        "chunk_start",
        "chunk_end",
        "source_file",
    ]
    existing_cols = [c for c in preferred_cols if c in continuous.columns]
    remaining_cols = [c for c in continuous.columns if c not in existing_cols]
    continuous = continuous[existing_cols + remaining_cols]

    return continuous


def main() -> None:
    merged = load_and_merge_parquets(INPUT_DIR)
    print("Raw merged shape:", merged.shape)

    cleaned = clean_merged_data(merged)
    print("Cleaned merged shape:", cleaned.shape)

    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(MERGED_OUTPUT, index=False)
    print(f"Saved merged cleaned dataset to: {MERGED_OUTPUT}")

    continuous = build_continuous_series(cleaned)
    print("Continuous dataset shape:", continuous.shape)

    print("\nContracts used:")
    print(
        continuous[["security_id", "symbol_name", "expiry_date"]]
        .drop_duplicates()
        .sort_values(["expiry_date", "security_id"])
        .to_string(index=False)
    )

    print("\nSample rows:")
    print(
        continuous[
            ["timestamp_ist", "close", "open_interest", "security_id", "symbol_name", "expiry_date", "days_to_expiry"]
        ].head(15).to_string(index=False)
    )

    CONTINUOUS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    continuous.to_parquet(CONTINUOUS_OUTPUT, index=False)
    print(f"\nSaved continuous futures dataset to: {CONTINUOUS_OUTPUT}")


if __name__ == "__main__":
    main()
