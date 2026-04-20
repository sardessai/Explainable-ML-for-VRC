from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

from src.ingestion.fetch_historical_data import fetch_intraday_data
from src.ingestion.parse_historical_data import dhan_response_to_dataframe


FUTURES_CSV = Path("data/raw/instrument_master/nifty_futures_rows.csv")
RAW_DIR = Path("data/raw/dhan_downloads/futures_chunks_json")
PARQUET_DIR = Path("data/interim/merged/futures_chunks_parquet")

CHUNK_DAYS = 90


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_date_chunks(start_date: str, end_date: str, chunk_days: int = CHUNK_DAYS):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + timedelta(days=1)


def main() -> None:
    futures_df = pd.read_csv(FUTURES_CSV, low_memory=False)

    print("Contracts found:", len(futures_df))
    print(futures_df[["SECURITY_ID", "SYMBOL_NAME", "DISPLAY_NAME", "SM_EXPIRY_DATE"]].to_string(index=False))

    # For now, keep this manageable. Later we can expand to a longer range.
    overall_start = "2026-01-01"
    overall_end = "2026-06-30"

    chunks = list(generate_date_chunks(overall_start, overall_end, CHUNK_DAYS))
    print("\nDate chunks:")
    for cstart, cend in chunks:
        print(f"{cstart} -> {cend}")

    for _, row in futures_df.iterrows():
        security_id = str(row["SECURITY_ID"])
        symbol_name = str(row["SYMBOL_NAME"])
        expiry = str(row["SM_EXPIRY_DATE"])

        print(f"\n=== Contract: {security_id} | {symbol_name} | expiry={expiry} ===")

        for chunk_start, chunk_end in chunks:
            print(f"Downloading chunk: {chunk_start} -> {chunk_end}")

            response = fetch_intraday_data(
                security_id=security_id,
                exchange_segment="NSE_FNO",
                instrument="FUTIDX",
                from_date=f"{chunk_start} 09:15:00",
                to_date=f"{chunk_end} 15:30:00",
                interval=15,
                oi=True,
            )

            raw_path = RAW_DIR / f"{security_id}_{symbol_name}_{chunk_start}_{chunk_end}.json"
            save_json(response, raw_path)

            df = dhan_response_to_dataframe(response)
            if not df.empty:
                df["security_id"] = security_id
                df["symbol_name"] = symbol_name
                df["expiry_date"] = expiry
                df["chunk_start"] = chunk_start
                df["chunk_end"] = chunk_end

            parquet_path = PARQUET_DIR / f"{security_id}_{symbol_name}_{chunk_start}_{chunk_end}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)

            print(f"Saved JSON: {raw_path}")
            print(f"Saved Parquet: {parquet_path}")
            print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
