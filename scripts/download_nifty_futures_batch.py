from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.ingestion.fetch_historical_data import fetch_intraday_data
from src.ingestion.parse_historical_data import dhan_response_to_dataframe


FUTURES_CSV = Path("data/raw/instrument_master/nifty_futures_rows.csv")
RAW_DIR = Path("data/raw/dhan_downloads/futures_json")
PARQUET_DIR = Path("data/interim/merged/futures_parquet")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    futures_df = pd.read_csv(FUTURES_CSV, low_memory=False)

    print("Contracts found:", len(futures_df))
    print(
        futures_df[
            ["SECURITY_ID", "SYMBOL_NAME", "DISPLAY_NAME", "SM_EXPIRY_DATE"]
        ].to_string(index=False)
    )

    for _, row in futures_df.iterrows():
        security_id = str(row["SECURITY_ID"])
        symbol_name = str(row["SYMBOL_NAME"])
        expiry = str(row["SM_EXPIRY_DATE"])

        print(f"\nDownloading contract: {security_id} | {symbol_name} | expiry={expiry}")

        response = fetch_intraday_data(
            security_id=security_id,
            exchange_segment="NSE_FNO",
            instrument="FUTIDX",
            from_date="2026-04-01 09:15:00",
            to_date="2026-04-10 15:30:00",
            interval=15,
            oi=True,
        )

        raw_path = RAW_DIR / f"{security_id}_{symbol_name}.json"
        save_json(response, raw_path)

        df = dhan_response_to_dataframe(response)
        if not df.empty:
            df["security_id"] = security_id
            df["symbol_name"] = symbol_name
            df["expiry_date"] = expiry

        parquet_path = PARQUET_DIR / f"{security_id}_{symbol_name}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        print(f"Saved JSON: {raw_path}")
        print(f"Saved Parquet: {parquet_path}")
        print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
