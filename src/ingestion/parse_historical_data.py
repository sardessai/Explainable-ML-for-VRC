from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def load_dhan_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dhan_response_to_dataframe(response: dict) -> pd.DataFrame:
    expected_keys = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]

    data = {}
    for key in expected_keys:
        if key in response:
            data[key] = response[key]
        else:
            data[key] = []

    df = pd.DataFrame(data)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "open_interest"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    input_path = Path("data/raw/dhan_downloads/nifty_fut_66691_sample.json")
    output_path = Path("data/interim/merged/nifty_fut_66691_sample.parquet")

    response = load_dhan_json(input_path)
    df = dhan_response_to_dataframe(response)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nHead:")
    print(df.head().to_string(index=False))

    save_parquet(df, output_path)
    print(f"\nSaved parquet to: {output_path}")
