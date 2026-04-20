from __future__ import annotations

import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/raw/instrument_master/api_scrip_master_detailed.csv")


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print("All columns:")
    print(df.columns.tolist())

    search_cols = [
        col for col in df.columns
        if any(x in col.lower() for x in ["name", "symbol", "trading", "display"])
    ]
    print("\nCandidate search columns:")
    print(search_cols)

    mask = pd.Series(False, index=df.index)
    for col in search_cols:
        mask = mask | df[col].astype(str).str.contains("NIFTY", case=False, na=False)

    result = df.loc[mask].copy()

    print(f"\nRows containing 'NIFTY': {len(result)}")
    print(result.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
