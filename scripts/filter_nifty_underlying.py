from __future__ import annotations

from pathlib import Path
import pandas as pd


INPUT_CSV = Path("data/raw/instrument_master/api_scrip_master_detailed.csv")
OUTPUT_CSV = Path("data/raw/instrument_master/nifty_underlying_row.csv")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    result = df[df["SECURITY_ID"].astype(str) == "26000"].copy()

    cols = [
        "EXCH_ID", "SEGMENT", "SECURITY_ID", "INSTRUMENT",
        "UNDERLYING_SECURITY_ID", "UNDERLYING_SYMBOL",
        "SYMBOL_NAME", "DISPLAY_NAME", "INSTRUMENT_TYPE",
        "SERIES", "LOT_SIZE", "SM_EXPIRY_DATE",
        "STRIKE_PRICE", "OPTION_TYPE"
    ]

    available_cols = [c for c in cols if c in result.columns]
    result = result[available_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved NIFTY underlying row to: {OUTPUT_CSV}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
