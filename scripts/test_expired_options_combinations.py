from __future__ import annotations

from itertools import product
from src.ingestion.dhan_client import build_dhan_client


def main() -> None:
    client = build_dhan_client()

    expiry_codes = [1, 2]
    option_types = ["CALL", "PUT"]
    strikes = ["ATM", "ATM+1", "ATM-1"]

    for expiry_code, option_type, strike in product(expiry_codes, option_types, strikes):
        payload = {
            "exchangeSegment": "NSE_FNO",
            "interval": "15",
            "securityId": 26000,
            "instrument": "OPTIDX",
            "expiryFlag": "MONTH",
            "expiryCode": expiry_code,
            "strike": strike,
            "drvOptionType": option_type,
            "requiredData": ["open", "high", "low", "close", "iv", "volume", "oi", "spot"],
            "fromDate": "2026-04-01",
            "toDate": "2026-04-10",
        }

        print("\nTesting:", payload)

        try:
            response = client.post("/charts/rollingoption", payload)
            ce_len = len(response.get("data", {}).get("ce", {}).get("timestamp", []))
            pe_len = len(response.get("data", {}).get("pe", {}).get("timestamp", []))
            print(f"Result -> ce rows: {ce_len}, pe rows: {pe_len}")
        except Exception as e:
            print("Error:", str(e))


if __name__ == "__main__":
    main()
