from __future__ import annotations

from itertools import product
from pprint import pprint

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

        print("\n" + "=" * 100)
        print("Testing payload:")
        pprint(payload)

        try:
            response = client.post("/charts/rollingoption", payload)

            print("Raw response type:", type(response).__name__)

            if response is None:
                print("Response is None")
                continue

            if not isinstance(response, dict):
                print("Response is not a dict:")
                print(response)
                continue

            print("Top-level keys:", list(response.keys()))

            data = response.get("data")
            if data is None:
                print("response['data'] is None")
                print("Full response:")
                pprint(response)
                continue

            print("Data keys:", list(data.keys()) if isinstance(data, dict) else data)

            ce = data.get("ce") if isinstance(data, dict) else None
            pe = data.get("pe") if isinstance(data, dict) else None

            if isinstance(ce, dict):
                ce_rows = len(ce.get("timestamp", []) or [])
                print(f"CE rows: {ce_rows}")
            else:
                print("CE is missing or not a dict")

            if isinstance(pe, dict):
                pe_rows = len(pe.get("timestamp", []) or [])
                print(f"PE rows: {pe_rows}")
            else:
                print("PE is missing or not a dict")

        except Exception as e:
            print("Error:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
