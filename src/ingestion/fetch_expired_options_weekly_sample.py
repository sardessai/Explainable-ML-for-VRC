from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
from pprint import pprint

from src.ingestion.dhan_client import build_dhan_client
from src.config.settings import load_env_config


def fetch_expired_options_weekly_sample() -> Dict[str, Any]:
    client = build_dhan_client()

    payload = {
        "exchangeSegment": "NSE_FNO",
        "interval": "15",
        "securityId": 26000,
        "instrument": "OPTIDX",
        "expiryFlag": "WEEK",
        "expiryCode": 1,
        "strike": "ATM",
        "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "iv", "volume", "oi", "spot", "strike"],
        "fromDate": "2026-04-01",
        "toDate": "2026-04-10",
    }

    print("Payload:")
    pprint(payload)

    response = client.post("/charts/rollingoption", payload)
    return response


def save_raw_response(response: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    env = load_env_config()

    response = fetch_expired_options_weekly_sample()

    output_path = env.data_raw_dir / "dhan_downloads" / "expired_options_weekly_atm_call_sample.json"
    save_raw_response(response, output_path)

    print(f"\nSaved sample response to: {output_path}")

    if response is None:
        print("Response is None")
    elif not isinstance(response, dict):
        print("Response is not a dict:")
        print(response)
    else:
        print("Top-level keys:", list(response.keys()))
        data = response.get("data")

        if data is None:
            print("response['data'] is None")
            print("Full response:")
            pprint(response)
        else:
            print("Data keys:", list(data.keys()) if isinstance(data, dict) else data)

            ce = data.get("ce") if isinstance(data, dict) else None
            pe = data.get("pe") if isinstance(data, dict) else None

            if isinstance(ce, dict):
                ce_rows = len(ce.get("timestamp", []) or [])
                print(f"CE rows: {ce_rows}")
                if ce_rows > 0:
                    print("CE keys:", list(ce.keys()))
            else:
                print("CE is missing or not a dict")

            if isinstance(pe, dict):
                pe_rows = len(pe.get("timestamp", []) or [])
                print(f"PE rows: {pe_rows}")
                if pe_rows > 0:
                    print("PE keys:", list(pe.keys()))
            else:
                print("PE is missing or not a dict")
