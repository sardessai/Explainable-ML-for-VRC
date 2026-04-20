from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from src.ingestion.dhan_client import build_dhan_client
from src.config.settings import load_env_config


def fetch_expired_options_sample() -> Dict[str, Any]:
    client = build_dhan_client()

    payload = {
        "exchangeSegment": "NSE_FNO",
        "interval": "15",
        "securityId": 26000,
        "instrument": "OPTIDX",
        "expiryFlag": "MONTH",
        "expiryCode": 1,
        "strike": "ATM",
        "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "iv", "volume", "oi", "spot"],
        "fromDate": "2026-04-01",
        "toDate": "2026-04-10",
    }

    print("Payload:", payload)
    response = client.post("/charts/rollingoption", payload)
    return response


def save_raw_response(response: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    env = load_env_config()

    response = fetch_expired_options_sample()

    output_path = env.data_raw_dir / "dhan_downloads" / "expired_options_atm_call_sample.json"
    save_raw_response(response, output_path)

    print(f"Saved sample response to: {output_path}")
    if isinstance(response, dict):
        print("Top-level keys:", list(response.keys()))
        if "data" in response:
            print("Data keys:", list(response["data"].keys()))
            for side, side_data in response["data"].items():
                if isinstance(side_data, dict):
                    print(f"{side} keys:", list(side_data.keys()))
                    for key, value in side_data.items():
                        if isinstance(value, list):
                            print(f"{side}.{key}: list with {len(value)} items")
