from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from src.config.settings import load_env_config
from src.ingestion.dhan_client import build_dhan_client


def fetch_intraday_data(
    security_id: str,
    exchange_segment: str,
    instrument: str,
    from_date: str,
    to_date: str,
    interval: int = 15,
    oi: bool = False,
    expiry_code: int | None = None,
) -> Dict[str, Any]:
    client = build_dhan_client()

    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": interval,
        "fromDate": from_date,
        "toDate": to_date,
        "oi": oi,
    }

    if expiry_code is not None:
        payload["expiryCode"] = expiry_code

    print("Payload:", payload)
    response = client.post("/charts/intraday", payload)
    return response


def save_raw_response(response: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    env = load_env_config()

    response = fetch_intraday_data(
        security_id="66691",
        exchange_segment="NSE_FNO",
        instrument="FUTIDX",
        from_date="2026-04-01 09:15:00",
        to_date="2026-04-10 15:30:00",
        interval=15,
        oi=True,
    )

    output_path = env.data_raw_dir / "dhan_downloads" / "nifty_fut_66691_sample.json"
    save_raw_response(response, output_path)

    print(f"Saved sample response to: {output_path}")
    if isinstance(response, dict):
        print("Top-level keys:", list(response.keys()))
        for key, value in response.items():
            if isinstance(value, list):
                print(f"{key}: list with {len(value)} items")