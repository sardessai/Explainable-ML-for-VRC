from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import requests

from src.config.settings import load_env_config


BASE_URL = "https://api.dhan.co/v2"


@dataclass
class DhanClient:
    client_id: str
    access_token: str
    base_url: str = BASE_URL

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }

    def post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)

        if not response.ok:
            print("Status code:", response.status_code)
            print("Response text:", response.text)
            response.raise_for_status()

        return response.json()

    def get_csv(self, url: str, timeout: int = 60) -> bytes:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content


def build_dhan_client() -> DhanClient:
    env = load_env_config()
    if not env.dhan_client_id or not env.dhan_access_token:
        raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in .env")

    return DhanClient(
        client_id=env.dhan_client_id,
        access_token=env.dhan_access_token,
    )


if __name__ == "__main__":
    client = build_dhan_client()
    print("Dhan client created successfully.")
    print(f"Base URL: {client.base_url}")