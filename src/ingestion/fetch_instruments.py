from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config.settings import load_env_config
from src.ingestion.dhan_client import build_dhan_client


COMPACT_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DETAILED_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"


def download_instrument_master(detailed: bool = True) -> Path:
    env = load_env_config()
    client = build_dhan_client()

    output_dir = env.data_raw_dir / "instrument_master"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = DETAILED_URL if detailed else COMPACT_URL
    filename = "api_scrip_master_detailed.csv" if detailed else "api_scrip_master.csv"
    output_path = output_dir / filename

    content = client.get_csv(url)
    output_path.write_bytes(content)

    return output_path


def preview_instrument_master(csv_path: Path, n: int = 5) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.head(n)


if __name__ == "__main__":
    csv_path = download_instrument_master(detailed=True)
    print(f"Downloaded instrument master to: {csv_path}")

    df_preview = preview_instrument_master(csv_path)
    print(df_preview)