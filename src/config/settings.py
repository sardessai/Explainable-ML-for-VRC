from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


load_dotenv()


@dataclass
class EnvConfig:
    dhan_client_id: str
    dhan_access_token: str
    project_root: Path
    data_raw_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    output_dir: Path
    log_level: str


def load_env_config() -> EnvConfig:
    """Load environment configuration from .env."""
    return EnvConfig(
        dhan_client_id=os.getenv("DHAN_CLIENT_ID", ""),
        dhan_access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
        project_root=Path(os.getenv("PROJECT_ROOT", ".")),
        data_raw_dir=Path(os.getenv("DATA_RAW_DIR", "data/raw")),
        data_interim_dir=Path(os.getenv("DATA_INTERIM_DIR", "data/interim")),
        data_processed_dir=Path(os.getenv("DATA_PROCESSED_DIR", "data/processed")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    env_config = load_env_config()
    yaml_config = load_yaml_config()

    print("Environment config loaded successfully.")
    print(env_config)
    print("\nYAML config loaded successfully.")
    print(yaml_config)
