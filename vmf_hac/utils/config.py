import json
from typing import Any

from vmf_hac.definitions import ROOT_DIR


def get_config() -> dict[str, Any]:
    config_path = ROOT_DIR / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    return config
