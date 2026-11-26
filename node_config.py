import json
import os
from pathlib import Path
from typing import Any


def get_data_dir() -> Path:
    return Path(os.getenv("AXIOM_DATA_DIR", "data"))


def load_node_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "observer_id": "OBS-IL-01",
        "latitude": 40.4842,
        "longitude": -88.9937,
        "elevation_m": 250.0,
        "profile_key": "dark_sky",
        "spacetrack_user": "",
        "data_source": "synthetic",
        "replay_file_path": "",
    }
    cfg_path = get_data_dir() / "config" / "node_config.json"
    if not cfg_path.exists():
        return defaults
    try:
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return {**defaults, **loaded}
    except Exception:
        pass
    return defaults