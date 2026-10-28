# src/pace_pic_validation/io/config.py
import yaml

from pathlib import Path

def load_paths_config(config_name="paths.yaml", config_dir=None):

    project_root = Path(__file__).resolve().parents[3]

    if config_dir is None:
        config_dir = project_root / "configs"

    config_path = Path(config_name)

    if not config_path.is_absolute():
        config_path = Path(config_dir) / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)