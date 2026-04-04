"""
Trinetra — Config & Logger utilities
"""

import yaml
from pathlib import Path
from loguru import logger
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = str(BASE_DIR / 'configs' / 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str | None = None, level: str = "INFO"):
    logger.remove()
    logger.add(sys.stdout, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(f"{log_dir}/trinetra.log", rotation="10 MB", level="DEBUG")
    return logger


def get_project_root() -> Path:
    return BASE_DIR
