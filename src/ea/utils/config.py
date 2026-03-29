from pathlib import Path
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ENV_FILE = PROJECT_ROOT / ".env"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_DEVICE = "cpu"
BATCH_SIZE = 32


def _load_env():
    """Parse .env file and set as environment variable defaults."""
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _deep_merge(base, override):
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config():
    """Load base config overlaid with the machine-specific config.

    The machine config is chosen via the ``EA_CONFIG`` env var
    (also read from ``.env``).  Returns a plain dict.
    """
    _load_env()

    base = yaml.safe_load((CONFIGS_DIR / "base.yaml").read_text()) or {}

    config_name = os.environ.get("EA_CONFIG", "base")
    machine_file = CONFIGS_DIR / f"{config_name}.yaml"
    if machine_file.is_file() and config_name != "base":
        machine = yaml.safe_load(machine_file.read_text()) or {}
        _deep_merge(base, machine)

    return base
