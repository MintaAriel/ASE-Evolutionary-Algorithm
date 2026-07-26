"""DeepMD model registry shared by the USPEX relaxation workers.

Keys are the ``--model`` choices accepted by the workers; values are file
names inside the ``deepmd.models_path`` directory of the EA config.
"""

from pathlib import Path

MODELS = {
    'base_deepmd': 'dpa3_12.03.2026.pth',
    'deepmd_d3': 'dpa3-d3_torch.pth',
    'deepmd_d4': 'dpa3-d4.pth',
    'deepmd_d3_abs': 'dpa3-d3_abs_torch.pth',
    'deepmd_d3_mbj': 'dpa3-d3-cpu_mbj.pth',
    'deepmd_d3_mbj_abs': 'dpa3-d3-cpu_mbj_abs.pth',
}


def resolve_model(models_dir, key):
    """Absolute path of model *key* under *models_dir*, checked for existence."""
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; "
                       f"known: {', '.join(sorted(MODELS))}")
    path = Path(models_dir) / MODELS[key]
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path
