from dataclasses import dataclass
from pathlib import Path

@dataclass
class RelaxConfig:
    model_key: str = "base_deepmd"
    fire_fmax: float = 0.10
    fire_steps: int = 500
    lbfgs_stages: tuple = (0.03, 0.01, 0.005, 0.002, 0.001)
    lbfgs_steps: int = 1200
    maxstep: float = 0.03
    threads: int = 2
    cores: list[int] | None = None

MODELS = {
    'base_deepmd': 'dpa3_12.03.2026.pth',
    'deepmd_d3': 'dpa3-d3_torch.pth',
    'deepmd_d4': 'dpa3-d4.pth',
    'deepmd_d3_abs': 'dpa3-d3_abs_torch.pth',
    'deepmd_d3_mbj': 'dpa3-d3-cpu_mbj.pth',
    'deepmd_d3_mbj_abs': 'dpa3-d3-cpu_mbj_abs.pth',
}

def resolve_model(models_dir: Path, key: str) -> tuple[Path, str]:
    path = models_dir / MODELS[key]
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path, key