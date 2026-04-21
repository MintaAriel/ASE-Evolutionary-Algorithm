#!/usr/bin/env python3
from __future__ import annotations

import os
from ase.calculators.mixing import SumCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from dataclasses import dataclass
from pathlib import Path
from ea.utils.config import load_config
from deepmd.calculator import DP

from ase.io import read


cfg = load_config()
ms = cfg['deepmd']
MODELS = ms['models_path']

crystal1= read('/home/vito/uspex_matlab/theo_uspex/test_2/CalcFold1/geom.in')
crystal2 = read('/home/vito/uspex_matlab/theo_uspex/test_2/CalcFold2/geom.in')
crystal3 = read('/home/vito/PythonProjects/ASEProject/container_gpu_2/structures/128707.cif')

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


model_path, _ = resolve_model(Path(MODELS),'deepmd_d3')

calc = DP(model=str(model_path), device='gpu')


def run(self, atoms, out_dir):
    log_path = out_dir / "relax.log"
    traj_path = out_dir / "opt.traj"

    # FIRE stage
    fire = FIRE(
        FrechetCellFilter(atoms),
        logfile=str(log_path),
        trajectory=str(traj_path),
        maxstep=self.cfg.maxstep,
    )
    fire.run(fmax=self.cfg.fire_fmax, steps=self.cfg.fire_steps)

    # LBFGS stages
    lbfgs = LBFGS(
        FrechetCellFilter(atoms),
        logfile=str(log_path),
        trajectory=str(traj_path),
        append_trajectory=True,
        maxstep=self.cfg.maxstep,
        memory=40,
    )

    for fmax in self.cfg.lbfgs_stages:
        lbfgs.run(fmax=fmax, steps=self.cfg.lbfgs_steps)

    energy = atoms.get_potential_energy()

    return {
        "optimizer": "fire_then_staged_lbfgs",
        "nsteps": int(fire.nsteps + lbfgs.nsteps),
        "lbfgs_stages": list(self.cfg.lbfgs_stages),
        "energy": energy
    }