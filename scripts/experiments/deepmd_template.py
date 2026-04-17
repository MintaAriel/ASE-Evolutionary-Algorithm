#!/usr/bin/env python3
from __future__ import annotations

import os
import torch
from ase.calculators.mixing import SumCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from deepmd.calculator import DP
from dataclasses import dataclass
from pathlib import Path
#
# _NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

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


class DeepMDRelaxation:
    """Bare DeepMD model — no ASE-level dispersion correction."""

    def __init__(self, config: RelaxConfig):
        self.cfg = config

    def build_calculator(self, models_dir, device, threads=None):
        model_path, _ = resolve_model(models_dir, self.cfg.model_key)

        cores = self.cfg.cores
        nthreads = threads if threads is not None else self.cfg.threads
        if cores is not None:
            os.sched_setaffinity(0, cores)
            print(f"CPU affinity set to cores: {cores}")

        # DeepMD's pt backend captures env.DEVICE at import time via many
        # `from deepmd.pt.utils.env import DEVICE` statements. Patch env.DEVICE
        # and every already-imported deepmd.pt.* module that holds a DEVICE name.
        import sys
        target = torch.device("cpu") if device == "cpu" else torch.device("cuda")
        os.environ["DEVICE"] = "cpu" if device == "cpu" else "cuda"
        from deepmd.pt.utils import env as _dpenv
        _dpenv.DEVICE = target
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("deepmd."):
                continue
            if getattr(mod, "DEVICE", None) is not None and isinstance(
                getattr(mod, "DEVICE"), torch.device
            ):
                mod.DEVICE = target
        calc = DP(model=str(model_path), device=device)
        torch.set_num_threads(nthreads)  # DP() resets threads to 1

        # Force every submodule/buffer onto the target device. torch.jit.load's
        # map_location doesn't always migrate tensors that are constructed inside
        # scripted code (e.g. type_mask), so we explicitly walk into the backend
        # and call .to(target).
        target = torch.device("cpu") if device == "cpu" else torch.device("cuda")
        try:
            model_wrapper = calc.dp.deep_eval.dp
            model_wrapper.to(target)
        except AttributeError as e:
            print(f"(could not move backend to {target}: {e})")

        try:
            p = next(calc.dp.deep_eval.dp.parameters(), None)
            if p is not None:
                print("DP backend device:", p.device)
        except Exception as e:
            print(f"(device check skipped: {e})")
        return calc

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
            "energy":energy
        }


RELAXATION_CLASSES = {
    "deepmd": DeepMDRelaxation,
    # "deepmd_d3": DeepMDRelaxationD3,
    # "deepmd_d4": DeepMDRelaxationD4,
}
