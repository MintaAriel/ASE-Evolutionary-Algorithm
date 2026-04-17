#!/usr/bin/env python3
from __future__ import annotations

import os
from ase.calculators.mixing import SumCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
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

def configure_cpu_runtime(cores: list[int], nthreads: int) -> None:
    # Restrict this process to the requested CPUs
    os.sched_setaffinity(0, set(cores))

    # Threading env vars for OpenMP / MKL / DeePMD / BLAS
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

    # DeePMD CPU parallelism
    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(nthreads)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = "1"



class DeepMDRelaxation:
    """Bare DeepMD model — no ASE-level dispersion correction."""

    def __init__(self, config: RelaxConfig):
        self.cfg = config

    def build_calculator(self, models_dir, device, threads=None):


        cores = self.cfg.cores
        nthreads = threads if threads is not None else self.cfg.threads

        configure_cpu_runtime(cores, nthreads)

        # Import only after affinity + env are set
        import torch
        from deepmd.calculator import DP
        import sys

        print("CPU affinity now:", sorted(os.sched_getaffinity(0)))
        print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
        print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
        print("DP_INTRA_OP_PARALLELISM_THREADS =", os.environ.get("DP_INTRA_OP_PARALLELISM_THREADS"))
        print("DP_INTER_OP_PARALLELISM_THREADS =", os.environ.get("DP_INTER_OP_PARALLELISM_THREADS"))

        # PyTorch thread pools
        torch.set_num_threads(nthreads)
        torch.set_num_interop_threads(1)

        print("torch.get_num_threads() =", torch.get_num_threads())
        print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())

        model_path, _ = resolve_model(models_dir, self.cfg.model_key)
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
