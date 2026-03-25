#!/usr/bin/env python3
from __future__ import annotations

from ase.calculators.mixing import SumCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from mattersim.forcefield import MatterSimCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

NAME = "mattersim_d3"

def build_calculator(ctx):
    model = ctx.models_dir / "tuned_mattersim_12.03.2026.pth"
    if not model.is_file():
        raise FileNotFoundError(f"MatterSim model not found: {model}")
    base_calc = MatterSimCalculator(load_path=str(model), device=ctx.device)
    d3_calc = TorchDFTD3Calculator(device=ctx.device, xc="pbe", damping="bj")
    return SumCalculator([d3_calc, base_calc])


def run_relax(atoms, ctx):
    log_path = ctx.outdir / "relax.log"
    traj_path = ctx.outdir / "opt.traj"

    opt = FIRE(
        FrechetCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
        logfile=str(log_path),
        trajectory=str(traj_path),
        maxstep=0.03,
    )
    opt.run(fmax=0.10, steps=500)
    fire_nsteps = int(opt.nsteps)

    opt = LBFGS(
        FrechetCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
        logfile=str(log_path),
        trajectory=str(traj_path),
        append_trajectory=True,
        maxstep=0.03,
        memory=40,
    )
    opt.run(fmax=0.03, steps=1200)
    opt.run(fmax=0.01, steps=1200)
    opt.run(fmax=0.005, steps=1200)
    opt.run(fmax=0.002, steps=1200)
    opt.run(fmax=0.001, steps=1200)

    total_steps = fire_nsteps + int(opt.nsteps)

    return {
        "optimizer": "fire_then_staged_lbfgs",
        "converged": None,
        "nsteps": int(total_steps),
        "lbfgs_stages": [0.03, 0.01, 0.005, 0.002, 0.001],
    }
