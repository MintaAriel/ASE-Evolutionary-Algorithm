#!/usr/bin/env python3
"""DeepMD relaxation using the DeepMDClient calculator.

Runs multi-stage FIRE -> LBFGS structural relaxation. The calculator
auto-selects direct mode (fast, in deepmd_env) or worker mode
(subprocess, from ase_env).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS

from ea.backends.deepmd_client import DeepMDClient
from ea.utils.config import load_config


@dataclass
class RelaxConfig:
    model_key: str = "base_deepmd"
    fire_fmax: float = 0.10
    fire_steps: int = 500
    lbfgs_stages: tuple = (0.03, 0.01, 0.005, 0.002, 0.001)
    lbfgs_steps: int = 1200
    maxstep: float = 0.03


class DeepMDRelaxation:
    """Multi-stage structural relaxation backed by DeepMD."""

    def __init__(self, config=None, relax_config=None, device=None,
                 n_threads=None):
        """
        Parameters
        ----------
        config : dict, optional
            Project config. Loaded via ``load_config()`` if not provided.
        relax_config : RelaxConfig, optional
            Relaxation parameters. Uses defaults if not provided.
        device : str, optional
            ``"cpu"`` or ``"cuda"``. Overrides config value.
        n_threads : int, optional
            Thread count passed to the calculator.
        """
        if config is None:
            config = load_config()
        self.config = config
        self.cfg = relax_config or RelaxConfig()
        self.device = device
        self.n_threads = n_threads

    def build_calculator(self):
        """Return a ready-to-use ASE calculator."""
        return DeepMDClient(
            config=self.config,
            model_key=self.cfg.model_key,
            device=self.device,
            n_threads=self.n_threads,
        )

    def run(self, atoms, outdir):
        """Run multi-stage relaxation.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to relax. Calculator will be attached.
        outdir : str or Path
            Output directory for logs and trajectory.

        Returns
        -------
        dict
            Summary with optimizer info and final energy.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        log_path = outdir / "relax.log"
        traj_path = outdir / "opt.traj"

        atoms.calc = self.build_calculator()

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

        final_energy = atoms.get_potential_energy()
        atoms.calc.close()

        return {
            "optimizer": "fire_then_staged_lbfgs",
            "nsteps": int(fire.nsteps + lbfgs.nsteps),
            "lbfgs_stages": list(self.cfg.lbfgs_stages),
            "final_energy": final_energy,
        }
