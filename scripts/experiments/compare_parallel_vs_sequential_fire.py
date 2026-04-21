"""Compare Parallel{FIRE,LBFGS} vs sequential ASE {FIRE,LBFGS}(FrechetCellFilter)
on the same 4 structures. The algorithms are identical per structure, so step
counts and final energies should match to within numerical noise.
"""

import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")

import numpy as np
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter

from create_batch_deepmd import (
    ParallelFIRE,
    calc,
    new_batch,
)
from ea.parallel.LBFGS_parallel import ParallelLBFGS


def _run_sequential(optimizer_cls, opt_kwargs, atoms_list, fmax, max_steps):
    """Run ASE optimizer_cls(FrechetCellFilter(atoms)) one structure at a time."""
    results = []
    for i, at_init in enumerate(atoms_list):
        at = at_init.copy()
        at.calc = calc
        flt = FrechetCellFilter(at)
        dyn = optimizer_cls(flt, logfile=None, **opt_kwargs)
        dyn.run(fmax=fmax, steps=max_steps)
        fmax_final = float(np.linalg.norm(flt.get_forces(), axis=1).max())
        converged = fmax_final < fmax
        e_final = at.get_potential_energy()
        print(f"  struct {i}: steps = {dyn.nsteps:3d}  "
              f"E = {e_final:.4f}  fmax = {fmax_final:.4f}  "
              f"vol = {at.get_volume():.2f}  "
              f"[{'CONVERGED' if converged else 'not converged'}]")
        results.append((dyn.nsteps, e_final, fmax_final,
                        at.get_volume(), converged))
    return results


def _collect_parallel(opt):
    return [(s.energy, s.fmax_current, s.atoms.get_volume(), s.converged)
            for s in opt.states]


def _print_comparison(label, par_results, seq_results, par_nsteps):
    print(f"\n=== {label}: comparison ===")
    print(f"  Parallel{label} steps (max across batch): {par_nsteps}")
    print(f"  Sequential steps per struct:              "
          f"{[r[0] for r in seq_results]}")
    print("\n  per-structure delta (parallel - sequential):")
    for i in range(len(par_results)):
        pe, pf, pv, _ = par_results[i]
        sn, se, sf, sv, _ = seq_results[i]
        print(f"    struct {i}: dE = {pe - se:+.6f} eV   "
              f"dfmax = {pf - sf:+.6f} eV/A   "
              f"dvol = {pv - sv:+.4f} A^3")


def main():
    fmax = 0.05
    max_steps = 300

    # =====================================================================
    # FIRE
    # =====================================================================
    print("=" * 70)
    print("FIRE")
    print("=" * 70)

    print("\n--- ParallelFIRE (batched, positions + cell) ---")
    test_batch = [c.copy() for c in new_batch[:2]]
    opt_fire = ParallelFIRE(test_batch, calc, fmax=fmax, max_steps=max_steps)
    opt_fire.run()
    par_fire = _collect_parallel(opt_fire)

    print("\n--- sequential FIRE(FrechetCellFilter) ---")
    seq_fire = _run_sequential(
        FIRE, dict(dt=0.1, maxstep=0.03),
        new_batch[:2], fmax=fmax, max_steps=max_steps,
    )

    _print_comparison("FIRE", par_fire, seq_fire, opt_fire.nsteps_done)

    # =====================================================================
    # LBFGS
    # =====================================================================
    print("\n" + "=" * 70)
    print("LBFGS")
    print("=" * 70)
    #
    # print("\n--- ParallelLBFGS (batched, positions + cell) ---")
    # test_batch = [c.copy() for c in new_batch[:2]]
    # opt_lbfgs = ParallelLBFGS(test_batch, calc, fmax=fmax, max_steps=max_steps)
    # opt_lbfgs.run()
    # par_lbfgs = _collect_parallel(opt_lbfgs)
    #
    # print("\n--- sequential LBFGS(FrechetCellFilter) ---")
    # seq_lbfgs = _run_sequential(
    #     LBFGS, dict(maxstep=0.2, memory=100, damping=1.0, alpha=70.0),
    #     new_batch[:2], fmax=fmax, max_steps=max_steps,
    # )
    #
    # _print_comparison("LBFGS", par_lbfgs, seq_lbfgs, opt_lbfgs.nsteps_done)


if __name__ == "__main__":
    main()
