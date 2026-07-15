"""Batched USPEX worker.

Drop-in replacement for the per-CalcFolder ``uspex_deepmd_gfnff.py``
workflow.  Where the gfnff script processes one CalcFolder per process,
this script processes ALL CalcFolders of the current USPEX generation in
a single batched DeepMD evaluation:

    1. discover ``CalcFold[N]`` directories under the USPEX workdir,
    2. read each ``geom.in`` (VASP / POSCAR format),
    3. run staged ParallelFIRE + ParallelLBFGS just like
       ``create_batch_deepmd.run_full_optimization``,
    4. (optional) compute Gamma-point ZPE per structure with
       ``ea.parallel.zpe.ParallelVibrations``,
    5. write ``geom.out`` (VASP, direct) and ``energy.txt`` back into
       each CalcFolder — energy.txt is written LAST so the
       per-CalcFolder run-stub treats it as the completion marker.

Driven by ``scripts/run/parallel_run_uspex.py``; not meant to be run
N times in parallel — exactly once per USPEX generation.
"""

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

from ase.io import read, write

from ea.parallel.create_batch import batch_calculator_deepmd
from ea.parallel.FIRE_parallel import ParallelFIRE
from ea.parallel.LBFGS_parallel import ParallelLBFGS
from ea.parallel.zpe import ParallelVibrations
from ea.utils.config import load_config


MODELS = {
    'base_deepmd': 'dpa3_12.03.2026.pth',
    'deepmd_d3': 'dpa3-d3_torch.pth',
    'deepmd_d4': 'dpa3-d4.pth',
    'deepmd_d3_abs': 'dpa3-d3_abs_torch.pth',
    'deepmd_d3_mbj': 'dpa3-d3-cpu_mbj.pth',
    'deepmd_d3_mbj_abs': 'dpa3-d3-cpu_mbj_abs.pth',
}

LBFGS_STAGES = (0.03, 0.01, 0.005, 0.002, 0.001)
FIRE_FMAX = 0.10
FIRE_STEPS = 500
LBFGS_STEPS = 1200
MAXSTEP = 0.03
LBFGS_MEMORY = 40


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

def _resolve_model(models_dir, key):
    path = Path(models_dir) / MODELS[key]
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


def make_calculator(model_key, device):
    from deepmd.calculator import DP
    cfg = load_config()
    model_path = _resolve_model(cfg['deepmd']['models_path'], model_key)
    print(f"[batch_worker] loading {model_key} from {model_path} on {device}")
    return DP(model=str(model_path), device=device)


def make_evaluator(calculator):
    def _eval(atoms_list):
        return batch_calculator_deepmd(atoms_list, calculator)
    return _eval


# ---------------------------------------------------------------------------
# CalcFolder discovery
# ---------------------------------------------------------------------------

_CALC_RE = re.compile(r"CalcFold(\d+)$")


def discover_calcfolders(workdir):
    """Return [(idx, path)] for CalcFold* with a geom.in but no energy.txt."""
    out = []
    for cf in Path(workdir).iterdir():
        if not cf.is_dir():
            continue
        m = _CALC_RE.match(cf.name)
        if not m:
            continue
        if not (cf / "geom.in").is_file():
            continue
        if (cf / "energy.txt").is_file():
            continue
        out.append((int(m.group(1)), cf))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Optimization (mirrors create_batch_deepmd.run_full_optimization)
# ---------------------------------------------------------------------------

def _reset_traj_files(workdir, out_dir):
    """Drop ``batch.traj`` and ``<out_dir>/output.traj`` from any previous
    generation BEFORE the optimizers start writing.  If this generation is
    interrupted, the partial trajectory written by ParallelFIRE/LBFGS is
    preserved for inspection."""
    for f in (Path(workdir) / "batch.traj", Path(out_dir) / "output.traj"):
        if f.is_file():
            f.unlink()
            print(f"[batch_worker] removed stale trajectory: {f}")


def run_full_optimization(batch, calc, out_dir):
    evaluator = make_evaluator(calc)

    print(f"\n=== ParallelFIRE  fmax={FIRE_FMAX}  on {len(batch)} structures ===")
    opt = ParallelFIRE(batch, batch_evaluator=evaluator,
                       fmax=FIRE_FMAX, max_steps=FIRE_STEPS, maxstep=MAXSTEP)
    opt.run()
    batch = opt.get_atoms()

    for fmax in LBFGS_STAGES:
        print(f"\n=== ParallelLBFGS  fmax={fmax} ===")
        opt = ParallelLBFGS(batch, batch_evaluator=evaluator,
                            fmax=fmax, max_steps=LBFGS_STEPS,
                            maxstep=MAXSTEP, memory=LBFGS_MEMORY)
        opt.run(out_dir=out_dir)
        batch = opt.get_atoms()

    print("\n=== final energies ===")
    for i, s in enumerate(opt.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")

    return batch, [s.energy for s in opt.states]


# ---------------------------------------------------------------------------
# ZPE
# ---------------------------------------------------------------------------

def compute_zpe(batch, calc):
    print(f"\n=== ParallelVibrations on {len(batch)} structures ===")
    vib = ParallelVibrations(batch, batch_evaluator=make_evaluator(calc))
    vib.run()
    vib.read()
    return vib.get_zero_point_energies()


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_calcfolder_result(cf_path, atoms, energy):
    """Write geom.out then energy.txt (energy.txt is the completion marker)."""
    write(cf_path / "geom.out", atoms, format="vasp", direct=True)
    (cf_path / "energy.txt").write_text(f"{energy}\n")


def write_failure(cf_path, atoms):
    """Fallback: keep input structure, energy = 0 (matches uspex_deepmd_gfnff)."""
    write(cf_path / "geom.out", atoms, format="vasp", direct=True)
    (cf_path / "energy.txt").write_text("0\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Batched DeepMD relaxation across all CalcFolders of "
                    "the current USPEX generation."
    )
    p.add_argument("workdir", type=Path,
                   help="USPEX working directory containing CalcFold[N]")
    p.add_argument("--size", type=int, default=None,
                   help="Cap batch size (default: all CalcFolders)")
    p.add_argument("--model", default="deepmd_d3",
                   choices=sorted(MODELS),
                   help="DeepMD model key from sirena.yaml deepmd.models_path")
    p.add_argument("--device", default=None,
                   help="DeepMD device override (default: deepmd.device from config)")
    p.add_argument("--zpe", action="store_true",
                   help="Add Gamma-point ZPE to the final energy "
                        "(matches uspex_deepmd_gfnff.py)")
    p.add_argument("--traj-name", default="batch.traj",
                   help="Filename for the assembled input trajectory "
                        "(written under workdir)")
    args = p.parse_args()

    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        sys.exit(f"workdir not found: {workdir}")

    calcfolders = discover_calcfolders(workdir)
    if not calcfolders:
        print(f"[batch_worker] no pending CalcFolders under {workdir}; nothing to do")
        return
    print(f"[batch_worker] {len(calcfolders)} pending CalcFolders")

    # ---- read inputs -------------------------------------------------------
    indices, paths, atoms_in = [], [], []
    for idx, cf in calcfolders:
        try:
            a = read(cf / "geom.in", format="vasp")
        except Exception as e:
            print(f"  [warn] CalcFold{idx}: failed to read geom.in: {e}")
            continue
        indices.append(idx)
        paths.append(cf)
        atoms_in.append(a)

    # ---- relax -------------------------------------------------------------
    out_dir = workdir / "batch_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe last generation's trajectories now (not at the end), so an
    # interrupted run still leaves the optimizer's partial output.traj on disk.
    _reset_traj_files(workdir, out_dir)

    # Snapshot the as-loaded batch as a single trajectory (debug aid).
    traj_path = workdir / args.traj_name
    write(traj_path, atoms_in)
    print(f"[batch_worker] wrote input batch trajectory: {traj_path}")

    cfg = load_config()
    device = args.device or cfg["deepmd"].get("device", "cpu")
    calc = make_calculator(args.model, device)

    keep_idx = list(range(len(atoms_in)))
    if args.size is not None:
        keep_idx = keep_idx[: args.size]
    batch = [atoms_in[i].copy() for i in keep_idx]

    relaxed, energies = run_full_optimization(batch, calc, out_dir)

    # ---- ZPE ---------------------------------------------------------------
    if args.zpe:
        try:
            zpes = compute_zpe(relaxed, calc)
        except Exception:
            print("[batch_worker] ZPE failed:", traceback.format_exc())
            zpes = [0.0] * len(relaxed)
    else:
        zpes = [0.0] * len(relaxed)

    # ---- distribute results back to CalcFolders ---------------------------
    final_energies = {keep_idx[k]: energies[k] + zpes[k] for k in range(len(relaxed))}
    final_atoms = {keep_idx[k]: relaxed[k] for k in range(len(relaxed))}

    n_ok = n_fail = 0
    for i, cf in enumerate(paths):
        try:
            if i in final_energies:
                write_calcfolder_result(cf, final_atoms[i], final_energies[i])
                n_ok += 1
            else:
                write_failure(cf, atoms_in[i])
                n_fail += 1
        except Exception:
            print(f"[batch_worker] failed to write {cf.name}:", traceback.format_exc())
            try:
                write_failure(cf, atoms_in[i])
            except Exception:
                pass
            n_fail += 1

    print(f"[batch_worker] done — relaxed={n_ok}  fallback={n_fail}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[batch_worker] total wall time: {time.perf_counter() - t0:.1f}s")
