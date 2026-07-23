"""Batched USPEX worker.

Drop-in replacement for the per-CalcFolder ``uspex_deepmd_gfnff.py``
workflow.  Where the gfnff script processes one CalcFolder per process,
this script processes ALL CalcFolders of the current USPEX generation in
a single batched DeepMD evaluation:

    1. discover ``CalcFold[N]`` directories under the USPEX workdir,
    2. read each USPEX 26 ASE ``input.xyz`` (preferred) or legacy
       USER_CODE ``geom.in`` file,
    3. run staged ParallelFIRE + ParallelLBFGS just like
       ``create_batch_deepmd.run_full_optimization``,
    4. (optional) compute Gamma-point ZPE per structure with
       ``ea.parallel.zpe.ParallelVibrations``,
    5. write ``output.xyz`` with energy metadata (ASE/code 20) or legacy
       ``geom.out`` + ``energy.txt`` (USER_CODE/code 99) back into each
       CalcFolder.  The completion marker is published last/atomically.

USPEX 26 molecular calculations must use the ASE/code-20 interface.  Its
extended-XYZ files preserve molecule/template atom order.  The code-99 POSCAR
writer groups all atoms globally by element, after which USPEX cannot rebuild
the original molecular components even though it can still read energy.txt.

Driven by ``run_uspex26.py`` (this package); not meant to be run
N times in parallel — exactly once per USPEX 26 relaxation wave.
"""

import argparse
import os
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

# Legacy USPEX: 'CalcFold<N>' directly under the workdir.
_CALC_RE = re.compile(r"CalcFold(\d+)$")
# USPEX 26: 'Calcfold_<system>_<step>' under '<workdir>/Calculation'
# (base_calc_folder = 'Calculation' in the USPEX 26 defaults).
_CALC26_RE = re.compile(r"Calcfold_(\d+)_(.+)$")
# Directories that may hold USPEX 26 Calcfold_* folders.
_CALC26_BASES = ("Calculation", "CalculationTemp")


ASE_MODE = "ase"
USER_CODE_MODE = "user_code"


def calcfolder_mode(cf):
    """Return the external-relaxation interface used by *cf*, if ready.

    USPEX code 20 writes ``input.xyz`` and consumes ``output.xyz``.  Legacy
    code 99 writes ``geom.in`` and consumes ``geom.out`` + ``energy.txt``.
    Prefer ASE if stale files from both interfaces happen to coexist.
    """
    cf = Path(cf)
    ase_input = cf / "input.xyz"
    if ase_input.is_file() and ase_input.stat().st_size > 0:
        return ASE_MODE

    user_input = cf / "geom.in"
    if user_input.is_file() and user_input.stat().st_size > 0:
        return USER_CODE_MODE

    return None


def _pending(cf):
    """Whether a calc folder has a complete input but no completion marker."""
    mode = calcfolder_mode(cf)
    if mode == ASE_MODE:
        return not (cf / "output.xyz").is_file()
    if mode == USER_CODE_MODE:
        return not (cf / "energy.txt").is_file()
    return False


def discover_calcfolders(workdir):
    """Return pending ``(sort_key, path)`` calculation folders.

    Supports both the legacy layout (``CalcFold<N>`` under the workdir) and
    USPEX 26 (``Calculation/Calcfold_<system>_<step>``), as well as the ASE
    and legacy USER_CODE file protocols.
    """
    workdir = Path(workdir)
    out = []

    # Legacy 'CalcFold<N>' directly under the workdir.
    for cf in workdir.iterdir():
        if cf.is_dir():
            m = _CALC_RE.match(cf.name)
            if m and _pending(cf):
                out.append(((0, int(m.group(1)), ""), cf))

    # USPEX 26 'Calcfold_<system>_<step>' under 'Calculation'.
    for base in _CALC26_BASES:
        base_dir = workdir / base
        if not base_dir.is_dir():
            continue
        for cf in base_dir.iterdir():
            if cf.is_dir():
                m = _CALC26_RE.match(cf.name)
                if m and _pending(cf):
                    out.append(((1, int(m.group(1)), m.group(2)), cf))

    out.sort(key=lambda t: t[0])
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


def run_full_optimization(
    batch,
    calc,
    out_dir,
    *,
    fire_steps=FIRE_STEPS,
    lbfgs_steps=LBFGS_STEPS,
    lbfgs_stages=LBFGS_STAGES,
):
    evaluator = make_evaluator(calc)

    print(f"\n=== ParallelFIRE  fmax={FIRE_FMAX}  on {len(batch)} structures ===")
    opt = ParallelFIRE(batch, batch_evaluator=evaluator,
                       fmax=FIRE_FMAX, max_steps=fire_steps, maxstep=MAXSTEP)
    opt.run()
    batch = opt.get_atoms()

    for fmax in lbfgs_stages:
        print(f"\n=== ParallelLBFGS  fmax={fmax} ===")
        opt = ParallelLBFGS(batch, batch_evaluator=evaluator,
                            fmax=fmax, max_steps=lbfgs_steps,
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

def _write_ase_result(cf_path, atoms, energy):
    """Atomically publish an order-preserving USPEX ASE ``output.xyz``."""
    output_path = cf_path / "output.xyz"
    temporary_path = cf_path / ".output.xyz.tmp"
    output_atoms = atoms.copy()
    output_atoms.info["energy"] = float(energy)
    write(temporary_path, output_atoms, format="extxyz")
    os.replace(temporary_path, output_path)


def write_calcfolder_result(cf_path, atoms, energy, mode):
    """Write a result using the interface with which USPEX made the input."""
    cf_path = Path(cf_path)
    if mode == ASE_MODE:
        _write_ase_result(cf_path, atoms, energy)
        return
    if mode == USER_CODE_MODE:
        # energy.txt remains the legacy completion marker and is written last.
        write(cf_path / "geom.out", atoms, format="vasp", direct=True)
        (cf_path / "energy.txt").write_text(f"{energy}\n")
        return
    raise ValueError(f"Unknown USPEX calc-folder mode: {mode!r}")


def write_failure(cf_path, atoms, mode):
    """Fallback: keep input structure, energy = 0 (matches uspex_deepmd_gfnff)."""
    write_calcfolder_result(cf_path, atoms, 0.0, mode)


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
    p.add_argument("--smoke", action="store_true",
                   help="Run a minimal FIRE/LBFGS sequence for integration "
                        "smoke tests; do not use for production relaxation")
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
    indices, paths, modes, atoms_in = [], [], [], []
    for idx, cf in calcfolders:
        mode = calcfolder_mode(cf)
        try:
            if mode == ASE_MODE:
                a = read(cf / "input.xyz", format="extxyz")
            elif mode == USER_CODE_MODE:
                a = read(cf / "geom.in", format="vasp")
            else:
                raise ValueError("no supported non-empty input file")
        except Exception as e:
            print(f"  [warn] {cf.name}: failed to read relaxation input: {e}")
            continue
        indices.append(idx)
        paths.append(cf)
        modes.append(mode)
        atoms_in.append(a)

    if not atoms_in:
        sys.exit("[batch_worker] no readable pending structures")

    mode_counts = {mode: modes.count(mode) for mode in sorted(set(modes))}
    print(f"[batch_worker] interfaces: {mode_counts}")
    if USER_CODE_MODE in mode_counts and (workdir / "MOL_1").is_file():
        sys.exit(
            "[batch_worker] molecular MOL_* input detected with legacy USPEX "
            "USER_CODE/code 99. Use abinitioCode 20 so extended XYZ preserves "
            "molecular atom order."
        )

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

    optimization_kwargs = {}
    if args.smoke:
        optimization_kwargs = {
            "fire_steps": 2,
            "lbfgs_steps": 2,
            "lbfgs_stages": (0.03,),
        }
        print("[batch_worker] SMOKE profile: abbreviated relaxation")
    relaxed, energies = run_full_optimization(
        batch, calc, out_dir, **optimization_kwargs
    )

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
                write_calcfolder_result(
                    cf, final_atoms[i], final_energies[i], modes[i]
                )
                n_ok += 1
            else:
                write_failure(cf, atoms_in[i], modes[i])
                n_fail += 1
        except Exception:
            print(f"[batch_worker] failed to write {cf.name}:", traceback.format_exc())
            try:
                write_failure(cf, atoms_in[i], modes[i])
            except Exception:
                pass
            n_fail += 1

    print(f"[batch_worker] done — relaxed={n_ok}  fallback={n_fail}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[batch_worker] total wall time: {time.perf_counter() - t0:.1f}s")
