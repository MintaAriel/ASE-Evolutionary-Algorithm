#!/usr/bin/env python3
"""Evaluate DeePMD models on a trajectory without changing frames or atom order.

Example (run inside an environment containing ASE and DeePMD):
    python create_ml_traj.py --input reference.traj --model model.pth \
        --output-dir parity --device cuda --gpu GPU-... --plot

Repeat --model for additional models; optional --label values set their output
and plot names in the same order. If DeePMD is installed in another
environment, pass --worker-python /path/to/that/environment/bin/python.
Outputs include vasp.traj (an exact reference copy), MODEL_ML.traj, energy
tables, run metadata/metrics, and optional energy/force parity plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Input ASE trajectory.")
    parser.add_argument("--model", required=True, action="append", type=Path,
                        help="DeePMD model path; repeat for multiple models.")
    parser.add_argument("--label", action="append",
                        help="Output/plot label; repeat in model order. Defaults to model stems.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", help="CUDA GPU UUID or index; UUID avoids device-order ambiguity.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--worker-python", type=Path,
                        help="Run inference with this Python interpreter (must have ASE/DeePMD).")
    parser.add_argument("--plot", action="store_true", help="Save energy/force parity PNGs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing model outputs.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.gpu is not None and args.device != "cuda":
        parser.error("--gpu requires --device cuda")
    args.input = args.input.expanduser().resolve(strict=True)
    args.model = [p.expanduser().resolve(strict=True) for p in args.model]
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.label is not None and len(args.label) != len(args.model):
        parser.error("Provide one --label per --model, or omit --label")
    args.label = args.label if args.label is not None else [p.stem for p in args.model]
    if any(label in {"", ".", ".."} or any(c in label for c in "/\\\x00\n\r")
           for label in args.label):
        parser.error("Labels must be nonempty filename components without path separators")
    if len(set(args.label)) != len(args.label):
        parser.error("Model labels must be distinct to avoid output collisions")
    return args


def configure_runtime(args):
    # DeePMD reads these variables when its backend is first imported.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(args.threads)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
    os.environ["DEVICE"] = args.device
    os.environ["LOCAL_RANK"] = "0"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def run_worker(args):
    tail = []
    original = iter(sys.argv[1:])
    for value in original:
        if value == "--worker-python":
            next(original)
        elif not value.startswith("--worker-python="):
            tail.append(value)
    env = os.environ.copy()
    env["ML_TRAJ_LAUNCHER_PYTHON"] = sys.executable
    env["ML_TRAJ_LAUNCH_COMMAND"] = json.dumps([sys.executable, *sys.argv])
    return subprocess.call([str(args.worker_python.expanduser().resolve(strict=True)),
                            str(Path(__file__).resolve()), *tail], env=env)


def error_statistics(reference, predicted, scale=1.0):
    import numpy as np

    diff = (np.asarray(predicted) - np.asarray(reference)) * scale
    return {"mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))), "count": len(diff)}


def plot_parity(path, energy_ref, energy_ml, force_ref, force_ml, title,
                *, style=None, limits=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if energy_ref:
        panels.append((energy_ref, energy_ml, "Energy (eV/atom)", 1000, "meV/atom"))
    if force_ref:
        panels.append((force_ref, force_ml, "Force component (eV/Å)", 1, "eV/Å"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for panel_index, (ax, (reference, predicted, label, scale, unit)) in enumerate(zip(axes[0], panels)):
        lo, hi = min(min(reference), min(predicted)), max(max(reference), max(predicted))
        pad = max((hi - lo) * 0.04, 1e-6)
        lo, hi = limits[panel_index] if limits is not None else (lo - pad, hi + pad)
        ax.scatter(reference, predicted, rasterized=True,
                   **(style or {"s": 10, "color": "tab:blue", "alpha": 1}))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, zorder=1)
        ax.set(xlabel=f"VASP {label}", ylabel=f"ML {label}",
               xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal", adjustable="box")
        stats = error_statistics(reference, predicted, scale)
        ax.text(0.04, 0.96,
                f"RMSE = {stats['rmse']:.4g} {unit}\nMAE = {stats['mae']:.4g} {unit}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                      "edgecolor": "0.75", "alpha": 0.95})
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(args):
    import numpy as np
    import torch
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import Trajectory
    from ase.stress import full_3x3_to_voigt_6_stress
    from deepmd.infer import DeepPot

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
    gpu = {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "name": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    print(f"Interpreter: {sys.executable}; device: {args.device}; GPU: {gpu}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.input)
    reference_copy = args.output_dir / "vasp.traj"
    if reference_copy.exists():
        if sha256(reference_copy) != source_hash:
            raise FileExistsError(f"{reference_copy} contains a different reference dataset")
    else:
        shutil.copy2(args.input, reference_copy)

    for model, label in zip(args.model, args.label):
        prefix = args.output_dir / f"{label}_ML"
        paths = {key: Path(str(prefix) + ext) for key, ext in
                 [("traj", ".traj"), ("csv", ".csv"), ("json", ".json"), ("plot", ".png")]}
        for key, path in paths.items():
            if path == args.input or path in args.model:
                raise ValueError(f"Output would overwrite an input: {path}")
            if (key != "plot" or args.plot) and path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; choose another directory or use --overwrite")

        started = time.monotonic()
        model_hash = sha256(model)
        potential = DeepPot(str(model), auto_batch_size=False)
        type_map = potential.get_type_map()
        atom_types = {symbol: index for index, symbol in enumerate(type_map)}
        rows, energy_ref, energy_ml, force_ref, force_ml = [], [], [], [], []
        partial = Path(str(paths["traj"]) + ".partial")
        with Trajectory(str(args.input), "r") as source, Trajectory(str(partial), "w") as target:
            if not len(source):
                raise ValueError("Input trajectory is empty")
            for index, reference in enumerate(source):
                atoms = reference.copy()
                if atoms.pbc.any() and not atoms.pbc.all():
                    raise ValueError(f"Frame {index}: mixed periodic/nonperiodic axes are unsupported")
                types = [atom_types[symbol] for symbol in atoms.get_chemical_symbols()]
                cell = atoms.cell.array.reshape(1, 9) if atoms.pbc.all() else None
                energy, forces, virial = potential.eval(atoms.positions.reshape(1, -1), cell, types)
                energy = float(np.asarray(energy).reshape(-1)[0])
                forces = np.asarray(forces).reshape(len(atoms), 3)
                results = {"energy": energy, "free_energy": energy, "forces": forces}
                if atoms.pbc.all():
                    virial = np.asarray(virial).reshape(3, 3)
                    stress = -(virial + virial.T) / (2 * atoms.get_volume())
                    results["stress"] = full_3x3_to_voigt_6_stress(stress)
                if not all(np.isfinite(value).all() for value in results.values()):
                    raise ValueError(f"Frame {index}: model returned nonfinite values")
                atoms.info.update(ml_model=str(model), ml_model_sha256=model_hash,
                                  ml_label=label, source_frame=index)
                atoms.calc = SinglePointCalculator(atoms, **results)
                atoms.calc.name = "deepmd"
                target.write(atoms)

                reference_results = reference.calc.results if reference.calc is not None else {}
                ref_energy = reference_results.get("energy")
                error = None if ref_energy is None else energy - float(ref_energy)
                if error is not None:
                    energy_ref.append(float(ref_energy) / len(atoms))
                    energy_ml.append(energy / len(atoms))
                rows.append({"frame": index, "natoms": len(atoms), "energy_vasp_eV": ref_energy,
                             "energy_ml_eV": energy, "error_eV": error,
                             "error_meV_per_atom": None if error is None else 1000 * error / len(atoms)})
                if "forces" in reference_results:
                    force_ref.extend(np.asarray(reference_results["forces"]).ravel().tolist())
                    force_ml.extend(forces.ravel().tolist())
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(source):
                    print(f"{label}: {index + 1}/{len(source)} frames", flush=True)
        partial.replace(paths["traj"])

        metrics = {}
        for name, ref, predicted in [("energy_meV_per_atom", energy_ref, energy_ml),
                                      ("force_eV_per_A", force_ref, force_ml)]:
            if ref:
                metrics[name] = error_statistics(ref, predicted,
                                                 1000 if name.startswith("energy") else 1)
        with paths["csv"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {"model": str(model), "model_sha256": model_hash, "label": label,
                    "input": str(args.input), "input_sha256": source_hash,
                    "reference_copy": str(reference_copy), "output": str(paths["traj"]),
                    "frames": len(rows), "type_map": type_map, "device": args.device, "gpu": gpu,
                    "python": sys.executable, "launcher_python": os.environ.get("ML_TRAJ_LAUNCHER_PYTHON"),
                    "command": [sys.executable, *sys.argv],
                    "launcher_command": json.loads(os.environ.get("ML_TRAJ_LAUNCH_COMMAND", "null")),
                    "versions": {name: version(name) for name in ("ase", "torch", "deepmd-kit", "numpy")},
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "seconds": time.monotonic() - started, "metrics": metrics}
        paths["json"].write_text(json.dumps(metadata, indent=2) + "\n")
        if args.plot:
            plot_parity(paths["plot"], energy_ref, energy_ml, force_ref, force_ml, label)
        print(json.dumps({"output": str(paths["traj"]), "metrics": metrics}), flush=True)
        del potential
        if args.device == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.worker_python is not None:
        return run_worker(args)
    configure_runtime(args)
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
