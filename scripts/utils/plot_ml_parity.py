#!/usr/bin/env python3
"""Replot saved ML trajectories with consistent colors and axis limits.

No model inference is performed. Example:
    python plot_ml_parity.py --reference reference.traj --output-dir parity

By default, reads all *_ML.traj files in --output-dir. Use --predictions to
select trajectories and their legend order explicitly. Individual plots use
filled circular points. --exclude-combined omits labels from the combined
plot while retaining their individual plots. Saved ordering and exclusions
are reused when those options are omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ase.io import Trajectory

from create_ml_traj import error_statistics, plot_parity


def plot_style(index):
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "o", "x", "^", "D", "P", "v", "+", "*")
    color, marker = colors[index % len(colors)], markers[index % len(markers)]
    hollow = index % len(markers) in (2, 5)
    style = {"marker": marker, "alpha": 1, "zorder": 2 + index}
    if hollow:
        style.update(s=32, facecolors="none", edgecolors=color, linewidths=0.9)
    elif marker in ("x", "+"):
        style.update(s=13, color=color, linewidths=0.8)
    else:
        style.update(s=12, color=color, linewidths=0)
    handle = Line2D([], [], linestyle="none", marker=marker, color=color,
                    markerfacecolor="none" if hollow else color,
                    markeredgewidth=0.9, markersize=6)
    return style, handle


def load_predictions(reference, path):
    with Trajectory(str(path)) as trajectory:
        predicted = list(trajectory)
    if len(reference) != len(predicted):
        raise ValueError(f"Frame count mismatch: {path}")
    data = {"name": path.stem.removesuffix("_ML"), "path": path,
            "energy_ref": [], "energy_ml": [], "force_ref": [], "force_ml": []}
    for index, (a, b) in enumerate(zip(reference, predicted)):
        if not (np.array_equal(a.numbers, b.numbers)
                and np.array_equal(a.positions, b.positions)
                and np.array_equal(a.cell.array, b.cell.array)
                and np.array_equal(a.pbc, b.pbc)):
            raise ValueError(f"Frame {index}: geometry or atom order differs in {path}")
        data["energy_ref"].append(float(a.calc.results["energy"]) / len(a))
        data["energy_ml"].append(float(b.calc.results["energy"]) / len(b))
        data["force_ref"].extend(np.asarray(a.calc.results["forces"]).ravel().tolist())
        data["force_ml"].extend(np.asarray(b.calc.results["forces"]).ravel().tolist())
    for key in ("energy_ref", "energy_ml", "force_ref", "force_ml"):
        if not data[key] or not np.isfinite(data[key]).all():
            raise ValueError(f"Empty or nonfinite {key}: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--predictions", nargs="+", type=Path,
                        help="Prediction trajectories in the desired legend order.")
    parser.add_argument("--exclude-combined", nargs="*", default=None,
                        help="Labels to omit from the combined plot only; saved exclusions are reused by default.")
    args = parser.parse_args()
    manifest_path = args.output_dir / "plot_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = args.predictions if args.predictions is not None else sorted(args.output_dir.glob("*_ML.traj"))
    if args.predictions is None:
        found = {str(path.resolve()): path for path in paths}
        ordered = [row["trajectory"] for row in previous.get("models", [])
                   if row["trajectory"] in found]
        paths = [found[key] for key in ordered] + [path for path in paths if str(path.resolve()) not in ordered]
    if not paths:
        parser.error("No prediction trajectories found")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Prediction filenames must have distinct stems")
    with Trajectory(str(args.reference)) as source:
        reference = list(source)
    datasets = [load_predictions(reference, path) for path in paths]
    names = {data["name"] for data in datasets}
    if args.exclude_combined is None:
        excluded = set(previous.get("excluded_from_combined", [])) & names
    else:
        excluded = set(args.exclude_combined)
        if excluded - names:
            parser.error(f"Unknown labels in --exclude-combined: {sorted(excluded - names)}")
    combined_names = [data["name"] for data in datasets if data["name"] not in excluded]
    if not combined_names:
        parser.error("At least one model must remain in the combined plot")
    limits = []
    for key in ("energy", "force"):
        arrays = [values for data in datasets for values in (data[key + "_ref"], data[key + "_ml"])]
        lo, hi = min(min(values) for values in arrays), max(max(values) for values in arrays)
        pad = max((hi - lo) * 0.04, 1e-6)
        limits.append((lo - pad, hi + pad))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7 + 0.3 * len(combined_names)))
    handles, labels = [], [[], []]
    manifest = {"reference": str(args.reference.resolve()), "frames": len(reference),
                "limits": limits, "models": [], "opaque_markers": True,
                "excluded_from_combined": sorted(excluded), "combined_models": combined_names}
    for index, data in enumerate(datasets):
        style, handle = plot_style(index)
        included = data["name"] not in excluded
        if included:
            handles.append(handle)
        individual_style = {"marker": "o", "color": handle.get_color(), "s": 10,
                            "alpha": 1, "linewidths": 0, "zorder": 2 + index}
        plot_parity(args.output_dir / (data["path"].stem + ".png"),
                    data["energy_ref"], data["energy_ml"],
                    data["force_ref"], data["force_ml"], data["name"],
                    style=individual_style, limits=limits)
        record = {"name": data["name"], "trajectory": str(data["path"].resolve()),
                  "style": style, "individual_style": individual_style,
                  "included_in_combined": included, "metrics": {}}
        for panel, (key, scale, unit) in enumerate((("energy", 1000, "meV/atom"), ("force", 1, "eV/Å"))):
            ref, ml = data[key + "_ref"], data[key + "_ml"]
            stats = error_statistics(ref, ml, scale)
            if included:
                axes[panel].scatter(ref, ml, rasterized=True, **style)
                labels[panel].append(f"{data['name']}\nRMSE: {stats['rmse']:.4g}, MAE: {stats['mae']:.4g} {unit}")
            record["metrics"][key] = stats
        manifest["models"].append(record)

    for panel, (ax, label) in enumerate(zip(axes, ("Energy (eV/atom)", "Force component (eV/Å)"))):
        lo, hi = limits[panel]
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, zorder=1)
        ax.set(xlabel="VASP " + label, ylabel="ML " + label, xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles, labels[panel], fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), labelspacing=0.65)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.34, top=0.97, wspace=0.25)
    fig.savefig(args.output_dir / "parity_all_models.png", dpi=200,
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated {len(datasets)} individual plots with filled points; combined plot contains {len(combined_names)} models.")


if __name__ == "__main__":
    main()
