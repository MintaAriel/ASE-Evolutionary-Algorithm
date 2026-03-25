"""MatterSim benchmark: test relaxation performance across CPU thread counts."""

import os
import sys
import json
import time
import resource
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import shutil
import tempfile

from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter


class MatterSimTester:
    """Test MatterSim with a given trained model checkpoint and structure.

    Parameters
    ----------
    model_path : str or Path
        Path to the MatterSim checkpoint (used in-process mode only).
    device : str
        "cpu" or "cuda".
    container_root : str or Path or None
        Root of the container_cpu_2 directory.  When given, relaxation
        runs inside the Singularity/Apptainer SIF image with D3 corrections.
        When *None*, relaxation runs in-process without D3.
    input_template : str
        Filename of the input template inside ``container_root/templates/``
        (container mode only).
    n_threads : int
        Thread count forwarded to the container environment variables.
    """

    def __init__(self, model_path, device="cpu", container_root=None,
                 input_template="input_mattersim_d3.py", n_threads=4):
        self.model_path = str(model_path)
        self.device = device
        self.container_root = Path(container_root) if container_root else None
        self.input_template = input_template
        self.n_threads = n_threads

        if self.container_root is None:
            from mattersim.forcefield import MatterSimCalculator
            self.calc = MatterSimCalculator(load_path=self.model_path, device=device)
        else:
            self.calc = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def relax(self, atoms, fire_fmax=0.10, fire_steps=500,
              lbfgs_steps=1200, lbfgs_stages=None):
        """Relax a structure using FIRE followed by staged LBFGS.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to relax (a copy is made internally).
        fire_fmax : float
            Force convergence for the FIRE pre-relaxation.
        fire_steps : int
            Max FIRE steps.
        lbfgs_steps : int
            Max steps per LBFGS stage.
        lbfgs_stages : list[float] | None
            Successive fmax thresholds for LBFGS refinement.

        Returns
        -------
        dict with initial/final energies, step counts, and relaxed Atoms.
        """
        if self.container_root is not None:
            return self._relax_container(atoms)

        if lbfgs_stages is None:
            lbfgs_stages = [0.03, 0.01, 0.005, 0.002, 0.001]

        atoms = atoms.copy()
        atoms.calc = self.calc

        initial_energy = atoms.get_potential_energy()

        # FIRE stage
        fire = FIRE(
            FrechetCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
            logfile=None,
            trajectory=None,
            maxstep=0.03,
        )
        fire.run(fmax=fire_fmax, steps=fire_steps)
        fire_steps_done = int(fire.nsteps)

        # Staged LBFGS
        lbfgs = LBFGS(
            FrechetCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
            logfile=None,
            trajectory=None,
            maxstep=0.03,
            memory=40,
        )
        for stage_fmax in lbfgs_stages:
            lbfgs.run(fmax=stage_fmax, steps=lbfgs_steps)
        lbfgs_steps_done = int(lbfgs.nsteps)

        return {
            "initial_energy": initial_energy,
            "final_energy": atoms.get_potential_energy(),
            "fire_steps": fire_steps_done,
            "lbfgs_steps": lbfgs_steps_done,
            "total_steps": fire_steps_done + lbfgs_steps_done,
            "relaxed_atoms": atoms,
        }

    # ------------------------------------------------------------------
    # container-based relaxation (D3 corrections via SIF)
    # ------------------------------------------------------------------

    @staticmethod
    def _container_runtime():
        for name in ("singularity", "apptainer"):
            if shutil.which(name):
                return name
        raise RuntimeError("Neither 'singularity' nor 'apptainer' found in PATH.")

    def _relax_container(self, atoms):
        root = self.container_root
        sif = root / "containers" / "ml-relax-cpu.sif"
        if not sif.is_file():
            raise FileNotFoundError(f"SIF image not found: {sif}")

        tmpdir = Path(tempfile.mkdtemp(dir=root, prefix="relax_"))
        try:
            struct_path = tmpdir / "input.cif"
            write(str(struct_path), atoms)

            container_struct = f"/work/{tmpdir.name}/input.cif"
            container_outdir = f"/work/{tmpdir.name}/out"

            runtime = self._container_runtime()
            n = str(self.n_threads)

            env = os.environ.copy()
            thread_vars = [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "DP_INTRA_OP_PARALLELISM_THREADS",
            ]
            for var in thread_vars:
                env[var] = n
                env[f"SINGULARITYENV_{var}"] = n
            env["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
            env["SINGULARITYENV_DP_INTER_OP_PARALLELISM_THREADS"] = "1"

            cmd = [
                runtime, "exec",
                "--bind", f"{root}:/work",
                str(sif),
                "python", "/work/runner/ml_relax_cli.py",
                "--input-script", f"/work/templates/{self.input_template}",
                "--structure", container_struct,
                "--models-dir", "/work/models",
                "--device", self.device,
                "--outdir", container_outdir,
            ]

            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

            result_json_path = tmpdir / "out" / "result.json"
            if not result_json_path.is_file():
                raise RuntimeError(
                    f"Container produced no result.json (rc={proc.returncode}).\n"
                    f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
                )

            result_data = json.loads(result_json_path.read_text())

            if result_data["status"] != "ok":
                raise RuntimeError(
                    f"Container relaxation failed: {result_data.get('error')}"
                )

            relaxed_atoms = read(str(tmpdir / "out" / "final.cif"))
            relax_info = result_data.get("relax_info", {})

            return {
                "initial_energy": result_data["energy_initial_eV"],
                "final_energy": result_data["energy_final_eV"],
                "fire_steps": None,
                "lbfgs_steps": None,
                "total_steps": relax_info.get("nsteps", 0),
                "relaxed_atoms": relaxed_atoms,
                "fmax_final": result_data.get("fmax_final_eV_A"),
                "runtime_s": result_data.get("runtime_s"),
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def _run_worker(model_path, structure_path, n_threads, device):
    """Launch a subprocess that benchmarks a single thread count."""
    worker_script = str(Path(__file__).parent / "_worker.py")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["NUMEXPR_NUM_THREADS"] = str(n_threads)

    # Ensure the src/ dir is on PYTHONPATH so the worker can import this package
    src_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, worker_script,
         "--model", str(model_path),
         "--structure", str(structure_path),
         "--n-threads", str(n_threads),
         "--device", device],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Worker failed (threads={n_threads}):\n{result.stderr}"
        )

    # Find the marked JSON line — libraries may print other things to stdout
    marker = "__WORKER_JSON__"
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])

    raise RuntimeError(
        f"Worker produced no JSON output (threads={n_threads}).\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


def run_all_benchmarks(model_path, structure_path,
                       cores_range=range(1, 16),
                       device="cpu", output_dir=None):
    """Run relaxation benchmarks across *cores_range* thread counts.

    Saves a CSV and plots (cores-vs-time, cores-vs-RAM) to *output_dir*.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "test" / "matersim_benchmark"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n_cores in cores_range:
        print(f"Benchmarking with {n_cores} thread(s)...")
        try:
            result = _run_worker(model_path, structure_path, n_cores, device)
            results.append(result)
            print(f"  Time: {result['elapsed_time']:.2f}s | "
                  f"Peak RSS: {result['peak_rss_mb']:.1f} MB")
        except RuntimeError as e:
            print(f"  FAILED: {e}")

    if not results:
        print("No successful runs — nothing to save.")
        return None

    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to {json_path}")

    _plot_results(df, output_dir)
    return df


def _plot_results(df, output_dir):
    """Generate and save cores-vs-time and cores-vs-RAM plots."""
    output_dir = Path(output_dir)

    # --- Combined figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["n_threads"], df["elapsed_time"], "o-",
             color="tab:blue", linewidth=2, markersize=6)
    ax1.set_xlabel("Number of CPU Threads", fontsize=12)
    ax1.set_ylabel("Elapsed Time (s)", fontsize=12)
    ax1.set_title("MatterSim Relaxation: Cores vs Time", fontsize=13)
    ax1.set_xticks(df["n_threads"])
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["n_threads"], df["peak_rss_mb"], "s-",
             color="tab:red", linewidth=2, markersize=6)
    ax2.set_xlabel("Number of CPU Threads", fontsize=12)
    ax2.set_ylabel("Peak RSS (MB)", fontsize=12)
    ax2.set_title("MatterSim Relaxation: Cores vs RAM", fontsize=13)
    ax2.set_xticks(df["n_threads"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "benchmark_combined.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Individual plots ---
    for col, ylabel, title, fname, color, marker in [
        ("elapsed_time", "Elapsed Time (s)", "Cores vs Time",
         "cores_vs_time.png", "tab:blue", "o"),
        ("peak_rss_mb", "Peak RSS (MB)", "Cores vs RAM",
         "cores_vs_ram.png", "tab:red", "s"),
    ]:
        fig2, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["n_threads"], df[col], f"{marker}-",
                color=color, linewidth=2, markersize=6)
        ax.set_xlabel("Number of CPU Threads", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"MatterSim Relaxation: {title}", fontsize=13)
        ax.set_xticks(df["n_threads"])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"Plots saved to {output_dir}")
