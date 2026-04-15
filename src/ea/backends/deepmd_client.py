"""ASE calculator backed by DeepMD — direct or worker mode.

* **direct** — imports ``deepmd.calculator.DP`` in-process (fast, requires
  the ``deepmd`` package in the current environment).
* **worker** — spawns a long-lived subprocess inside ``deepmd_env`` and
  communicates via JSON over stdin/stdout (no deepmd dependency needed).

Mode is auto-detected unless explicitly set.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ea.utils.config import load_config

WORKER_SCRIPT = Path(__file__).resolve().parent / "deepmd_worker.py"

MODELS = {
    "base_deepmd": "dpa3_12.03.2026.pth",
    "deepmd_d3": "dpa3-d3_torch.pth",
    "deepmd_d4": "dpa3-d4.pth",
    "deepmd_d3_abs": "dpa3-d3_abs_torch.pth",
    "deepmd_d3_mbj": "dpa3-d3-cpu_mbj.pth",
    "deepmd_d3_mbj_abs": "dpa3-d3-cpu_mbj_abs.pth",
}


def resolve_model(models_dir: str | Path, key: str) -> Path:
    """Map a model key to its file path inside *models_dir*."""
    if key not in MODELS:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(MODELS)}")
    path = Path(models_dir) / MODELS[key]
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


class DeepMDClient(Calculator):
    """ASE calculator backed by a DeepMD potential.

    Parameters
    ----------
    config : dict, optional
        Project config. Loaded via ``load_config()`` if not provided.
    model_key : str, optional
        Key into MODELS dict. Overrides ``config["deepmd"]["model_key"]``.
    device : str, optional
        ``"cpu"`` or ``"cuda"``. Overrides ``config["deepmd"]["device"]``.
    n_threads : int, optional
        Number of threads for torch (direct mode). Defaults to
        ``OMP_NUM_THREADS`` or 1.
    mode : {"direct", "worker"}, optional
        ``"direct"`` uses in-process DP (fast, needs deepmd installed).
        ``"worker"`` delegates to a subprocess in deepmd_env.
        Auto-detected when *None*.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, config=None, model_key=None, device=None,
                 n_threads=None, mode=None, **kwargs):
        super().__init__(**kwargs)

        if config is None:
            config = load_config()

        deepmd_cfg = config["deepmd"]
        self.conda_env = deepmd_cfg["conda_env"]
        self.device = device or deepmd_cfg.get("device", "cpu")
        self.n_threads = n_threads or int(os.environ.get("OMP_NUM_THREADS", 1))

        key = model_key or deepmd_cfg.get("model_key", "base_deepmd")
        self.model_path = resolve_model(deepmd_cfg["models_path"], key)

        if mode is None:
            try:
                from deepmd.calculator import DP  # noqa: F401
                mode = "direct"
            except ImportError:
                mode = "worker"
        self.mode = mode

        self._proc = None   # worker subprocess handle
        self._dp = None     # direct DP calculator instance

    # -- direct mode ---------------------------------------------------

    def _ensure_direct(self):
        if self._dp is not None:
            return
        import torch
        from deepmd.calculator import DP

        self._dp = DP(model=str(self.model_path), device=self.device)
        torch.set_num_threads(self.n_threads)

    def _calculate_direct(self):
        self._ensure_direct()
        self._dp.calculate(atoms=self.atoms)
        self.results.update(self._dp.results)

    # -- worker mode ---------------------------------------------------

    def _ensure_worker(self):
        if self._proc is not None and self._proc.poll() is None:
            return

        self._proc = subprocess.Popen(
            [
                "conda", "run", "--no-capture-output",
                "-n", self.conda_env,
                "python", str(WORKER_SCRIPT),
                str(self.model_path),
                "--device", self.device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self._proc.stdout.readline()
        msg = json.loads(ready)
        if msg.get("status") != "ready":
            raise RuntimeError(f"Worker failed to start: {msg}")

    def _calculate_worker(self):
        self._ensure_worker()

        atoms = self.atoms
        request = json.dumps({
            "coords": atoms.get_positions().flatten().tolist(),
            "cell": atoms.get_cell().array.flatten().tolist(),
            "symbols": atoms.get_chemical_symbols(),
        })
        self._proc.stdin.write(request + "\n")
        self._proc.stdin.flush()

        response = self._proc.stdout.readline()
        if not response:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Worker died: {stderr}")

        data = json.loads(response)
        if "error" in data:
            raise RuntimeError(f"Worker error: {data['error']}")

        self.results["energy"] = data["energy"]
        self.results["forces"] = np.array(data["forces"])
        self.results["stress"] = np.array(data["stress"])

    # -- ASE interface -------------------------------------------------

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if self.mode == "direct":
            self._calculate_direct()
        else:
            self._calculate_worker()

    def close(self):
        """Terminate the worker subprocess (no-op in direct mode)."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def __del__(self):
        self.close()
