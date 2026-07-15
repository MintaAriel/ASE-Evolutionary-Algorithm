#!/usr/bin/env python3
"""Python driver for USPEX 26 + batched DeepMD relaxation.

Replaces the old ``run_uspex26_deepmd.sh``.  USPEX 26 runs the whole
evolutionary search in ONE continuous process.  For abinitioCode 99
(USER_CODE) it drops, per structure of the current relaxation wave, a
folder::

    Calculation/Calcfold_<system>_<step>/

containing ``geom.in`` (POSCAR) and launches ``commandExecutable``
(``run_batch.py``), which just waits for ``energy.txt``.
``numParallelCalcs`` sets how many run at once and USPEX pins one core to
each (``whichCluster = 0`` -> 'local, free cores').

Once a whole wave of folders is present and settled, this driver calls
the batched worker (``worker.py``) ONCE; the worker relaxes every pending
Calcfold and writes ``geom.out`` + ``energy.txt`` back, letting each
``run_batch.py`` exit so USPEX collects results and launches the next wave.

The worker needs the DeepMD environment, so it is run in the ``deepmd``
conda env (``deepmd.conda_env`` in the EA config) via ``conda run`` unless
``--worker-python`` / ``$WORKER_PY`` points at an explicit interpreter.
This driver itself needs no DeepMD dependency.

Usage::

    python run_uspex26.py                       # workdir = cwd, model=deepmd_d3, cpu
    python run_uspex26.py --workdir /path/to/run --model deepmd_d3 --device cuda

Environment overrides: ``USPEX26_EXE``, ``WORKER_PY``, ``EA_CONFIG``.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Make ``ea.utils.config`` importable when run as a script.
# This file lives at src/ea/uspex/uspex26/run_uspex26.py, so parents[3] == src.
EA_SRC = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, EA_SRC)
from ea.utils.config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "worker.py"
CALC_DIR = "Calculation"      # USPEX 26 base_calc_folder
DEFAULT_USPEX = "/home/vito/uspex_python/USPEX26/uspex_2607_linux/uspex_2607.sh"

POLL = 3        # seconds between scans
SETTLE = 3      # require the pending set to be stable this long


def resolve_uspex(cfg, override=None):
    """USPEX 26 binary: --uspex-exe -> $USPEX26_EXE -> config uspex26.exe -> default."""
    if override:
        return os.path.expanduser(override)
    env = os.environ.get("USPEX26_EXE")
    if env:
        return os.path.expanduser(env)
    exe = (cfg.get("uspex26") or {}).get("exe")
    return os.path.expanduser(exe) if exe else DEFAULT_USPEX


def count_pending(workdir):
    """Calcfold_* folders with a non-empty geom.in but no energy.txt."""
    base = Path(workdir) / CALC_DIR
    if not base.is_dir():
        return 0
    n = 0
    for d in base.glob("Calcfold_*"):
        gi = d / "geom.in"
        if (d.is_dir() and gi.is_file() and gi.stat().st_size > 0
                and not (d / "energy.txt").is_file()):
            n += 1
    return n


def build_worker_cmd(cfg, workdir, model, device, worker_python):
    """Command that runs worker.py in the DeepMD environment."""
    tail = [str(WORKER), str(workdir), "--model", model]
    if device:
        tail += ["--device", device]
    if worker_python:
        return [worker_python, *tail]
    conda_env = (cfg.get("deepmd") or {}).get("conda_env", "deepmd_env")
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *tail]


def run_worker(cmd, workdir, n, device, log_path):
    print(f"[launcher] {time.strftime('%H:%M:%S')} relaxing wave of {n} "
          f"folder(s) with DeepMD ({device})", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = EA_SRC + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_path, "a") as log:
        subprocess.run(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=os.getcwd(),
                   help="Directory USPEX 26 runs in (default: cwd)")
    p.add_argument("--model", default="deepmd_d3",
                   help="DeepMD model key (default: deepmd_d3)")
    p.add_argument("--device", default=None,
                   help="DeepMD device: 'cpu' (default) or 'cuda' "
                        "(falls back to config deepmd.device)")
    p.add_argument("--uspex-exe", default=None,
                   help="USPEX 26 binary (default: config uspex26.exe)")
    p.add_argument("--worker-python", default=os.environ.get("WORKER_PY"),
                   help="Explicit python for worker.py; default runs it via "
                        "'conda run -n <deepmd.conda_env>'")
    p.add_argument("--keep", action="store_true",
                   help="Do NOT wipe Calculation/ CalculationTemp/ Specific/ "
                        "results1/ before starting")
    args = p.parse_args()

    cfg = load_config()
    workdir = os.path.abspath(os.path.expanduser(args.workdir))
    uspex = resolve_uspex(cfg, args.uspex_exe)
    device = args.device or (cfg.get("deepmd") or {}).get("device", "cpu")

    if not (os.path.isfile(uspex) and os.access(uspex, os.X_OK)):
        sys.exit(f"[launcher] ERROR: USPEX 26 binary not found/executable: {uspex}\n"
                 f"[launcher]        set --uspex-exe, $USPEX26_EXE, or uspex26.exe in the EA config.")
    if not os.path.isdir(workdir):
        sys.exit(f"[launcher] ERROR: workdir not found: {workdir}")

    os.chdir(workdir)

    # Fresh run: USPEX 26 regenerates these every launch.
    if not args.keep:
        subprocess.run(["rm", "-rf", "Calculation", "CalculationTemp",
                        "Specific", "results1"], cwd=workdir)

    worker_cmd = build_worker_cmd(cfg, workdir, args.model, device, args.worker_python)
    worker_log = os.path.join(workdir, "deepmd_worker.log")

    print(f"[launcher] workdir : {workdir}")
    print(f"[launcher] uspex26 : {uspex}")
    print(f"[launcher] worker  : {' '.join(worker_cmd)}")
    print(f"[launcher] starting USPEX 26 (model={args.model}, device={device})", flush=True)

    uspex_log = open(os.path.join(workdir, "uspex_run.log"), "w")
    proc = subprocess.Popen([uspex], cwd=workdir, stdout=uspex_log,
                            stderr=subprocess.STDOUT)
    print(f"[launcher] USPEX 26 PID {proc.pid}  (log: uspex_run.log)", flush=True)

    def cleanup(*_):
        if proc.poll() is None:
            print(f"[launcher] stopping USPEX 26 (PID {proc.pid})", flush=True)
            proc.terminate()

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(143)))

    try:
        print(f"[launcher] DeepMD worker driver started (worker: {WORKER})", flush=True)
        while proc.poll() is None:
            n = count_pending(workdir)
            if n > 0:
                # Wait until the wave stops growing, so USPEX has finished
                # writing every geom.in before the batched worker reads them.
                time.sleep(SETTLE)
                if count_pending(workdir) == n and proc.poll() is None:
                    run_worker(worker_cmd, workdir, n, device, worker_log)
                    continue
            time.sleep(POLL)

        print("[launcher] USPEX 26 finished; final worker sweep", flush=True)
        n = count_pending(workdir)
        if n > 0:
            run_worker(worker_cmd, workdir, n, device, worker_log)
    finally:
        cleanup()
        rc = proc.wait()
        uspex_log.close()
        print(f"[launcher] done (USPEX exit={rc})", flush=True)


if __name__ == "__main__":
    main()
