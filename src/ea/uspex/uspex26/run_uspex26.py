#!/usr/bin/env python3
"""Python driver for USPEX 26 + batched DeepMD relaxation.

Replaces the old ``run_uspex26_deepmd.sh``.  USPEX 26 runs the whole
evolutionary search in ONE continuous process.  For abinitioCode 20
(ASE, required for molecular atom-order preservation) it drops, per
structure of the current relaxation wave, a
folder::

    Calculation/Calcfold_<system>_<step>/

containing ``input.xyz`` and launches ``commandExecutable``
(``run_batch.py``), which just waits for ``output.xyz``.  Legacy code-99
``geom.in``/``energy.txt`` folders remain supported for non-molecular runs.
``numParallelCalcs`` sets how many run at once and USPEX pins one core to
each (``whichCluster = 0`` -> 'local, free cores').

Once a whole wave of folders is present and settled, this driver calls
the batched worker (``worker.py``) ONCE; the worker relaxes every pending
Calcfold and writes ``output.xyz`` back, letting each
``run_batch.py`` exit so USPEX collects results and launches the next wave.

That per-wave batching is a GPU strategy.  With ``--worker cpu`` the driver
instead starts ``worker_cpu.py`` once, in watch mode, for the whole search:
it keeps one DeepMD process per core alive and relaxes each structure
independently as its folder appears, so no core waits on a batch barrier.
Pick the cores with ``--cores`` (a count like ``8``, or a range like
``0-11``); under SLURM the selection is intersected with the job's
allocation.

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
import re
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
WORKER_CPU = SCRIPT_DIR / "worker_cpu.py"
WORKER_CPU_DYN = SCRIPT_DIR / "worker_cpu_dynamic.py"
CALC_DIR = "Calculation"      # USPEX 26 base_calc_folder
DEFAULT_USPEX = "/home/vito/uspex_python/USPEX26/uspex_2607_linux/uspex_2607.sh"

POLL = 3        # seconds between scans
SETTLE = 3      # require the pending set to be stable this long
STOP_FILE = ".cpu_worker_stop"   # graceful-stop marker for worker_cpu --watch
CPU_WORKER_GRACE = 1800          # seconds to let in-flight relaxations finish


def validate_molecular_interface(workdir):
    """Fail early if a molecular run is configured with order-losing code 99."""
    workdir = Path(workdir)
    if not any(workdir.glob("MOL_*")):
        return

    input_path = workdir / "INPUT.txt"
    if not input_path.is_file():
        sys.exit(f"[launcher] ERROR: INPUT.txt not found in {workdir}")

    text = input_path.read_text(errors="replace")
    match = re.search(
        r"%\s*abinitioCode\s*\r?\n\s*([^\s%]+)", text, flags=re.IGNORECASE
    )
    code = match.group(1) if match else None
    if code != "20":
        found = code if code is not None else "missing"
        sys.exit(
            "[launcher] ERROR: molecular MOL_* input requires abinitioCode 20 "
            f"(found {found}). Code 99 globally groups atoms by element and "
            "USPEX 26 cannot reconstruct the relaxed molecules."
        )


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
    """Count ready Calcfold inputs that do not have their interface output."""
    base = Path(workdir) / CALC_DIR
    if not base.is_dir():
        return 0
    n = 0
    for d in base.glob("Calcfold_*"):
        ase_input = d / "input.xyz"
        user_input = d / "geom.in"
        ase_pending = (ase_input.is_file() and ase_input.stat().st_size > 0
                       and not (d / "output.xyz").is_file())
        user_pending = (user_input.is_file() and user_input.stat().st_size > 0
                        and not (d / "energy.txt").is_file())
        if d.is_dir() and (ase_pending or user_pending):
            n += 1
    return n


def _in_deepmd_env(cfg, worker_python, tail):
    """Wrap a worker command so it runs with the DeepMD dependencies."""
    if worker_python:
        return [worker_python, *tail]
    conda_env = (cfg.get("deepmd") or {}).get("conda_env", "deepmd_env")
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *tail]


def build_worker_cmd(cfg, workdir, model, device, worker_python, smoke=False):
    """Command that runs the batched (GPU-shaped) worker.py."""
    tail = [str(WORKER), str(workdir), "--model", model]
    if device:
        tail += ["--device", device]
    if smoke:
        tail.append("--smoke")
    return _in_deepmd_env(cfg, worker_python, tail)


def build_cpu_worker_cmd(cfg, workdir, model, worker_python, args, stop_file,
                         dynamic=False):
    """Command that runs the per-structure CPU worker in watch mode.

    *dynamic* selects worker_cpu_dynamic.py, which sizes its own concurrency
    from measured RSS instead of the static --mem-per-worker-gb clamp.
    """
    script = WORKER_CPU_DYN if dynamic else WORKER_CPU
    tail = [str(script), str(workdir), "--model", model,
            "--watch", "--stop-file", str(stop_file),
            "--threads", str(args.threads)]
    if args.cores:
        tail += ["--cores", str(args.cores)]
    if args.workers:
        tail += ["--workers", str(args.workers)]
    if args.smt:
        tail.append("--smt")
    if args.oversubscribe:
        tail.append("--oversubscribe")
    if args.no_pin:
        tail.append("--no-pin")
    if args.mem_per_worker_gb is not None:
        tail += ["--mem-per-worker-gb", str(args.mem_per_worker_gb)]
    if args.reserve_gb is not None:
        tail += ["--reserve-gb", str(args.reserve_gb)]
    if dynamic:
        for flag, value in (("--max-ram-gb", args.max_ram_gb),
                            ("--min-node-free-gb", args.min_node_free_gb),
                            ("--check-interval", args.check_interval),
                            ("--admit-per-cycle", args.admit_per_cycle)):
            if value is not None:
                tail += [flag, str(value)]
    if args.smoke:
        tail.append("--smoke")
    return _in_deepmd_env(cfg, worker_python, tail)


def worker_env():
    """Environment for the workers: this checkout's ``src`` on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = EA_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_worker(cmd, workdir, n, device, log_path):
    print(f"[launcher] {time.strftime('%H:%M:%S')} relaxing wave of {n} "
          f"folder(s) with DeepMD ({device})", flush=True)
    with open(log_path, "a") as log:
        subprocess.run(
            cmd, cwd=workdir, env=worker_env(), stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )


def start_cpu_worker(cmd, workdir, log_path):
    """Launch the per-structure CPU worker in watch mode for the whole run."""
    log = open(log_path, "a")
    proc = subprocess.Popen(cmd, cwd=workdir, env=worker_env(), stdout=log,
                            stderr=subprocess.STDOUT)
    print(f"[launcher] CPU worker PID {proc.pid} (watch mode, "
          f"log: {os.path.basename(log_path)})", flush=True)
    return proc, log


def stop_cpu_worker(proc, stop_file, timeout=CPU_WORKER_GRACE):
    """Ask the CPU worker to finish in-flight relaxations, then make sure it did.

    The stop file is the graceful path: the worker publishes the results it is
    already computing before exiting, so no structure is lost.
    """
    if proc is None or proc.poll() is not None:
        return
    print(f"[launcher] asking CPU worker (PID {proc.pid}) to stop after its "
          f"in-flight relaxations (up to {timeout}s)", flush=True)
    try:
        Path(stop_file).touch()
    except OSError as exc:
        print(f"[launcher] could not create stop file {stop_file}: {exc}",
              flush=True)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[launcher] CPU worker still running; terminating", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[launcher] CPU worker ignored SIGTERM; killing", flush=True)
        proc.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=os.getcwd(),
                   help="Directory USPEX 26 runs in (default: cwd)")
    p.add_argument("--model", default="deepmd_d3",
                   help="DeepMD model key (default: deepmd_d3)")
    p.add_argument("--device", default=None,
                   help="DeepMD device for the batched worker: 'cpu' (default) "
                        "or 'cuda' (falls back to config deepmd.device)")
    p.add_argument("--worker", choices=("batch", "cpu", "cpu-dyn"),
                   default="batch",
                   help="'batch' (default): one batched relaxation per wave "
                        "(worker.py, built for a GPU). 'cpu': one independent "
                        "relaxation per core, run continuously (worker_cpu.py). "
                        "'cpu-dyn': same, but concurrency is set by measured "
                        "RAM instead of a fixed count (worker_cpu_dynamic.py)")

    cpu = p.add_argument_group(
        "cpu worker (--worker cpu / cpu-dyn)",
        "Core selection for the per-structure CPU worker. Intersected with "
        "this process' affinity mask, so a SLURM allocation is respected.")
    cpu.add_argument("--cores", default=None,
                     help="How many cores, or exactly which: '8' (count), "
                          "'0-11' (range), '0,2,4-6' (list). "
                          "Default: every allowed physical core")
    cpu.add_argument("--threads", type=int, default=1,
                     help="CPUs/threads per relaxation (default: 1)")
    cpu.add_argument("--workers", type=int, default=None,
                     help="Cap on concurrent relaxations "
                          "(default: as many as the cores allow)")
    cpu.add_argument("--smt", action="store_true",
                     help="Also use SMT sibling threads as separate workers")
    cpu.add_argument("--oversubscribe", action="store_true",
                     help="Allow --workers to exceed the available cores")
    cpu.add_argument("--no-pin", action="store_true",
                     help="Do not set CPU affinity, only cap thread pools")
    cpu.add_argument("--mem-per-worker-gb", type=float, default=None,
                     help="RAM one relaxation needs; clamps the worker count "
                          "so the node does not swap (worker default: 3.0, "
                          "0 disables)")
    cpu.add_argument("--reserve-gb", type=float, default=None,
                     help="RAM held back for USPEX itself and the OS "
                          "(worker default: 2.0)")

    dyn = p.add_argument_group(
        "dynamic memory scheduler (--worker cpu-dyn)",
        "Concurrency is decided from the measured resident size of the live "
        "relaxations, so --mem-per-worker-gb only seeds the estimate.")
    dyn.add_argument("--max-ram-gb", type=float, default=None,
                     help="Total resident budget for all relaxations, GiB "
                          "(default: allocation minus --reserve-gb). Set it "
                          "below the cgroup limit so an overshoot queues "
                          "instead of being OOM-killed")
    dyn.add_argument("--min-node-free-gb", type=float, default=None,
                     help="Stop admitting while the NODE's MemAvailable is "
                          "below this, GiB (worker default: 24)")
    dyn.add_argument("--check-interval", type=float, default=None,
                     help="Seconds between admission decisions "
                          "(worker default: 15)")
    dyn.add_argument("--admit-per-cycle", type=int, default=None,
                     help="Most workers to start per --check-interval "
                          "(worker default: 1)")

    p.add_argument("--uspex-exe", default=None,
                   help="USPEX 26 binary (default: config uspex26.exe)")
    p.add_argument("--worker-python", default=os.environ.get("WORKER_PY"),
                   help="Explicit python for worker.py; default runs it via "
                        "'conda run -n <deepmd.conda_env>'")
    p.add_argument("--keep", action="store_true",
                   help="Do NOT wipe Calculation/ CalculationTemp/ Specific/ "
                        "results1/ before starting")
    p.add_argument("--fresh", action="store_true",
                   help="Confirm deleting an existing Calculation/ or "
                        "results1/ and restarting at generation 1. Required "
                        "whenever a previous search is present, so a stray "
                        "relaunch cannot silently discard days of work")
    p.add_argument("--smoke", action="store_true",
                   help="Pass the abbreviated smoke-test optimization profile "
                        "to the worker; do not use for production")
    args = p.parse_args()

    cfg = load_config()
    workdir = os.path.abspath(os.path.expanduser(args.workdir))
    uspex = resolve_uspex(cfg, args.uspex_exe)
    device = args.device or (cfg.get("deepmd") or {}).get("device", "cpu")

    if args.worker in ("cpu", "cpu-dyn"):
        if args.device and args.device != "cpu":
            sys.exit(f"[launcher] ERROR: --worker {args.worker} cannot run on "
                     f"--device {args.device}; drop --device or use "
                     f"--worker batch")
        device = "cpu"

    if not (os.path.isfile(uspex) and os.access(uspex, os.X_OK)):
        sys.exit(f"[launcher] ERROR: USPEX 26 binary not found/executable: {uspex}\n"
                 f"[launcher]        set --uspex-exe, $USPEX26_EXE, or uspex26.exe in the EA config.")
    if not os.path.isdir(workdir):
        sys.exit(f"[launcher] ERROR: workdir not found: {workdir}")

    validate_molecular_interface(workdir)

    os.chdir(workdir)

    # Fresh run: USPEX 26 regenerates these every launch.  Wiping stays the
    # default for an empty workdir, but it is never silent once there is a
    # search to lose: a requeue that relaunched this script destroyed four
    # partial runs (37 h of relaxations) before anyone noticed, because USPEX
    # restarts at generation 1 with no trace of what it deleted.
    if not args.keep:
        existing = [d for d in ("Calculation", "results1")
                    if os.path.isdir(os.path.join(workdir, d))
                    and os.listdir(os.path.join(workdir, d))]
        if existing and not args.fresh:
            sys.exit(
                f"[launcher] ERROR: {' and '.join(existing)} already hold a "
                f"previous search under\n"
                f"[launcher]        {workdir}\n"
                f"[launcher]        Starting would delete it and restart at "
                f"generation 1. Choose explicitly:\n"
                f"[launcher]          --keep   leave it in place\n"
                f"[launcher]          --fresh  delete it and start over")
        subprocess.run(["rm", "-rf", "Calculation", "CalculationTemp",
                        "Specific", "results1"], cwd=workdir)

    dyn_mode = args.worker == "cpu-dyn"
    cpu_mode = args.worker in ("cpu", "cpu-dyn")
    stop_file = os.path.join(workdir, STOP_FILE)
    if cpu_mode:
        # A leftover marker from a previous run would stop the worker at once.
        if os.path.exists(stop_file):
            os.remove(stop_file)
        worker_cmd = build_cpu_worker_cmd(
            cfg, workdir, args.model, args.worker_python, args, stop_file,
            dynamic=dyn_mode
        )
    else:
        worker_cmd = build_worker_cmd(
            cfg, workdir, args.model, device, args.worker_python, args.smoke
        )
    worker_log = os.path.join(workdir, "deepmd_worker.log")

    print(f"[launcher] workdir : {workdir}")
    print(f"[launcher] uspex26 : {uspex}")
    mode_name = ("per-structure CPU pool, memory-scheduled" if dyn_mode
                 else "per-structure CPU pool" if cpu_mode
                 else "batched per wave")
    print(f"[launcher] mode    : {mode_name}")
    print(f"[launcher] worker  : {' '.join(worker_cmd)}")
    print(f"[launcher] starting USPEX 26 (model={args.model}, device={device})", flush=True)

    uspex_log = open(os.path.join(workdir, "uspex_run.log"), "w")
    proc = subprocess.Popen([uspex], cwd=workdir, stdout=uspex_log,
                            stderr=subprocess.STDOUT)
    print(f"[launcher] USPEX 26 PID {proc.pid}  (log: uspex_run.log)", flush=True)

    cpu_proc = cpu_log = None
    if cpu_mode:
        cpu_proc, cpu_log = start_cpu_worker(worker_cmd, workdir, worker_log)

    def cleanup(*_):
        if cpu_mode:
            stop_cpu_worker(cpu_proc, stop_file)
        if proc.poll() is None:
            print(f"[launcher] stopping USPEX 26 (PID {proc.pid})", flush=True)
            proc.terminate()

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(143)))

    try:
        if cpu_mode:
            # The worker watches for Calcfolds itself; just supervise the pair.
            print(f"[launcher] CPU worker driver started "
                  f"(worker: {WORKER_CPU_DYN if dyn_mode else WORKER_CPU})",
                  flush=True)
            while proc.poll() is None:
                if cpu_proc.poll() is not None:
                    print(f"[launcher] ERROR: CPU worker exited "
                          f"(code {cpu_proc.returncode}) while USPEX is still "
                          f"running — see {os.path.basename(worker_log)}. "
                          f"Stopping USPEX.", flush=True)
                    break
                time.sleep(POLL)
            else:
                print("[launcher] USPEX 26 finished", flush=True)
            n = count_pending(workdir)
            if n and cpu_proc.poll() is None:
                # cleanup() below sets the stop file, and the worker then only
                # finishes what it already started -- it accepts no new
                # folders. Anything still unclaimed here stays unrelaxed, but
                # USPEX has exited so nothing is waiting on it either.
                print(f"[launcher] {n} folder(s) were still pending when USPEX "
                      f"exited; the CPU worker publishes its in-flight "
                      f"relaxations only", flush=True)
        else:
            print(f"[launcher] DeepMD worker driver started (worker: {WORKER})",
                  flush=True)
            while proc.poll() is None:
                n = count_pending(workdir)
                if n > 0:
                    # Wait until the wave stops growing, so USPEX has finished
                    # writing every input file before the batched worker reads them.
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
        if cpu_log is not None:
            cpu_log.close()
        if cpu_mode and os.path.exists(stop_file):
            os.remove(stop_file)
        print(f"[launcher] done (USPEX exit={rc})", flush=True)


if __name__ == "__main__":
    main()
