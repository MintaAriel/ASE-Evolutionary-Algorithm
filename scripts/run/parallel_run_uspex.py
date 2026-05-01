#!/usr/bin/env python3
"""USPEX runner for the BATCHED DeepMD workflow.

Sibling of ``run_uspex.py``.  In the per-CalcFolder flow, each
CalcFolder runs ``uspex_deepmd_gfnff.py`` independently.  In the
batched flow, every CalcFolder's ``run.sh`` is just a stub that prints
and waits for ``energy.txt`` to appear, while ONE call to
``uspex_deemdp_parallel.py`` per generation does the heavy lifting
across all CalcFolders at once.

Per-generation cycle:

    1. ``USPEX -r``  — USPEX populates each CalcFolder with ``geom.in``
       and spawns the stub ``run.sh`` (one PID per folder, all alive).
    2. Wait until every active CalcFolder has a ``geom.in`` (so the
       batch sees the final inputs USPEX wrote, not a half-written one).
    3. Run ``uspex_deemdp_parallel.py`` once on the workdir — it writes
       ``geom.out`` then ``energy.txt`` into each CalcFolder.
    4. The stub ``run.sh`` in each folder sees ``energy.txt`` and exits;
       wait for all those PIDs to drop.
    5. Loop back to (1) — the next ``USPEX -r`` collects results and
       generates the next generation.

Stops when USPEX writes ``USPEX_IS_DONE`` or ``NOT_YET`` to the workdir.
Requires ``whichCluster = 1`` so ``submitJob_local.py`` writes
``job.info`` per CalcFolder (same requirement as ``run_uspex.py``).
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Reuse parse_uspex_input / get_active_pids / is_done / wait_for_uspex_x
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_uspex import (
    get_active_pids,
    is_done,
    parse_uspex_input,
    wait_for_uspex_x,
)

# Project imports — make ``ea.utils.config`` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ea.utils.config import load_config

log = logging.getLogger("parallel_run_uspex")
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Workdir state
# ---------------------------------------------------------------------------

def pending_calcfolders(workdir):
    """CalcFolders that have geom.in but no energy.txt yet."""
    pending = []
    for cf in sorted(Path(workdir).glob("CalcFold[0-9]*")):
        if not cf.is_dir():
            continue
        if (cf / "geom.in").is_file() and not (cf / "energy.txt").is_file():
            pending.append(cf)
    return pending


def wait_until_inputs_ready(workdir, num_parallel, poll_interval, settle_secs=2.0):
    """Block until #pending CalcFolders == num_parallel and the count is
    stable for ``settle_secs`` (so we don't catch USPEX mid-write).

    Returns the final pending list.
    """
    last_n = -1
    last_change = time.monotonic()
    while not _stop_event.is_set():
        pending = pending_calcfolders(workdir)
        n = len(pending)
        if n != last_n:
            log.info("Waiting for inputs: %d/%d CalcFolders ready", n, num_parallel)
            last_n = n
            last_change = time.monotonic()
        if n >= num_parallel and (time.monotonic() - last_change) >= settle_secs:
            return pending
        if is_done(workdir):
            return pending
        _stop_event.wait(poll_interval)
    return pending_calcfolders(workdir)


def wait_for_pids_to_die(workdir, poll_interval):
    """Block until every CalcFolder's job.info PID has exited.

    The stub run.sh in each CalcFolder polls for energy.txt and exits
    once it appears, so this should drain quickly after the batch worker
    finishes writing.
    """
    while not _stop_event.is_set():
        active = get_active_pids(workdir)
        if not active:
            return
        names = ", ".join(f"{k} (pid {v})" for k, v in active.items())
        log.info("Waiting for stub PIDs to exit: %s", names)
        _stop_event.wait(poll_interval)


# ---------------------------------------------------------------------------
# USPEX + batch worker invocation
# ---------------------------------------------------------------------------

def run_uspex_resume(workdir, log_file, uspex_cmd, env):
    log.info("Running %s -r", uspex_cmd)
    result = subprocess.run(
        [uspex_cmd, "-r"],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"USPEX -r  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")
        f.write(result.stdout or "")
    if result.returncode != 0:
        log.warning("USPEX -r exited with code %d", result.returncode)
    return result


def run_batch_worker(workdir, worker_script, model, device, zpe, batch_log,
                     worker_python):
    """Invoke uspex_deemdp_parallel.py on the workdir; block until done.

    ``worker_python`` is the absolute path of the python that has DeepMD
    installed (typically ``~/miniconda3/envs/deepmd_env/bin/python``);
    falls back to ``sys.executable`` when unset.
    """
    py = worker_python or sys.executable
    cmd = [py, str(worker_script), str(workdir), "--model", model]
    if device:
        cmd += ["--device", device]
    if zpe:
        cmd += ["--zpe"]
    log.info("Running batch worker: %s", " ".join(cmd))
    with open(batch_log, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"batch worker  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log.error("batch worker exited with code %d (see %s)",
                  result.returncode, batch_log)
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop(workdir, num_parallel, poll_interval, log_file, batch_log,
              uspex_cmd, env, worker_script, model, device, zpe, worker_python):
    wait_for_uspex_x(workdir)

    if is_done(workdir):
        log.info("Already done — exiting")
        return

    while not _stop_event.is_set():
        if is_done(workdir):
            log.info("USPEX signaled completion — stopping")
            break

        # 1. submit a generation
        run_uspex_resume(workdir, log_file, uspex_cmd, env)

        if is_done(workdir):
            break

        # 2. wait for inputs
        pending = wait_until_inputs_ready(workdir, num_parallel, poll_interval)
        if _stop_event.is_set():
            break
        if not pending:
            log.info("No pending CalcFolders after USPEX -r; calling USPEX -r again")
            continue

        # 3. batch-process them
        log.info("Batch-processing %d CalcFolders", len(pending))
        run_batch_worker(workdir, worker_script, model, device, zpe, batch_log,
                         worker_python)

        # 4. let stubs exit so USPEX sees PIDs gone on next -r
        wait_for_pids_to_die(workdir, poll_interval)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

DEFAULT_WORKER = (Path(__file__).resolve().parents[1]
                  / "experiments" / "uspex_deemdp_parallel.py")


def main():
    p = argparse.ArgumentParser(
        description="USPEX runner for batched DeepMD relaxation.",
    )
    p.add_argument("--workdir", default=os.getcwd(),
                   help="USPEX working directory (default: cwd)")
    p.add_argument("--poll-interval", type=float, default=5.0,
                   help="Seconds between status polls (default: 5)")
    p.add_argument("--clean", action="store_true",
                   help="Run USPEX --clean before starting")
    p.add_argument("--log-file", default=None,
                   help="USPEX -r output log (default: <workdir>/uspex_runner.log)")
    p.add_argument("--batch-log", default=None,
                   help="Batch-worker output log "
                        "(default: <workdir>/uspex_batch_worker.log)")
    p.add_argument("--worker-script", default=str(DEFAULT_WORKER),
                   help=f"Path to the batch worker (default: {DEFAULT_WORKER})")
    p.add_argument("--model", default="deepmd_d3",
                   help="DeepMD model key passed to the batch worker")
    p.add_argument("--device", default=None,
                   help="DeepMD device override (default: deepmd.device from config)")
    p.add_argument("--zpe", action="store_true",
                   help="Add Gamma-point ZPE per structure inside the batch worker")
    p.add_argument("--worker-python", default=None,
                   help="Absolute python path for the batch worker "
                        "(default: ~/miniconda3/envs/<deepmd.conda_env>/bin/python "
                        "from config; falls back to current python)")
    p.add_argument("--uspex-cmd", default=None,
                   help="USPEX executable (default: uspex.exe from sirena.yaml)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    workdir = os.path.abspath(args.workdir)
    log_file = args.log_file or os.path.join(workdir, "uspex_runner.log")
    batch_log = args.batch_log or os.path.join(workdir, "uspex_batch_worker.log")
    worker_script = Path(args.worker_script).expanduser().resolve()
    if not worker_script.is_file():
        sys.exit(f"worker script not found: {worker_script}")

    # USPEX-related paths from config
    cfg = load_config()
    uspex_cfg = cfg.get("uspex") or {}
    uspex_cmd = args.uspex_cmd or os.path.expanduser(uspex_cfg.get("exe", "USPEX"))

    env = os.environ.copy()
    if "uspex_path" in uspex_cfg:
        env["MYUSPEXPATH"] = uspex_cfg["uspex_path"]
        env["USPEXPATH"] = uspex_cfg["uspex_path"]
    if "mcr_root" in uspex_cfg:
        env["MCRROOT"] = uspex_cfg["mcr_root"]

    # Resolve which python to use for the batch worker.
    worker_python = args.worker_python
    if worker_python is None:
        worker_env = (cfg.get("deepmd") or {}).get("conda_env")
        if worker_env:
            cand = Path(f"~/miniconda3/envs/{worker_env}/bin/python").expanduser()
            if cand.is_file():
                worker_python = str(cand)
    if worker_python is None:
        worker_python = sys.executable
        log.warning("worker python not configured — falling back to %s. "
                    "DeepMD must be importable here, or pass --worker-python.",
                    worker_python)

    # Parse USPEX INPUT.txt
    uspex_params = parse_uspex_input(workdir)
    num_parallel = int(uspex_params.get("numParallelCalcs", 1))
    abinitio_code = uspex_params.get("abinitioCode", "?")
    which_cluster = int(uspex_params.get("whichCluster", 0))

    if which_cluster != 1:
        log.error(
            "whichCluster = %s but this runner requires whichCluster = 1 "
            "(submitJob_local.py writes job.info per CalcFolder).",
            which_cluster,
        )
        sys.exit(1)

    log.info("parallel_run_uspex starting")
    log.info("  uspex command:    %s", uspex_cmd)
    log.info("  workdir:          %s", workdir)
    log.info("  abinitioCode:     %s", abinitio_code)
    log.info("  numParallelCalcs: %d", num_parallel)
    log.info("  worker script:    %s", worker_script)
    log.info("  worker python:    %s", worker_python)
    log.info("  model:            %s", args.model)
    log.info("  zpe:              %s", args.zpe)

    def _on_signal(signum, _frame):
        log.info("Signal %d received — stopping after current step", signum)
        _stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if args.clean:
        log.info("Running %s --clean", uspex_cmd)
        subprocess.run([uspex_cmd, "--clean"], cwd=workdir, env=env)

    main_loop(workdir, num_parallel, args.poll_interval,
              log_file, batch_log, uspex_cmd, env,
              worker_script, args.model, args.device, args.zpe,
              worker_python)
    log.info("Done.")


if __name__ == "__main__":
    main()
