#!/usr/bin/env python3
"""
USPEX runner — manages the USPEX -r resume loop with instance-safe
PID tracking via CalcFolder job.info files.

Requires whichCluster = 1 in INPUT.txt so USPEX uses submitJob_local.py
(which writes job.info with the PID of each CalcFolder process).

Usage:
    python run_uspex.py --workdir /path/to/uspex_run
    python run_uspex.py --workdir /path/to/uspex_run --clean
    python run_uspex.py --workdir /path/to/uspex_run --poll-interval 10
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

log = logging.getLogger("run_uspex")

# Event set by signal handler to request graceful shutdown
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# INPUT.txt parser (USPEX format, not the EA's ase_INPUT.txt)
# ---------------------------------------------------------------------------

def parse_uspex_input(workdir):
    """Parse USPEX INPUT.txt for runner-relevant parameters.

    Returns a flat dict with string values; caller converts types.
    """
    input_file = Path(workdir) / "INPUT.txt"
    if not input_file.is_file():
        raise FileNotFoundError(f"INPUT.txt not found in {workdir}")

    params = {}
    lines = input_file.read_text().splitlines()
    i, n = 0, len(lines)

    while i < n:
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("*"):
            continue

        # Block: % key ... % End*
        if line.startswith("%") and not line.lower().startswith("% end"):
            key = line.lstrip("% ").strip()
            values = []
            while i < n:
                vl = lines[i].strip()
                i += 1
                if not vl or vl.startswith("*"):
                    continue
                if vl.lower().startswith("% end"):
                    break
                values.append(vl)
            params[key] = values[0] if len(values) == 1 else values
            continue

        # Colon-separated: value : key
        if ":" in line:
            val_str, _, key_str = line.partition(":")
            key = key_str.strip()
            if key:
                params[key] = val_str.strip()

    return params


# ---------------------------------------------------------------------------
# PID tracking — instance-safe via CalcFolder job.info
# ---------------------------------------------------------------------------

def get_active_pids(workdir):
    """Read job.info from each CalcFolder and probe whether the PID is alive.

    Returns {calcfolder_name: pid} for running processes only.
    """
    active = {}
    for calcfold in sorted(Path(workdir).glob("CalcFold[0-9]*")):
        if not calcfold.is_dir():
            continue
        job_info = calcfold / "job.info"
        if not job_info.is_file():
            continue
        try:
            pid = int(job_info.read_text().strip())
            os.kill(pid, 0)          # signal-0 probe
            active[calcfold.name] = pid
        except (ValueError, ProcessLookupError):
            continue                 # bad file or process finished
        except PermissionError:
            active[calcfold.name] = pid  # alive but owned by another user
    return active


def is_done(workdir):
    """Return the stop-signal filename if USPEX finished, else None."""
    for name in ("USPEX_IS_DONE", "NOT_YET"):
        if (Path(workdir) / name).exists():
            return name
    return None


def wait_for_uspex_x(workdir):
    """Block until no uspex.x process has its CWD in *workdir*."""
    target = str(Path(workdir).resolve())
    while not _stop_event.is_set():
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                exe = str((entry / "exe").resolve())
                cwd = str((entry / "cwd").resolve())
                if "uspex" in exe.lower() and cwd == target:
                    found = True
                    break
            except (OSError, PermissionError):
                continue
        if not found:
            return
        log.info("Waiting for existing uspex.x to finish...")
        _stop_event.wait(10)


# ---------------------------------------------------------------------------
# USPEX invocation
# ---------------------------------------------------------------------------

def run_uspex_resume(workdir, log_file, uspex_cmd="USPEX"):
    """Call ``<uspex_cmd> -r`` and append its output to *log_file*."""
    log.info("Running %s -r", uspex_cmd)
    env = os.environ.copy()
    env["MYUSPEXPATH"] = "/home/sirena/Brian/uspex-matlab/application/archive/src"
    env["USPEXPATH"] = "/home/sirena/Brian/uspex-matlab/application/archive/src"
    env["MCRROOT"] = "/home/sirena/Brian/uspex-matlab"

    result = subprocess.run(
        [uspex_cmd, "-r"],
        cwd=workdir,
        env = env,
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop(workdir, max_active, poll_interval, log_file, uspex_cmd="USPEX"):
    wait_for_uspex_x(workdir)

    done = is_done(workdir)
    if done:
        log.info("Found %s — nothing to do", done)
        return

    while not _stop_event.is_set():
        done = is_done(workdir)
        if done:
            log.info("Found %s — stopping", done)
            break

        run_uspex_resume(workdir, log_file, uspex_cmd)

        # Throttle: wait until at least one CalcFolder slot is free
        _stop_event.wait(poll_interval)
        active = get_active_pids(workdir)

        while len(active) >= max_active and not _stop_event.is_set():
            names = ", ".join(f"{k} (pid {v})" for k, v in active.items())
            log.info("Waiting — %d/%d slots busy: %s",
                     len(active), max_active, names)
            _stop_event.wait(poll_interval)
            active = get_active_pids(workdir)

            if is_done(workdir):
                return

        if active:
            names = ", ".join(f"{k} (pid {v})" for k, v in active.items())
            log.info("Slot freed — still active: %s", names)

    if _stop_event.is_set():
        active = get_active_pids(workdir)
        if active:
            log.info("Shutdown requested — %d CalcFolder(s) still running, "
                     "they will finish on their own", len(active))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="USPEX runner with instance-safe PID tracking."
    )
    parser.add_argument("--workdir", default=os.getcwd(),
                        help="USPEX working directory (default: cwd)")
    parser.add_argument("--max-active", type=int, default=None,
                        help="Max concurrent CalcFolder jobs "
                             "(default: numParallelCalcs from INPUT.txt)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between status checks (default: 5)")
    parser.add_argument("--clean", action="store_true",
                        help="Run USPEX --clean before starting")
    parser.add_argument("--log-file", default=None,
                        help="USPEX output log (default: <workdir>/uspex_runner.log)")
    parser.add_argument("--uspex-cmd", default="USPEX",
                        help="Path or name of the USPEX executable "
                             "(default: USPEX)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    workdir = os.path.abspath(args.workdir)
    log_file = args.log_file or os.path.join(workdir, "uspex_runner.log")

    # --- Parse USPEX INPUT.txt ---
    uspex_params = parse_uspex_input(workdir)
    num_parallel = int(uspex_params.get("numParallelCalcs", 1))
    abinitio_code = uspex_params.get("abinitioCode", "?")
    which_cluster = int(uspex_params.get("whichCluster", 0))

    if which_cluster != 1:
        log.error(
            "whichCluster = %s but this runner requires whichCluster = 1 "
            "for PID tracking via job.info.  Update INPUT.txt and re-run.",
            which_cluster,
        )
        sys.exit(1)

    max_active = args.max_active if args.max_active is not None else num_parallel
    uspex_cmd = args.uspex_cmd

    log.info("USPEX runner starting")
    log.info("  uspex command:      %s", uspex_cmd)
    log.info("  workdir:            %s", workdir)
    log.info("  abinitioCode:       %s", abinitio_code)
    log.info("  numParallelCalcs:   %d", num_parallel)
    log.info("  max_active:         %d", max_active)
    log.info("  available cores:    %d", os.cpu_count() or 0)

    # --- Signal handling ---
    def _on_signal(signum, _frame):
        log.info("Signal %d received — stopping after current cycle", signum)
        _stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Optional clean ---
    if args.clean:
        log.info("Running %s --clean", uspex_cmd)
        subprocess.run([uspex_cmd, "--clean"], cwd=workdir)

    # --- Go ---
    main_loop(workdir, max_active, args.poll_interval, log_file, uspex_cmd)
    log.info("Done.")


if __name__ == "__main__":
    main()
