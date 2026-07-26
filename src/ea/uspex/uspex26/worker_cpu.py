#!/usr/bin/env python3
"""CPU (process-pool) USPEX 26 worker — one relaxation per core.

Counterpart of the batched ``worker.py``.  ``worker.py`` is built for a GPU:
one process evaluates the whole wave of structures in a single ``dp.eval``
per optimizer step, so every structure sits on the same step barrier.  On a
CPU that is the wrong shape — there is no wide device to fill, the barrier
serializes, and cores go idle as structures converge out of the batch.

This worker instead gives **each structure its own process, its own DeepMD
calculator and its own pinned core(s)**, and runs exactly the same
relaxation recipe per structure by calling ``worker.run_full_optimization``
on a batch of one.  So where the GPU worker relaxes 40 structures as one
batch, this one launches 40 independent single-core relaxations (queued
against however many cores are available).

Because relaxations are independent, the pool never idles: as soon as one
structure finishes, its worker process picks up the next pending
CalcFolder — the model stays loaded in that process, so no reload cost.
Results are published per structure (atomically, via ``worker``'s writers)
the moment each relaxation ends, which lets each ``run_batch.py`` stub exit
as early as possible instead of waiting for the slowest structure.

Core selection (``--cores``)::

    --cores 8         use any 8 cores (default: every allowed physical core)
    --cores 0-11      use exactly logical CPUs 0..11
    --cores 0,2,4-6   use exactly logical CPUs 0, 2, 4, 5, 6
    --threads 2       give each relaxation 2 CPUs instead of 1
    --smt             also hand out SMT sibling threads as separate workers

The core inventory is intersected with ``os.sched_getaffinity``, so under
SLURM the worker automatically confines itself to the CPUs the job was
allocated (``--cpus-per-task``) and refuses to pin outside them.  Nothing
here is machine-specific; the same command works on a laptop and in a batch
job.

Modes::

    one-shot (default)  relax every pending CalcFolder, then exit.  Drop-in
                        replacement for worker.py in run_uspex26.py's
                        per-wave loop.
    --watch             stay alive, poll for new CalcFolders and feed them to
                        the warm pool as they appear.  Keeps the cores busy
                        across USPEX generation boundaries and loads each
                        model only once for the whole run.

Usage::

    python worker_cpu.py /path/to/uspex_run --model deepmd_d3 --cores 8
    python worker_cpu.py /path/to/uspex_run --watch --cores 0-11 --threads 1
"""

import argparse
import concurrent.futures
import contextlib
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# Make ``ea.*`` importable when run as a script (and in spawned children,
# which re-import this module by path).  parents[3] == src.
EA_SRC = str(Path(__file__).resolve().parents[3])
if EA_SRC not in sys.path:
    sys.path.insert(0, EA_SRC)

# cpu_layout is stdlib-only, so importing it here pulls in no numeric library.
from ea.utils import cpu_layout


def _early_threads(argv):
    """Read ``--threads`` straight from argv, before argparse exists.

    Only used to size the thread caps below; ``main`` re-parses properly.
    """
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--threads="):
            value = arg.split("=", 1)[1]
        else:
            continue
        try:
            return max(1, int(value))
        except ValueError:
            break
    return 1


# OpenBLAS/OpenMP fix their pool sizes when ``numpy`` is first imported, which
# happens on the ``batch_worker`` import below -- so the caps have to be in
# place BEFORE it, not in ``main``.  Otherwise this parent process alone spins
# one OpenBLAS thread per CPU on the node (measured: 25 threads, 600% CPU
# across every core), competing with the pinned relaxations it supervises.
# Spawned children inherit this environment, so it caps them too.
cpu_layout.set_thread_env(_early_threads(sys.argv))

# Likewise glibc: MALLOC_ARENA_MAX has to be in place before the first
# allocation of a process, so set it here and let the spawned children inherit
# it.  Without it each relaxation's RSS ratchets to ~4x its live working set.
cpu_layout.set_memory_env()

from ea.uspex.models import MODELS, resolve_model
from ea.uspex.uspex26 import worker as batch_worker
from ea.utils.config import load_config

# Seconds a replacement worker process waits for a free CPU chunk before
# giving up on pinning (only reachable if a worker process died).
_LEASE_TIMEOUT = 30

# Per-process state, populated by _init_worker.
_STATE = {}


class PoolBroken(RuntimeError):
    """The relaxation pool failed as a whole; nothing was published."""


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def force_cpu_only():
    """Hide every GPU from this process.

    ``DP(model, device="cpu")`` is NOT enough: deepmd's torch backend resolves
    its global device from ``torch.cuda.is_available()`` at import time and
    happily puts the model on the GPU anyway.  With one process per core that
    means N CUDA contexts fighting over one card — in practice an immediate
    "CUDA out of memory", not a CPU relaxation.  Emptying
    ``CUDA_VISIBLE_DEVICES`` before torch is imported removes the ambiguity
    at the source; :func:`_assert_cpu` then verifies it actually took.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DEVICE"] = "cpu"


def _model_device(calc):
    """Best-effort device of the loaded DeepMD model."""
    for path in (lambda c: next(c.dp.deep_eval.dp.parameters()).device,
                 lambda c: next(c.dp.deep_eval.model.parameters()).device):
        try:
            return path(calc)
        except Exception:
            continue
    return None


def _assert_cpu(calc):
    """Fail loudly if the model landed on a GPU despite force_cpu_only()."""
    device = _model_device(calc)
    if device is not None and device.type != "cpu":
        raise RuntimeError(
            f"DeepMD model loaded on {device}, not the CPU. worker_cpu runs "
            f"one process per core and would open one CUDA context each; "
            f"use worker.py --device cuda for GPU runs."
        )
    return device


def _init_worker(model_key, threads, pin, cpu_queue):
    """Pin this process, then load its own CPU-only DeepMD calculator.

    Runs once per worker process.  Affinity, the OpenMP/MKL/DeePMD thread caps
    and the GPU masking must all be in place before torch/deepmd is imported,
    which is why the calculator is built here and not in the parent.
    """
    try:
        cores = cpu_queue.get(timeout=_LEASE_TIMEOUT)
    except (queue.Empty, OSError):
        cores = []
        print(f"[cpu_worker:{os.getpid()}] no free CPU chunk to lease; "
              f"running unpinned", flush=True)

    pinned, message = cpu_layout.configure_cpu_runtime(cores, threads, pin=pin)
    force_cpu_only()

    # deepmd/torch import happens inside make_calculator, i.e. after the
    # environment above is in place.
    calc = batch_worker.make_calculator(model_key, "cpu")
    device = _assert_cpu(calc)
    try:
        import torch
        # DP() resets the intra-op pool to 1; restore the requested width.
        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass

    _STATE["calc"] = calc
    _STATE["device"] = device
    _STATE["cores"] = cores
    _STATE["threads"] = threads
    print(f"[cpu_worker:{os.getpid()}] ready — {message}, "
          f"{threads} thread(s), model={model_key} on {device or 'cpu'}",
          flush=True)


def _relax_one(task):
    """Relax a single structure and publish its result into the CalcFolder.

    Returns a small summary dict; the relaxed structure is written to disk
    here (not returned) so USPEX sees it as early as possible.
    """
    cf = Path(task["path"])
    mode = task["mode"]
    atoms = task["atoms"]
    out_dir = Path(task["out_dir"])
    calc = _STATE["calc"]

    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "output.traj"
    if stale.is_file():
        stale.unlink()

    log_path = out_dir / "relax.log"
    t0 = time.perf_counter()
    result = {"name": cf.name, "path": str(cf), "pid": os.getpid(),
              "cores": list(_STATE.get("cores") or []), "ok": False,
              "energy": None, "zpe": 0.0, "wall": 0.0, "error": None,
              "log": str(log_path)}

    try:
        with open(log_path, "w") as log:
            # run_full_optimization is chatty; keep each structure's optimizer
            # output in its own file instead of interleaving 12 of them.
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                relaxed, energies = batch_worker.run_full_optimization(
                    [atoms.copy()], calc, out_dir, **task["profile"]
                )
                energy = float(energies[0])
                if task["zpe"]:
                    energy += float(batch_worker.compute_zpe(relaxed, calc)[0])
                    result["zpe"] = energy - float(energies[0])
        batch_worker.write_calcfolder_result(cf, relaxed[0], energy, mode)
        result["ok"] = True
        result["energy"] = energy
    except Exception:
        result["error"] = traceback.format_exc()
        try:
            batch_worker.write_failure(cf, atoms, mode)
        except Exception:
            result["error"] += "\n" + traceback.format_exc()

    result["wall"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Parent: task construction
# ---------------------------------------------------------------------------

def _input_path(cf, mode):
    return Path(cf) / ("input.xyz" if mode == batch_worker.ASE_MODE else "geom.in")


def _is_settled(cf, mode, settle):
    """Whether USPEX has finished writing this CalcFolder's input file."""
    if settle <= 0:
        return True
    try:
        age = time.time() - _input_path(cf, mode).stat().st_mtime
    except OSError:
        return False
    return age >= settle


def build_task(cf, mode, workdir, profile, want_zpe):
    """Read a CalcFolder's input and package it for a worker process."""
    atoms = batch_worker.read_calcfolder_input(cf, mode)
    return {
        "path": str(cf),
        "mode": mode,
        "atoms": atoms,
        "out_dir": str(Path(workdir) / "cpu_out" / Path(cf).name),
        "profile": profile,
        "zpe": want_zpe,
    }


def collect_pending(workdir, submitted, skipped, attempts, settle, profile,
                    want_zpe, max_read_failures=3):
    """Build tasks for pending CalcFolders not already submitted.

    Folders whose input cannot be read are retried on later polls and given
    up on (with a warning) after *max_read_failures* attempts, mirroring
    worker.py's behaviour of skipping unreadable inputs.
    """
    tasks = []
    for _, cf in batch_worker.discover_calcfolders(workdir):
        key = str(cf.resolve())
        if key in submitted or key in skipped:
            continue
        mode = batch_worker.calcfolder_mode(cf)
        if mode is None or not _is_settled(cf, mode, settle):
            continue
        try:
            tasks.append(build_task(cf, mode, workdir, profile, want_zpe))
        except Exception as exc:
            attempts[key] = attempts.get(key, 0) + 1
            print(f"  [warn] {cf.name}: failed to read relaxation input "
                  f"({attempts[key]}/{max_read_failures}): {exc}", flush=True)
            if attempts[key] >= max_read_failures:
                skipped.add(key)
                print(f"  [warn] {cf.name}: giving up on this folder", flush=True)
            continue
        submitted.add(key)
    return tasks


def count_unsettled(workdir, submitted, skipped, settle):
    """Pending folders held back only because their input is still too fresh.

    One-shot mode must wait these out instead of exiting: USPEX has just
    written them and is blocking on their ``output.xyz``.
    """
    n = 0
    for _, cf in batch_worker.discover_calcfolders(workdir):
        key = str(cf.resolve())
        if key in submitted or key in skipped:
            continue
        mode = batch_worker.calcfolder_mode(cf)
        if mode is not None and not _is_settled(cf, mode, settle):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Parent: main loop
# ---------------------------------------------------------------------------

def _report(result, done, total):
    tag = "ok  " if result["ok"] else "FAIL"
    energy = f"{result['energy']:.4f}" if result["energy"] is not None else "n/a"
    extra = f"  zpe={result['zpe']:.4f}" if result["zpe"] else ""
    print(f"[cpu_worker] {tag} {done}/{total} {result['name']}  "
          f"E={energy}{extra}  {result['wall']:.1f}s  "
          f"pid={result['pid']} cores={result['cores']}", flush=True)
    if result["error"]:
        print(f"[cpu_worker] {result['name']} traceback:\n{result['error']}",
              flush=True)
    print(f"[cpu_worker] {result['name']} optimizer log: {result['log']}",
          flush=True)


def run_pool(workdir, chunks, model_key, threads, pin, profile, want_zpe,
             watch=False, poll=3.0, settle=2.0, max_idle=None,
             stop_file=None, stop_flag=None):
    """Relax pending CalcFolders with one process per CPU chunk.

    With *watch* the pool stays alive and keeps polling for new folders;
    otherwise it drains the currently pending set and returns.
    """
    workdir = Path(workdir)
    ctx = mp.get_context("spawn")     # never fork a process that will load torch

    submitted, skipped, attempts = set(), set(), {}
    pending = {}
    done = failed = consecutive_failures = 0
    idle_since = None

    def _should_stop():
        if stop_flag is not None and stop_flag.is_set():
            return "stop signal"
        if stop_file and Path(stop_file).exists():
            return f"stop file {stop_file}"
        return None

    with ctx.Manager() as manager:
        cpu_queue = manager.Queue()
        for chunk in chunks:
            cpu_queue.put(list(chunk))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(chunks),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model_key, threads, pin, cpu_queue),
        ) as pool:
            while True:
                reason = _should_stop()
                if reason and not pending:
                    print(f"[cpu_worker] stopping ({reason})", flush=True)
                    break

                # Rescanning is idempotent (collect_pending skips what it has
                # already submitted), so both modes scan every pass.  One-shot
                # mode needs it too: folders younger than --settle are held
                # back, and a single sweep would drop the whole wave.
                if not reason:
                    tasks = collect_pending(workdir, submitted, skipped,
                                            attempts, settle, profile, want_zpe)
                    for task in tasks:
                        pending[pool.submit(_relax_one, task)] = task
                    if tasks:
                        print(f"[cpu_worker] queued {len(tasks)} structure(s) "
                              f"({len(pending)} in flight, "
                              f"{len(chunks)} worker(s))", flush=True)

                if not pending:
                    if not watch:
                        # Nothing in flight: only keep going if USPEX is still
                        # mid-write on a folder we have not accepted yet.
                        waiting = count_unsettled(workdir, submitted, skipped,
                                                  settle)
                        if not waiting:
                            break
                        print(f"[cpu_worker] waiting for {waiting} folder(s) to "
                              f"settle ({settle}s)", flush=True)
                        time.sleep(poll)
                        continue
                    if max_idle is not None:
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since >= max_idle:
                            print(f"[cpu_worker] idle for {max_idle}s — exiting",
                                  flush=True)
                            break
                    time.sleep(poll)
                    continue

                idle_since = None
                # Block only as long as the poll interval so newly created
                # folders are picked up while long relaxations still run.
                finished, _ = concurrent.futures.wait(
                    list(pending),
                    timeout=poll if watch else None,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                broken = None
                for future in finished:
                    task = pending.pop(future)
                    try:
                        result = future.result()
                    except concurrent.futures.BrokenExecutor:
                        # Pool-wide failure (a worker was killed, or the
                        # initializer raised). Publishing energy-0 fallbacks
                        # here would poison the whole search with structures
                        # USPEX believes were relaxed, so leave the folders
                        # pending and stop.
                        broken = traceback.format_exc()
                        break
                    except Exception:
                        result = {"name": Path(task["path"]).name,
                                  "path": task["path"], "pid": None, "cores": [],
                                  "ok": False, "energy": None, "zpe": 0.0,
                                  "wall": 0.0, "log": task["out_dir"],
                                  "error": traceback.format_exc()}
                        try:
                            batch_worker.write_failure(Path(task["path"]),
                                                       task["atoms"],
                                                       task["mode"])
                        except Exception:
                            result["error"] += "\n" + traceback.format_exc()
                    if result["ok"]:
                        done += 1
                        consecutive_failures = 0
                    else:
                        failed += 1
                        consecutive_failures += 1
                    _report(result, done + failed, len(submitted))

                if broken:
                    raise PoolBroken(
                        f"the worker pool died and {len(pending) + 1} "
                        f"structure(s) were left unrelaxed (their CalcFolders "
                        f"stay pending, nothing was published):\n{broken}"
                    )
                # A fresh pool that fails every structure is an environment
                # problem (bad model, no memory), not bad chemistry: stop
                # instead of filling the search with energy-0 structures.
                if done == 0 and consecutive_failures >= len(chunks):
                    raise PoolBroken(
                        f"the first {consecutive_failures} relaxation(s) all "
                        f"failed and none succeeded — check the tracebacks "
                        f"above before continuing"
                    )

    return done, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("workdir", type=Path,
                   help="USPEX working directory (holds Calculation/Calcfold_* "
                        "or legacy CalcFold[N])")
    p.add_argument("--model", default="deepmd_d3", choices=sorted(MODELS),
                   help="DeepMD model key from the config's deepmd.models_path")
    p.add_argument("--cores", default=None,
                   help="How many cores, or exactly which ones: '8' (a count), "
                        "'0-11' (a range) or '0,2,4-6' (a list). "
                        "Default: every allowed physical core")
    p.add_argument("--threads", type=int, default=1,
                   help="CPUs/compute threads per relaxation (default: 1)")
    p.add_argument("--workers", type=int, default=None,
                   help="Cap on concurrent relaxations "
                        "(default: as many as the cores allow)")
    p.add_argument("--smt", action="store_true",
                   help="Also use SMT sibling threads as separate workers")
    p.add_argument("--oversubscribe", action="store_true",
                   help="Allow --workers to exceed the available cores "
                        "(workers then share CPUs)")
    p.add_argument("--no-pin", action="store_true",
                   help="Do not set CPU affinity; only cap thread pools "
                        "(use when the scheduler already pins tasks)")
    p.add_argument("--mem-per-worker-gb", type=float, default=3.5,
                   help="RAM one relaxation needs, used to clamp the worker "
                        "count so the node does not swap. Default 3.5 is the "
                        "measured peak of one force+stress evaluation of a "
                        "42-atom molecular cell at the ~400 A^3 target with "
                        "DPA3 (rcut 6 A). Resident memory is only ~1.3 GiB; "
                        "the transient backward-pass peak is what matters, and "
                        "it grows with neighbours-per-atom: ~2.1 GiB at 700 "
                        "A^3, ~3.5 at 400, ~4.8 at 246. Raise it for dense "
                        "cells. 0 disables the check")
    p.add_argument("--reserve-gb", type=float, default=2.0,
                   help="RAM held back for USPEX itself and the OS when "
                        "applying --mem-per-worker-gb (default: 2.0)")
    p.add_argument("--zpe", action="store_true",
                   help="Add Gamma-point ZPE to each final energy")
    p.add_argument("--watch", action="store_true",
                   help="Stay alive and keep relaxing new CalcFolders as USPEX "
                        "creates them (keeps the model loaded and the cores busy)")
    p.add_argument("--poll", type=float, default=3.0,
                   help="Seconds between CalcFolder scans in --watch mode")
    p.add_argument("--settle", type=float, default=2.0,
                   help="Require an input file to be this many seconds old "
                        "before reading it (0 disables)")
    p.add_argument("--max-idle", type=float, default=None,
                   help="In --watch mode, exit after this many seconds with "
                        "nothing to do")
    p.add_argument("--stop-file", default=None,
                   help="In --watch mode, exit once this file exists")
    p.add_argument("--smoke", action="store_true",
                   help="Abbreviated FIRE/LBFGS profile for integration smoke "
                        "tests; not a production relaxation")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        sys.exit(f"[cpu_worker] workdir not found: {workdir}")

    # Resolve the model in the parent: a missing file would otherwise only
    # surface inside every worker's initializer, breaking the pool and
    # publishing energy-0 fallbacks for the whole wave.
    try:
        model_path = resolve_model(load_config()["deepmd"]["models_path"],
                                   args.model)
    except (KeyError, FileNotFoundError) as exc:
        sys.exit(f"[cpu_worker] {exc}")
    print(f"[cpu_worker] model {args.model}: {model_path}", flush=True)

    # ---- CPU inventory ----------------------------------------------------
    try:
        pool_cpus, notes = cpu_layout.select_cpus(args.cores, use_smt=args.smt)
        chunks, more = cpu_layout.chunk_cpus(
            pool_cpus,
            threads_per_worker=args.threads,
            max_workers=args.workers,
            oversubscribe=args.oversubscribe,
        )
        chunks, mem_notes = cpu_layout.clamp_workers_by_memory(
            chunks, args.mem_per_worker_gb, args.reserve_gb
        )
        more += mem_notes
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"[cpu_worker] {exc}")

    print(cpu_layout.describe(chunks, args.threads), flush=True)
    for note in notes + more:
        print(f"[cpu_worker] note: {note}", flush=True)

    # Both inherited by the spawned children, so OpenBLAS/numpy are capped and
    # the GPU is hidden by the time they import (before any initializer runs).
    cpu_layout.set_thread_env(args.threads)
    force_cpu_only()

    # ---- relaxation profile ----------------------------------------------
    profile = {}
    if args.smoke:
        profile = {"fire_steps": 2, "lbfgs_steps": 2, "lbfgs_stages": (0.03,)}
        print("[cpu_worker] SMOKE profile: abbreviated relaxation", flush=True)

    # ---- molecular interface guard (same rule as worker.py) --------------
    if list(workdir.glob("MOL_*")):
        legacy = [cf for _, cf in batch_worker.discover_calcfolders(workdir)
                  if batch_worker.calcfolder_mode(cf) == batch_worker.USER_CODE_MODE]
        if legacy:
            sys.exit(
                "[cpu_worker] molecular MOL_* input detected with legacy USPEX "
                "USER_CODE/code 99. Use abinitioCode 20 so extended XYZ "
                "preserves molecular atom order."
            )

    stop_flag = _install_signal_handlers()

    if not args.watch and not batch_worker.discover_calcfolders(workdir):
        print(f"[cpu_worker] no pending CalcFolders under {workdir}; "
              f"nothing to do", flush=True)
        return 0

    t0 = time.perf_counter()
    try:
        done, failed = run_pool(
            workdir, chunks, args.model, args.threads, not args.no_pin,
            profile, args.zpe,
            watch=args.watch, poll=args.poll, settle=args.settle,
            max_idle=args.max_idle, stop_file=args.stop_file,
            stop_flag=stop_flag,
        )
    except PoolBroken as exc:
        print(f"[cpu_worker] ABORTING — {exc}", flush=True)
        return 1
    print(f"[cpu_worker] done — relaxed={done}  fallback={failed}  "
          f"wall={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


def _install_signal_handlers():
    flag = threading.Event()

    def _handler(signum, _frame):
        print(f"[cpu_worker] signal {signum} — finishing in-flight "
              f"relaxations, then exiting", flush=True)
        flag.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)
    return flag


if __name__ == "__main__":
    sys.exit(main())
