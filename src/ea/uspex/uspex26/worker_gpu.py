#!/usr/bin/env python3
"""GPU process-pool worker for USPEX 26.

This is the GPU counterpart of :mod:`worker_cpu`.  Each process owns one
DeepMD calculator and one CUDA device, then pulls independent CalcFolders from
the shared process pool.  Keeping one model loaded per GPU avoids the
single-device batch barrier in ``worker.py`` while still allowing USPEX to
create new relaxation folders continuously.

GPU IDs are the nvidia-smi / nvtop IDs (PCI bus order).  A worker masks itself with
``CUDA_VISIBLE_DEVICES=<id>`` before importing DeepMD, so the selected host
GPU appears as device 0 inside that worker.  Example::

    python worker_gpu.py /path/to/uspex_run --gpu-ids 0,1,2 --watch

The default is one-shot mode: drain the currently pending CalcFolders and
exit.  ``--watch`` keeps the pool alive for the duration of a USPEX search.
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

# Make ``ea.*`` importable when run as a script.  This must happen before the
# worker modules are imported, including in spawned children.
EA_SRC = str(Path(__file__).resolve().parents[3])
if EA_SRC not in sys.path:
    sys.path.insert(0, EA_SRC)

from ea.utils import cpu_layout


def _early_threads(argv):
    """Read ``--threads`` before importing numpy/ASE in spawned children."""
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
    return 2


# BLAS/OpenMP and glibc read these settings early.  The parent sets them so
# spawned children inherit them before importing torch/deepmd.
cpu_layout.set_thread_env(_early_threads(sys.argv))
cpu_layout.set_memory_env()

from ea.uspex.models import MODELS
from ea.uspex.uspex26 import worker as batch_worker


_GPU_LEASE_TIMEOUT = 30
_STATE = {}


class PoolBroken(RuntimeError):
    """The relaxation pool failed as a whole; nothing was safely published."""


def parse_gpu_ids(spec):
    """Parse ``0,1,2`` or ``0-2,5`` into ordered, unique GPU IDs."""
    if spec is None:
        raise ValueError("GPU IDs are required; use --gpu-ids 0,1,2")

    ids = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, sep, hi_text = part.partition("-")
            if not sep or not lo_text or not hi_text:
                raise ValueError(
                    f"cannot parse GPU ID spec {spec!r}; expected '0,1,2' "
                    "or '0-2'"
                )
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError as exc:
                raise ValueError(f"invalid GPU ID spec {spec!r}") from exc
            if lo < 0 or hi < 0 or hi < lo:
                raise ValueError(f"invalid GPU range {part!r}")
            ids.extend(range(lo, hi + 1))
        else:
            try:
                gpu_id = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid GPU ID {part!r}") from exc
            if gpu_id < 0:
                raise ValueError(f"GPU IDs must be non-negative: {gpu_id}")
            ids.append(gpu_id)

    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError(f"GPU ID spec {spec!r} selects no GPUs")
    return ids


def _model_device(calc):
    """Best-effort device of the loaded DeepMD model for diagnostics."""
    for path in (
        lambda c: next(c.dp.deep_eval.dp.parameters()).device,
        lambda c: next(c.dp.deep_eval.model.parameters()).device,
    ):
        try:
            return path(calc)
        except Exception:
            continue
    return None


def _init_worker(model_stages, threads, gpu_queue):
    """Assign one GPU and load its calculator exactly once in this process."""
    try:
        gpu_id = int(gpu_queue.get(timeout=_GPU_LEASE_TIMEOUT))
    except (queue.Empty, OSError):
        raise RuntimeError("no GPU ID available to lease")

    # This must precede the first import of torch/deepmd.  Because this module
    # only imports the ASE-facing worker above, the CUDA-dependent import still
    # happens here for the first time in make_calculator().
    # CUDA numbers devices fastest-first by default, which on mixed nodes is
    # not the nvidia-smi/nvtop numbering users pick IDs from; pin PCI order.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["DEVICE"] = "gpu"
    cpu_layout.set_thread_env(threads)

    calc = batch_worker.make_calculators(model_stages, "gpu")
    calcs = [c for _, c in calc] if isinstance(calc, list) else [calc]
    for stage_calc in calcs:
        device = _model_device(stage_calc)
        if device is not None and getattr(device, "type", None) != "cuda":
            raise RuntimeError(
                f"DeepMD model for host GPU {gpu_id} landed on {device}, "
                "not CUDA"
            )
    models = ", ".join(model for _, model in model_stages)

    try:
        import torch

        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass

    _STATE.update(calc=calc, gpu_id=gpu_id, device=device, threads=threads)
    print(
        f"[gpu_worker:{os.getpid()}] ready — host GPU {gpu_id} "
        f"(visible as CUDA:0), {threads} thread(s), "
        f"model={models} on {device or 'cuda'}",
        flush=True,
    )


def _relax_one(task):
    """Relax and publish one CalcFolder from the assigned GPU process."""
    cf = Path(task["path"])
    mode = task["mode"]
    atoms = task["atoms"]
    out_dir = Path(task["out_dir"])
    calc = _STATE["calc"]
    gpu_id = _STATE["gpu_id"]

    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "output.traj"
    if stale.is_file():
        stale.unlink()

    log_path = out_dir / "relax.log"
    t0 = time.perf_counter()
    result = {
        "name": cf.name,
        "path": str(cf),
        "pid": os.getpid(),
        "gpu_id": gpu_id,
        "ok": False,
        "energy": None,
        "zpe": 0.0,
        "wall": 0.0,
        "error": None,
        "log": str(log_path),
    }

    try:
        with open(log_path, "w") as log:
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


def _input_path(cf, mode):
    return Path(cf) / (
        "input.xyz" if mode == batch_worker.ASE_MODE else "geom.in"
    )


def _is_settled(cf, mode, settle):
    if settle <= 0:
        return True
    try:
        age = time.time() - _input_path(cf, mode).stat().st_mtime
    except OSError:
        return False
    return age >= settle


def build_task(cf, mode, workdir, profile, want_zpe):
    atoms = batch_worker.read_calcfolder_input(cf, mode)
    return {
        "path": str(cf),
        "mode": mode,
        "atoms": atoms,
        "out_dir": str(Path(workdir) / "gpu_out" / Path(cf).name),
        "profile": profile,
        "zpe": want_zpe,
    }


def collect_pending(workdir, submitted, skipped, attempts, settle, profile,
                    want_zpe, max_read_failures=3):
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
            print(
                f"  [warn] {cf.name}: failed to read relaxation input "
                f"({attempts[key]}/{max_read_failures}): {exc}",
                flush=True,
            )
            if attempts[key] >= max_read_failures:
                skipped.add(key)
                print(f"  [warn] {cf.name}: giving up on this folder", flush=True)
            continue
        submitted.add(key)
    return tasks


def count_unsettled(workdir, submitted, skipped, settle):
    n = 0
    for _, cf in batch_worker.discover_calcfolders(workdir):
        key = str(cf.resolve())
        if key in submitted or key in skipped:
            continue
        mode = batch_worker.calcfolder_mode(cf)
        if mode is not None and not _is_settled(cf, mode, settle):
            n += 1
    return n


def _report(result, done, total):
    tag = "ok  " if result["ok"] else "FAIL"
    energy = (
        f"{result['energy']:.4f}" if result["energy"] is not None else "n/a"
    )
    extra = f"  zpe={result['zpe']:.4f}" if result["zpe"] else ""
    print(
        f"[gpu_worker] {tag} {done}/{total} {result['name']}  "
        f"E={energy}{extra}  {result['wall']:.1f}s  "
        f"pid={result['pid']} gpu={result['gpu_id']}",
        flush=True,
    )
    if result["error"]:
        print(
            f"[gpu_worker] {result['name']} traceback:\n{result['error']}",
            flush=True,
        )
    print(
        f"[gpu_worker] {result['name']} optimizer log: {result['log']}",
        flush=True,
    )


def run_pool(workdir, gpu_ids, model_stages, threads, profile, want_zpe,
             watch=False, poll=3.0, settle=2.0, max_idle=None,
             stop_file=None, stop_flag=None):
    """Relax pending CalcFolders with one persistent process per GPU."""
    workdir = Path(workdir)
    ctx = mp.get_context("spawn")
    submitted, skipped, attempts = set(), set(), {}
    pending = {}
    done = failed = consecutive_failures = 0
    idle_since = None

    def should_stop():
        if stop_flag is not None and stop_flag.is_set():
            return "stop signal"
        if stop_file and Path(stop_file).exists():
            return f"stop file {stop_file}"
        return None

    with ctx.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(gpu_ids),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model_stages, threads, gpu_queue),
        ) as pool:
            while True:
                reason = should_stop()
                if reason and not pending:
                    print(f"[gpu_worker] stopping ({reason})", flush=True)
                    break

                if not reason:
                    tasks = collect_pending(
                        workdir, submitted, skipped, attempts, settle,
                        profile, want_zpe,
                    )
                    for task in tasks:
                        pending[pool.submit(_relax_one, task)] = task
                    if tasks:
                        print(
                            f"[gpu_worker] queued {len(tasks)} structure(s) "
                            f"({len(pending)} in flight, {len(gpu_ids)} GPU(s))",
                            flush=True,
                        )

                if not pending:
                    if not watch:
                        waiting = count_unsettled(
                            workdir, submitted, skipped, settle
                        )
                        if not waiting:
                            break
                        print(
                            f"[gpu_worker] waiting for {waiting} folder(s) to "
                            f"settle ({settle}s)",
                            flush=True,
                        )
                        time.sleep(poll)
                        continue
                    if max_idle is not None:
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since >= max_idle:
                            print(
                                f"[gpu_worker] idle for {max_idle}s — exiting",
                                flush=True,
                            )
                            break
                    time.sleep(poll)
                    continue

                idle_since = None
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
                        broken = traceback.format_exc()
                        break
                    except Exception:
                        result = {
                            "name": Path(task["path"]).name,
                            "path": task["path"],
                            "pid": None,
                            "gpu_id": "?",
                            "ok": False,
                            "energy": None,
                            "zpe": 0.0,
                            "wall": 0.0,
                            "log": task["out_dir"],
                            "error": traceback.format_exc(),
                        }
                        try:
                            batch_worker.write_failure(
                                Path(task["path"]), task["atoms"], task["mode"]
                            )
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
                        f"the GPU worker pool died and {len(pending) + 1} "
                        f"structure(s) were left unrelaxed (their CalcFolders "
                        f"stay pending, nothing was published):\n{broken}"
                    )
                if done == 0 and consecutive_failures >= len(gpu_ids):
                    raise PoolBroken(
                        f"the first {consecutive_failures} relaxation(s) all "
                        "failed and none succeeded — check the tracebacks above"
                    )

    return done, failed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workdir", type=Path,
                   help="USPEX working directory containing CalcFolders")
    p.add_argument("--gpu-ids", "--gpus", default="0",
                   help="Host GPU IDs, e.g. '0,1,2' or '0-2' (default: 0)")
    p.add_argument("--model", default="deepmd_d3", choices=sorted(MODELS),
                   help="DeepMD model key from the config; ignored when "
                        "INPUT.txt has a MODELS section")
    p.add_argument("--threads", type=int, default=2,
                   help="CPU threads per GPU worker (default: 2)")
    p.add_argument("--zpe", action="store_true",
                   help="Add Gamma-point ZPE to each final energy")
    p.add_argument("--watch", action="store_true",
                   help="Stay alive and process CalcFolders created later")
    p.add_argument("--poll", type=float, default=3.0,
                   help="Seconds between CalcFolder scans in --watch mode")
    p.add_argument("--settle", type=float, default=2.0,
                   help="Require an input file to be this old before reading")
    p.add_argument("--max-idle", type=float, default=None,
                   help="In --watch mode, exit after this many idle seconds")
    p.add_argument("--stop-file", default=None,
                   help="In --watch mode, finish in-flight work then exit")
    p.add_argument("--smoke", action="store_true",
                   help="Use an abbreviated FIRE/LBFGS profile")
    return p


def _install_signal_handlers():
    flag = threading.Event()

    def handler(signum, _frame):
        print(
            f"[gpu_worker] signal {signum} — finishing in-flight relaxations",
            flush=True,
        )
        flag.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    return flag


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        sys.exit("[gpu_worker] --threads must be >= 1")

    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except ValueError as exc:
        sys.exit(f"[gpu_worker] {exc}")

    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        sys.exit(f"[gpu_worker] workdir not found: {workdir}")

    try:
        model_stages = batch_worker.load_model_stages(workdir, args.model)
        for line in batch_worker.describe_model_stages(model_stages):
            print(f"[gpu_worker] model {line}", flush=True)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        sys.exit(f"[gpu_worker] {exc}")
    print(f"[gpu_worker] host GPUs: {gpu_ids}", flush=True)

    if list(workdir.glob("MOL_*")):
        legacy = [
            cf for _, cf in batch_worker.discover_calcfolders(workdir)
            if batch_worker.calcfolder_mode(cf) == batch_worker.USER_CODE_MODE
        ]
        if legacy:
            sys.exit(
                "[gpu_worker] molecular MOL_* input detected with legacy "
                "USPEX USER_CODE/code 99. Use abinitioCode 20 so extended "
                "XYZ preserves molecular atom order."
            )

    cpu_layout.set_thread_env(args.threads)
    profile = {}
    if args.smoke:
        profile = {"fire_steps": 2, "lbfgs_steps": 2,
                   "lbfgs_stages": (0.03,)}
        print("[gpu_worker] SMOKE profile: abbreviated relaxation", flush=True)

    stop_flag = _install_signal_handlers()
    if not args.watch and not batch_worker.discover_calcfolders(workdir):
        print(
            f"[gpu_worker] no pending CalcFolders under {workdir}; nothing to do",
            flush=True,
        )
        return 0

    t0 = time.perf_counter()
    try:
        done, failed = run_pool(
            workdir, gpu_ids, model_stages, args.threads, profile, args.zpe,
            watch=args.watch, poll=args.poll, settle=args.settle,
            max_idle=args.max_idle, stop_file=args.stop_file,
            stop_flag=stop_flag,
        )
    except PoolBroken as exc:
        print(f"[gpu_worker] ABORTING — {exc}", flush=True)
        return 1
    print(
        f"[gpu_worker] done — relaxed={done} fallback={failed} "
        f"wall={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
