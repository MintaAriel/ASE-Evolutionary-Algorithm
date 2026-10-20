#!/usr/bin/env python3
"""Memory-aware CPU USPEX 26 worker — admission control instead of a fixed pool.

Counterpart of ``worker_cpu.py``.  Both give each structure its own process,
its own DeepMD calculator and its own pinned core(s); they differ in *how
many run at once*.

``worker_cpu.py`` decides that once, up front::

    workers = min(cpu_chunks, (allocation - reserve) // mem_per_worker_gb)

which is only as good as the ``mem_per_worker_gb`` guess.  Guess low and the
guard silently does nothing: a 200 GiB allocation with ``--mem-per-worker-gb
4.5`` reports "room for 44 worker(s), 40 requested" and lets all 40 through,
while the real per-relaxation peak is ~9 GiB.  The measured result on a
240 GiB node was 180 GB resident across 40 workers, the page cache reduced to
875 MB, and — because the node is diskless and ``slurmd`` lives on NFS —
``slurmd`` blocked long enough on major faults for the controller to declare
NODE_FAIL and requeue the job.  Four times.

This worker never guesses.  It admits one structure at a time and only while
the *measured* footprint says there is room::

    while jobs_remaining:
        current = sum(VmRSS of live workers)
        if current + est_peak <= MAX_RAM_GB and node_free >= MIN_NODE_FREE_GB:
            launch_next_structure()
        else:
            sleep(CHECK_INTERVAL)
        reap_finished_workers()

Three things make that safe in practice, and they are the whole design:

**One process per structure, exiting when done.**  ``worker_cpu.py`` keeps a
warm ``ProcessPoolExecutor``: a worker holds its model (and glibc's retained
heap) between structures, so finishing a relaxation frees almost nothing and
the pool's footprint only ratchets up.  Here a worker *exits* after
publishing, returning every page to the OS, which is what makes "current RAM"
a number worth scheduling against.  The cost is one model load per structure;
measured relaxations in this configuration run 3.3-17.9 h (median 6.7 h) on
one core, so a ~10-30 s load is ~0.1% overhead.  On a workload of short
relaxations, prefer ``worker_cpu.py``.

**Workers that have not ramped yet are charged in full.**  RSS lags launch: a
worker that has just started is a few hundred MB while it loads the model,
then climbs to several GiB.  Admitting purely on "current < MAX" lets a whole
cohort in on the strength of memory none of them has claimed yet, and they
then peak together — measured in a 6-structure test, all 6 were admitted
against a 20 GiB budget at a 6 GiB/worker estimate (6x6 = 36) purely because
none of them had loaded yet.  So for its first ``--ramp-seconds`` (default
180) a worker counts as ``est_peak``, not as what /proc currently says; only
after that is its real RSS trusted.  ``--admit-per-cycle`` (default 1 per
``--check-interval``) further paces the ramp.  Reaching 30 workers then takes
minutes, which against multi-hour relaxations is free.

**The estimate scales with ``len(atoms)``, and is learned.**  Memory here is
not a per-worker constant.  A DeepMD force+stress evaluation holds per-atom
and per-neighbour-pair activations, so the transient backward-pass peak grows
with the cell: this search runs Z = 2,4,6,8 of a 21-atom molecule (42, 84,
126, 168 atoms, in roughly 4:14:11:11 proportion) and the measured peaks span
3.5 GiB at 42 atoms to 7.9 GiB at 168.  Charging every worker one flat number
is therefore wrong in both directions — size it for 168 and most of the node
sits idle behind Z=2 cells, size it for 42 and a Z=8-heavy wave overruns.
:class:`PeakModel` predicts ``base + per_atom * natoms`` instead, seeded from
those two measurements and refit by least squares once 3+ distinct sizes have
been seen.  Each child reports its own ``VmHWM`` on exit, and the worst peak
ever seen at a given size becomes a hard floor for that size.

Because estimates are per-structure, the queue is scanned with a lookahead: if
the head is a 168-atom cell that will not fit, a 42-atom one further back may,
and strict FIFO would idle cores for no reason.  ``--head-priority-seconds``
bounds that — once the head has waited long enough it is taken in order, so
large cells cannot be starved by a run of small ones.

Note that per-worker peaks are *staggered*, not synchronised: 40 workers with
an 8.9 GiB individual peak summed to 180 GB, not 356 GB.  That is why this
schedules on the measured sum rather than on ``n * est_peak`` — the latter is
the static clamp again, and it would leave half the node idle.

``--min-node-free-gb`` is the guard that addresses the failure above
directly: it watches ``MemAvailable`` for the *whole node*, not just this
job's allocation, and stops admitting when the node's page cache is getting
squeezed — regardless of what the job's own accounting says.

Usage::

    python worker_cpu_dynamic.py /path/to/uspex_run --model deepmd_d3 \
        --watch --max-ram-gb 180 --cores 30
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import queue as queue_mod
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Make ``ea.*`` importable when run as a script (and in spawned children,
# which re-import this module by path).  parents[3] == src.
EA_SRC = str(Path(__file__).resolve().parents[3])
if EA_SRC not in sys.path:
    sys.path.insert(0, EA_SRC)

# Importing worker_cpu also installs the OpenBLAS/OpenMP thread caps and
# MALLOC_ARENA_MAX at module scope, which must happen before numpy/torch are
# first imported.  Both modules read ``--threads`` out of argv the same way,
# so the caps are already correct for us.
from ea.uspex.uspex26 import worker_cpu
from ea.uspex.uspex26 import worker as batch_worker
from ea.uspex.models import MODELS, resolve_model
from ea.utils import cpu_layout
from ea.utils.config import load_config

KIB_PER_GIB = 1024 * 1024


# ---------------------------------------------------------------------------
# Memory probes
# ---------------------------------------------------------------------------

def _status_field_gb(pid, field):
    """One ``/proc/<pid>/status`` size field, in GiB (``None`` if unreadable).

    A worker that exits between the ``is_alive`` check and this read is
    normal, not an error — hence the quiet ``None``.
    """
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1]) / KIB_PER_GIB
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_gb(pid):
    """Resident set size of *pid* in GiB."""
    return _status_field_gb(pid, "VmRSS:")


def process_peak_gb(pid):
    """High-water resident size of *pid* in GiB."""
    return _status_field_gb(pid, "VmHWM:")


def node_available_gb():
    """``MemAvailable`` for the whole node, in GiB (``None`` if unreadable).

    Deliberately the node figure and not the cgroup's: the failure this
    guards against is the node's page cache being reclaimed out from under
    ``slurmd``, which the job's own allocation accounting cannot see.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / KIB_PER_GIB
    except (OSError, ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def _child_entry(task, model_stages, threads, pin, cores, result_q):
    """Relax exactly one structure, publish it, report the peak, exit.

    Runs in a spawned process.  Affinity, thread caps and GPU masking have to
    be in place before torch/deepmd is imported, which is why the calculator
    is built here rather than passed in.  Reuses ``worker_cpu``'s
    ``_relax_one`` so both workers run byte-identical relaxations.
    """
    result = None
    try:
        pinned, message = cpu_layout.configure_cpu_runtime(cores, threads,
                                                           pin=pin)
        worker_cpu.force_cpu_only()

        calc = batch_worker.make_calculators(model_stages, "cpu")
        device = worker_cpu.assert_cpu_stages(calc)
        try:
            import torch
            # DP() resets the intra-op pool to 1; restore the requested width.
            torch.set_num_threads(threads)
            with contextlib.suppress(RuntimeError):
                torch.set_num_interop_threads(1)
        except ImportError:
            pass

        worker_cpu._STATE["calc"] = calc
        worker_cpu._STATE["device"] = device
        worker_cpu._STATE["cores"] = cores
        worker_cpu._STATE["threads"] = threads
        print(f"[cpu_dyn:{os.getpid()}] start {Path(task['path']).name} — "
              f"{message}, {threads} thread(s), "
              f"model={worker_cpu.describe_models(model_stages)} "
              f"on {device or 'cpu'}", flush=True)

        result = worker_cpu._relax_one(task)
    except Exception:
        # An initialization failure (missing model, no memory) must not look
        # like a relaxation that merely returned a bad energy.
        result = {"name": Path(task["path"]).name, "path": task["path"],
                  "pid": os.getpid(), "cores": list(cores), "ok": False,
                  "energy": None, "zpe": 0.0, "wall": 0.0,
                  "log": task["out_dir"], "error": traceback.format_exc()}
    finally:
        if result is not None:
            # Read our own high-water mark rather than let the parent sample
            # for it: the parent polls every few seconds and would routinely
            # miss a transient backward-pass peak.
            result["peak_gb"] = process_peak_gb(os.getpid())
            with contextlib.suppress(Exception):
                result_q.put(result)


# ---------------------------------------------------------------------------
# Parent bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class LiveWorker:
    """One in-flight relaxation process."""
    proc: object
    task: dict
    cores: list
    started: float
    rss_gb: float = 0.0
    peak_gb: float = 0.0

    @property
    def name(self):
        return Path(self.task["path"]).name

    @property
    def natoms(self):
        return self.task.get("natoms") or 0

    def sample(self):
        """Refresh RSS/peak from /proc; returns the RSS actually seen."""
        rss = process_rss_gb(self.proc.pid)
        if rss is not None:
            self.rss_gb = rss
            self.peak_gb = max(self.peak_gb, rss)
        return self.rss_gb


class PeakModel:
    """Predicts a relaxation's peak RSS from its atom count.

    Memory here is not a per-worker constant: a DeepMD force+stress evaluation
    holds per-atom and per-neighbour-pair activations, so the transient
    backward-pass peak scales with ``len(atoms)``.  This search runs Z = 2,4,6,8
    of a 21-atom molecule — 42 to 168 atoms — and the documented production
    peaks span 3.5 GiB at 42 atoms to 7.9 GiB at 168.  Charging one flat
    figure therefore either strands most of the node (if it is sized for 168)
    or oversubscribes it (if it is sized for 42).

    The model is affine, ``base + per_atom * natoms``, seeded from those
    measurements and refit by least squares once enough distinct sizes have
    been observed.  Per-size observed maxima are kept as hard floors: a size
    that has ever peaked at X is never again estimated below X.
    """

    # Enough distinct sizes to make a slope meaningful rather than noise.
    MIN_SIZES_TO_FIT = 3
    MIN_SAMPLES_TO_FIT = 6

    def __init__(self, base_gb, per_atom_gb, floor_gb, safety=1.10):
        self.seed_base = base_gb
        self.seed_per_atom = per_atom_gb
        self.base_gb = base_gb
        self.per_atom_gb = per_atom_gb
        self.floor_gb = max(0.25, floor_gb)
        self.safety = safety
        self.observed = {}          # natoms -> worst peak GiB seen
        self.samples = 0
        self.fitted = False

    def estimate(self, natoms):
        """Predicted peak RSS in GiB for a structure of *natoms* atoms."""
        if not natoms:
            # Unknown size: fall back to the worst thing we have ever seen,
            # since guessing small is the failure that costs a node.
            return max(self.floor_gb, max(self.observed.values(), default=0.0))
        line = self.base_gb + self.per_atom_gb * natoms
        seen = self.observed.get(natoms, 0.0)
        return max(self.floor_gb, line, seen)

    def observe(self, natoms, peak_gb):
        """Fold one finished worker's true VmHWM into the model."""
        if not peak_gb or not natoms:
            return
        self.samples += 1
        # Worst case per size, not the mean: admission has to survive the
        # heaviest structure of that size, not the typical one.
        self.observed[natoms] = max(self.observed.get(natoms, 0.0), peak_gb)
        self._refit()

    def _refit(self):
        """Least-squares affine fit over the per-size maxima."""
        points = sorted(self.observed.items())
        if (len(points) < self.MIN_SIZES_TO_FIT
                or self.samples < self.MIN_SAMPLES_TO_FIT):
            return
        n = len(points)
        sx = sum(x for x, _ in points)
        sy = sum(y for _, y in points)
        sxx = sum(x * x for x, _ in points)
        sxy = sum(x * y for x, y in points)
        denom = n * sxx - sx * sx
        if denom <= 0:
            return
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        # A negative slope means the data disagree with the physics (noise, or
        # one huge outlier at a small size); keep the seed rather than learn
        # something that would under-charge the big cells.
        if slope <= 0:
            return
        self.base_gb = max(0.0, intercept) * self.safety
        self.per_atom_gb = slope * self.safety
        self.fitted = True

    def describe(self):
        src = "fitted" if self.fitted else "seed"
        sizes = ", ".join(f"{n}:{p:.1f}" for n, p in sorted(self.observed.items()))
        return (f"peak model ({src}): {self.base_gb:.2f} + "
                f"{self.per_atom_gb:.4f}/atom GiB, floor {self.floor_gb:.1f}"
                + (f" | observed {sizes}" if sizes else ""))


class MemoryScheduler:
    """Decides whether one more relaxation fits right now.

    Holds the size-aware :class:`PeakModel` and the observed high-water of the
    *sum* of resident sizes, which is what actually has to stay under budget.
    """

    def __init__(self, max_ram_gb, peaks, min_node_free_gb):
        self.max_ram_gb = max_ram_gb
        self.peaks = peaks
        self.min_node_free_gb = min_node_free_gb
        self.high_water_gb = 0.0

    def observe_peak(self, natoms, peak_gb):
        self.peaks.observe(natoms, peak_gb)

    def current_gb(self, live):
        total = sum(w.sample() for w in live)
        self.high_water_gb = max(self.high_water_gb, total)
        return total

    def projected_gb(self, live, now, ramp_seconds):
        """Budget-facing total: measured, but with ramping workers charged full.

        A worker that started seconds ago is still loading its model and reads
        as a few hundred MB, so scheduling on raw RSS silently under-counts
        every launch that has not peaked yet — admit fast enough and a whole
        cohort gets in on the strength of memory none of them has claimed yet.
        Until a worker is *ramp_seconds* old it is therefore charged at its
        own size's predicted peak; after that its real RSS is trusted, which
        is what lets a wave of small Z=2 cells run wider than a static clamp
        ever would.
        """
        total = 0.0
        for w in live:
            rss = w.rss_gb
            if now - w.started < ramp_seconds:
                rss = max(rss, self.peaks.estimate(w.natoms))
            total += rss
        return total

    def verdict(self, live, projected_gb, natoms):
        """``(admit, reason)`` for one more worker of *natoms* atoms."""
        # Never deadlock: something must run, even if the estimate says no.
        if not live:
            return True, "first worker"

        want = self.peaks.estimate(natoms)
        if projected_gb + want > self.max_ram_gb:
            return False, (f"{projected_gb:.1f}+{want:.1f} ({natoms} atoms) "
                           f"> {self.max_ram_gb:.0f} GiB budget")

        free = node_available_gb()
        if free is not None and free < self.min_node_free_gb:
            return False, (f"node has {free:.1f} GiB available, floor is "
                           f"{self.min_node_free_gb:.0f} GiB")
        return True, (f"{projected_gb:.1f}+{want:.1f} ({natoms} atoms) "
                      f"/{self.max_ram_gb:.0f} GiB")


# ---------------------------------------------------------------------------
# Parent: main loop
# ---------------------------------------------------------------------------

def run_dynamic(workdir, chunks, model_stages, threads, pin, profile, want_zpe,
                sched, watch=False, check_interval=15.0, poll=3.0, settle=2.0,
                admit_per_cycle=1, ramp_seconds=180.0,
                head_priority_seconds=600.0, retries=1,
                max_idle=None, stop_file=None, stop_flag=None):
    """Relax pending CalcFolders under a measured-memory admission limit."""
    workdir = Path(workdir)
    ctx = mp.get_context("spawn")   # never fork a process that will load torch
    result_q = ctx.Queue()

    free_chunks = deque(list(c) for c in chunks)
    max_workers = len(chunks)

    submitted, skipped, attempts = set(), set(), {}
    waiting = deque()               # tasks accepted but not yet launched
    live = []
    tries = {}                      # task path -> launch attempts
    done = failed = 0
    consecutive_failures = 0
    idle_since = None
    last_admit = 0.0
    last_report = 0.0
    last_hold = ["", 0.0]           # (message, when) — throttles the hold log
    hold_log_every = 300.0

    def _should_stop():
        if stop_flag is not None and stop_flag.is_set():
            return "stop signal"
        if stop_file and Path(stop_file).exists():
            return f"stop file {stop_file}"
        return None

    def _drain_results():
        out = {}
        while True:
            try:
                result = result_q.get_nowait()
            except (queue_mod.Empty, OSError):
                break
            out[result.get("pid")] = result
        return out

    print(f"[cpu_dyn] budget {sched.max_ram_gb:.0f} GiB, node floor "
          f"{sched.min_node_free_gb:.0f} GiB, up to {max_workers} worker(s), "
          f"admitting {admit_per_cycle}/{check_interval:.0f}s, "
          f"ramp {ramp_seconds:.0f}s", flush=True)

    pending_results = {}
    while True:
        reason = _should_stop()

        # ---- reap ---------------------------------------------------------
        pending_results.update(_drain_results())
        for w in list(live):
            if w.proc.is_alive():
                continue
            w.proc.join()
            live.remove(w)
            free_chunks.append(w.cores)
            result = pending_results.pop(w.proc.pid, None)

            if result is None:
                # Exited without publishing: almost always the kernel OOM
                # killer (exitcode -9).  Retry once on a quieter node before
                # writing a failure, and shrink the budget estimate so we do
                # not immediately walk into it again.
                code = w.proc.exitcode
                key = str(Path(w.task["path"]).resolve())
                tries[key] = tries.get(key, 0) + 1
                sched.observe_peak(w.natoms, w.peak_gb)
                if tries[key] <= retries:
                    print(f"[cpu_dyn] {w.name}: worker died (exit={code}, "
                          f"last RSS {w.peak_gb:.1f} GiB) — requeuing "
                          f"({tries[key]}/{retries})", flush=True)
                    waiting.appendleft(w.task)
                    continue
                print(f"[cpu_dyn] {w.name}: worker died (exit={code}) "
                      f"{tries[key]}x — publishing a failure", flush=True)
                result = {"name": w.name, "path": w.task["path"],
                          "pid": w.proc.pid, "cores": w.cores, "ok": False,
                          "energy": None, "zpe": 0.0,
                          "wall": time.monotonic() - w.started,
                          "log": w.task["out_dir"],
                          "error": f"worker process died with exit={code} "
                                   f"without publishing a result"}
                try:
                    batch_worker.write_failure(Path(w.task["path"]),
                                               w.task["atoms"], w.task["mode"])
                except Exception:
                    result["error"] += "\n" + traceback.format_exc()
            else:
                sched.observe_peak(w.natoms, result.get("peak_gb"))

            if result["ok"]:
                done += 1
                consecutive_failures = 0
            else:
                failed += 1
                consecutive_failures += 1
            worker_cpu._report(result, done + failed, len(submitted))
            peak = result.get("peak_gb")
            if peak:
                print(f"[cpu_dyn] {result['name']} peak RSS {peak:.1f} GiB "
                      f"at {w.natoms} atoms — {sched.peaks.describe()}",
                      flush=True)

        if reason and not live and not waiting:
            print(f"[cpu_dyn] stopping ({reason})", flush=True)
            break

        # A fresh run that fails everything is an environment problem, not bad
        # chemistry: stop instead of filling the search with energy-0 entries.
        if done == 0 and consecutive_failures >= max(2, max_workers):
            raise worker_cpu.PoolBroken(
                f"the first {consecutive_failures} relaxation(s) all failed "
                f"and none succeeded — check the tracebacks above"
            )

        # ---- scan ---------------------------------------------------------
        if not reason:
            tasks = worker_cpu.collect_pending(workdir, submitted, skipped,
                                               attempts, settle, profile,
                                               want_zpe)
            if tasks:
                for task in tasks:
                    # Size drives the memory estimate, so record it once here
                    # rather than re-deriving it on every admission pass.
                    task["natoms"] = len(task["atoms"])
                    task["queued_at"] = time.monotonic()
                waiting.extend(tasks)
                sizes = {}
                for task in tasks:
                    sizes[task["natoms"]] = sizes.get(task["natoms"], 0) + 1
                spread = ", ".join(f"{n}x{c}atoms" for n, c in sorted(sizes.items()))
                print(f"[cpu_dyn] queued {len(tasks)} structure(s) [{spread}] "
                      f"({len(waiting)} waiting, {len(live)} running)",
                      flush=True)

        # ---- admit --------------------------------------------------------
        now = time.monotonic()
        current = sched.current_gb(live)
        if waiting and len(live) < max_workers and free_chunks \
                and now - last_admit >= check_interval:
            admitted = 0
            # Provisional total: `current` stays the measured figure so the
            # status line below does not report charges as observations.
            projected = sched.projected_gb(live, now, ramp_seconds)
            while (waiting and admitted < admit_per_cycle
                   and len(live) < max_workers and free_chunks):
                # FIFO first, but if the head does not fit, a smaller
                # structure further back might — with 42..168-atom cells in
                # one wave, strict FIFO would idle cores behind a Z=8 cell
                # that needs twice the memory.  The head keeps absolute
                # priority once it has waited --head-priority-seconds, so
                # large structures cannot be starved by a run of small ones.
                index, ok, why = None, False, ""
                for i, candidate in enumerate(waiting):
                    ok, why = sched.verdict(live, projected, candidate["natoms"])
                    if ok:
                        index = i
                        break
                    if (i == 0 and head_priority_seconds > 0
                            and now - candidate.get("queued_at", now)
                            >= head_priority_seconds):
                        why += " (head has priority; not looking past it)"
                        break
                if index is None:
                    # Being at the budget is the normal steady state, not an
                    # event: without throttling this repeats every poll and
                    # buries the relaxation results in a multi-day log.
                    msg = f"holding {len(waiting)} structure(s): {why}"
                    if msg != last_hold[0] or now - last_hold[1] >= hold_log_every:
                        print(f"[cpu_dyn] {msg}", flush=True)
                        last_hold[0], last_hold[1] = msg, now
                    break
                if index:
                    waiting.rotate(-index)
                    task = waiting.popleft()
                    waiting.rotate(index)
                else:
                    task = waiting.popleft()
                cores = free_chunks.popleft()
                proc = ctx.Process(
                    target=_child_entry,
                    args=(task, model_stages, threads, pin, cores, result_q),
                    daemon=False,
                )
                proc.start()
                live.append(LiveWorker(proc=proc, task=task, cores=cores,
                                       started=time.monotonic()))
                admitted += 1
                print(f"[cpu_dyn] admit {Path(task['path']).name} "
                      f"pid={proc.pid} cores={cores} — {why}, "
                      f"{len(live)}/{max_workers} running, "
                      f"{len(waiting)} waiting", flush=True)
                # Charge the newcomer immediately so a multi-admit cycle does
                # not spend the same headroom twice.
                projected += sched.peaks.estimate(task["natoms"])
            if admitted:
                last_admit = now

        # ---- idle / exit ----------------------------------------------------
        if not live and not waiting:
            if not watch:
                unsettled = worker_cpu.count_unsettled(workdir, submitted,
                                                       skipped, settle)
                if not unsettled:
                    break
                print(f"[cpu_dyn] waiting for {unsettled} folder(s) to settle "
                      f"({settle}s)", flush=True)
            elif max_idle is not None:
                idle_since = idle_since or now
                if now - idle_since >= max_idle:
                    print(f"[cpu_dyn] idle for {max_idle}s — exiting",
                          flush=True)
                    break
        else:
            idle_since = None

        if now - last_report >= 60.0 and live:
            print(f"[cpu_dyn] {len(live)} running, {len(waiting)} waiting, "
                  f"{current:.1f} GiB used (high-water "
                  f"{sched.high_water_gb:.1f}, budget {sched.max_ram_gb:.0f}), "
                  f"node free {node_available_gb() or float('nan'):.1f} GiB",
                  flush=True)
            last_report = now

        time.sleep(poll)

    # Drain anything still in flight when we were asked to stop.
    for w in live:
        w.proc.join()
    with contextlib.suppress(Exception):
        result_q.close()
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
                   help="USPEX working directory (holds Calculation/Calcfold_*)")
    p.add_argument("--model", default="deepmd_d3", choices=sorted(MODELS),
                   help="DeepMD model key from the config's deepmd.models_path; "
                        "ignored when INPUT.txt has a MODELS section")
    p.add_argument("--cores", default=None,
                   help="How many cores, or exactly which ones: '30' (a count), "
                        "'0-29' (a range) or '0,2,4-6' (a list). "
                        "Default: every allowed physical core")
    p.add_argument("--threads", type=int, default=1,
                   help="CPUs/compute threads per relaxation (default: 1)")
    p.add_argument("--workers", type=int, default=None,
                   help="Hard cap on concurrent relaxations. Memory usually "
                        "binds first; this is the ceiling, not the target")
    p.add_argument("--smt", action="store_true",
                   help="Also use SMT sibling threads as separate workers")
    p.add_argument("--oversubscribe", action="store_true",
                   help="Allow --workers to exceed the available cores")
    p.add_argument("--no-pin", action="store_true",
                   help="Do not set CPU affinity; only cap thread pools")

    mem = p.add_argument_group("memory scheduling")
    mem.add_argument("--max-ram-gb", type=float, default=None,
                     help="Total resident budget for all relaxations, GiB. "
                          "Default: the SLURM allocation minus --reserve-gb. "
                          "Set this BELOW the cgroup limit so an overshoot "
                          "queues instead of being OOM-killed")
    mem.add_argument("--mem-base-gb", type=float, default=2.0,
                     help="Size-independent part of the peak estimate, GiB "
                          "(default: 2.0). Peak RSS is modelled as "
                          "base + per-atom * len(atoms); the default line fits "
                          "the measured DPA3 peaks of 3.5 GiB at 42 atoms and "
                          "7.9 GiB at 168")
    mem.add_argument("--mem-per-atom-gb", type=float, default=0.035,
                     help="Per-atom slope of the peak estimate, GiB/atom "
                          "(default: 0.035). Refit by least squares once "
                          "3+ distinct sizes and 6+ relaxations have been "
                          "measured; 0 makes the estimate size-independent")
    mem.add_argument("--mem-per-worker-gb", type=float, default=2.5,
                     help="Absolute floor for any peak estimate, GiB "
                          "(default: 2.5). Only binds for very small cells, "
                          "where the model would otherwise predict less than "
                          "the DeepMD model itself occupies")
    mem.add_argument("--peak-safety", type=float, default=1.10,
                     help="Multiplier applied to the refitted line "
                          "(default: 1.10). Headroom for the fact that a "
                          "sampled fit sits below the true worst case")
    mem.add_argument("--reserve-gb", type=float, default=8.0,
                     help="RAM held back from the allocation for USPEX itself "
                          "and this supervisor (default: 8.0)")
    mem.add_argument("--min-node-free-gb", type=float, default=24.0,
                     help="Stop admitting while the NODE's MemAvailable is "
                          "below this, GiB (default: 24). Guards the page "
                          "cache on diskless nodes, where evicting it stalls "
                          "slurmd on NFS and costs the whole job")
    mem.add_argument("--check-interval", type=float, default=15.0,
                     help="Seconds between admission decisions (default: 15)")
    mem.add_argument("--admit-per-cycle", type=int, default=1,
                     help="Most workers to start per --check-interval "
                          "(default: 1). Keeps RSS from lagging behind a burst "
                          "of launches")
    mem.add_argument("--ramp-seconds", type=float, default=180.0,
                     help="How long a freshly started worker is charged at "
                          "the full estimated peak instead of its measured "
                          "RSS, seconds (default: 180). Covers the model "
                          "load, during which a worker reads far smaller "
                          "than it will become")
    mem.add_argument("--head-priority-seconds", type=float, default=600.0,
                     help="After the queue head has waited this long it is "
                          "admitted strictly in order rather than skipped for "
                          "a smaller structure that fits (default: 600). "
                          "Stops large cells being starved by small ones; "
                          "0 disables the lookahead entirely")
    mem.add_argument("--retries", type=int, default=1,
                     help="Relaunches for a worker that died without "
                          "publishing, e.g. OOM-killed (default: 1)")

    p.add_argument("--zpe", action="store_true",
                   help="Add Gamma-point ZPE to each final energy")
    p.add_argument("--watch", action="store_true",
                   help="Stay alive and keep relaxing new CalcFolders as USPEX "
                        "creates them")
    p.add_argument("--poll", type=float, default=3.0,
                   help="Seconds between CalcFolder scans (default: 3)")
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
        sys.exit(f"[cpu_dyn] workdir not found: {workdir}")

    # Resolve the model in the parent: a missing file would otherwise only
    # surface once per child, one structure at a time.
    try:
        model_stages = batch_worker.load_model_stages(workdir, args.model)
        for line in batch_worker.describe_model_stages(model_stages):
            print(f"[cpu_dyn] model {line}", flush=True)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        sys.exit(f"[cpu_dyn] {exc}")

    # ---- CPU inventory ----------------------------------------------------
    # Note: no clamp_workers_by_memory here.  The chunk count is only a
    # ceiling; the scheduler decides how many of them are in use at any time.
    try:
        pool_cpus, notes = cpu_layout.select_cpus(args.cores, use_smt=args.smt)
        chunks, more = cpu_layout.chunk_cpus(
            pool_cpus,
            threads_per_worker=args.threads,
            max_workers=args.workers,
            oversubscribe=args.oversubscribe,
        )
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"[cpu_dyn] {exc}")

    print(cpu_layout.describe(chunks, args.threads), flush=True)
    for note in notes + more:
        print(f"[cpu_dyn] note: {note}", flush=True)

    # ---- memory budget ----------------------------------------------------
    max_ram = args.max_ram_gb
    if max_ram is None:
        allocation = cpu_layout.available_memory_gb()
        if allocation is None:
            sys.exit("[cpu_dyn] cannot read the memory allocation; pass "
                     "--max-ram-gb explicitly")
        max_ram = allocation - max(0.0, args.reserve_gb)
        print(f"[cpu_dyn] budget from allocation: {allocation:.1f} GiB - "
              f"{args.reserve_gb:.1f} reserved = {max_ram:.1f} GiB", flush=True)
    biggest = args.mem_base_gb + args.mem_per_atom_gb * 168
    if max_ram < biggest:
        sys.exit(f"[cpu_dyn] budget {max_ram:.1f} GiB cannot hold even one "
                 f"168-atom relaxation ({biggest:.1f} GiB estimated)")

    peaks = PeakModel(base_gb=args.mem_base_gb,
                      per_atom_gb=args.mem_per_atom_gb,
                      floor_gb=args.mem_per_worker_gb,
                      safety=args.peak_safety)
    print(f"[cpu_dyn] {peaks.describe()}", flush=True)
    for n in (42, 84, 126, 168):
        print(f"[cpu_dyn]   {n:>4} atoms -> {peaks.estimate(n):.1f} GiB",
              flush=True)
    sched = MemoryScheduler(max_ram_gb=max_ram, peaks=peaks,
                            min_node_free_gb=args.min_node_free_gb)

    # Both inherited by the spawned children, so OpenBLAS/numpy are capped and
    # the GPU is hidden by the time they import.
    cpu_layout.set_thread_env(args.threads)
    worker_cpu.force_cpu_only()

    profile = {}
    if args.smoke:
        profile = {"fire_steps": 2, "lbfgs_steps": 2, "lbfgs_stages": (0.03,)}
        print("[cpu_dyn] SMOKE profile: abbreviated relaxation", flush=True)

    # ---- molecular interface guard (same rule as worker.py) --------------
    if list(workdir.glob("MOL_*")):
        legacy = [cf for _, cf in batch_worker.discover_calcfolders(workdir)
                  if batch_worker.calcfolder_mode(cf) == batch_worker.USER_CODE_MODE]
        if legacy:
            sys.exit(
                "[cpu_dyn] molecular MOL_* input detected with legacy USPEX "
                "USER_CODE/code 99. Use abinitioCode 20 so extended XYZ "
                "preserves molecular atom order."
            )

    stop_flag = worker_cpu._install_signal_handlers()

    if not args.watch and not batch_worker.discover_calcfolders(workdir):
        print(f"[cpu_dyn] no pending CalcFolders under {workdir}; "
              f"nothing to do", flush=True)
        return 0

    t0 = time.perf_counter()
    try:
        done, failed = run_dynamic(
            workdir, chunks, model_stages, args.threads, not args.no_pin,
            profile, args.zpe, sched,
            watch=args.watch, check_interval=args.check_interval,
            poll=args.poll, settle=args.settle,
            admit_per_cycle=max(1, args.admit_per_cycle),
            ramp_seconds=max(0.0, args.ramp_seconds),
            head_priority_seconds=max(0.0, args.head_priority_seconds),
            retries=max(0, args.retries),
            max_idle=args.max_idle, stop_file=args.stop_file,
            stop_flag=stop_flag,
        )
    except worker_cpu.PoolBroken as exc:
        print(f"[cpu_dyn] ABORTING — {exc}", flush=True)
        return 1
    print(f"[cpu_dyn] done — relaxed={done}  fallback={failed}  "
          f"peak sum RSS={sched.high_water_gb:.1f} GiB  "
          f"{sched.peaks.describe()}  "
          f"wall={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
