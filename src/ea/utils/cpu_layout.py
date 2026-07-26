"""SLURM-aware CPU inventory, selection and partitioning.

Used by the CPU relaxation worker to decide *how many* independent
single-structure relaxations to run concurrently and *which* logical CPUs
each one is pinned to.  Everything is derived from what the process is
actually allowed to use (``os.sched_getaffinity``) plus the kernel's SMT
topology, so the same code gives the right answer on a workstation and
inside a SLURM cpuset/cgroup allocation — no per-machine configuration.

Typical use::

    pool, notes = select_cpus("0-11")        # explicit range
    pool, notes = select_cpus("8")           # "give me 8 cores"
    pool, notes = select_cpus(None)          # every allowed physical core
    chunks, more = chunk_cpus(pool, threads_per_worker=1)

``chunks[i]`` is the CPU set for worker *i*; ``len(chunks)`` is how many
workers to start.
"""

import os
from pathlib import Path

SYSFS_CPU = Path("/sys/devices/system/cpu")

# SLURM variables worth echoing into the log when running under a batch job.
SLURM_VARS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_CPU_BIND",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
)

# Thread-limiting variables that must be set before torch/deepmd import.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# ---------------------------------------------------------------------------
# Environment inspection
# ---------------------------------------------------------------------------

def under_slurm():
    return "SLURM_JOB_ID" in os.environ


def slurm_info():
    """Return the SLURM variables that are set (empty dict outside SLURM)."""
    return {k: os.environ[k] for k in SLURM_VARS if k in os.environ}


def allowed_cpus():
    """Logical CPUs this process may run on.

    Under SLURM with cgroup/cpuset confinement this is the allocation, not
    the whole node — which is exactly what we want to partition.
    """
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:                      # non-Linux
        return list(range(os.cpu_count() or 1))


def _parse_cpu_list(text):
    """Parse a sysfs CPU list ('0,6' or '0-1' or '0-3,8') into a list of ints."""
    out = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def thread_siblings(cpu):
    """SMT siblings of *cpu* per sysfs, or ``[cpu]`` if unknown."""
    path = SYSFS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        return sorted(_parse_cpu_list(path.read_text()))
    except (OSError, ValueError):
        return [cpu]


def physical_groups(cpus=None):
    """Group *cpus* by physical core.

    Returns a list of lists; each inner list holds the SMT siblings of one
    physical core that are present in *cpus*, ordered by CPU id.  Cores are
    ordered by their lowest CPU id.  CPUs whose topology cannot be read are
    treated as cores of their own.
    """
    cpus = allowed_cpus() if cpus is None else sorted(cpus)
    allowed = set(cpus)
    seen = set()
    groups = []
    for cpu in cpus:
        if cpu in seen:
            continue
        siblings = [c for c in thread_siblings(cpu) if c in allowed]
        if not siblings:                        # defensive
            siblings = [cpu]
        seen.update(siblings)
        groups.append(siblings)
    return groups


def cpu_pool(cpus=None, use_smt=False):
    """Ordered CPU list to hand out to workers.

    Without *use_smt* only one CPU per physical core is offered, so one
    worker per core never shares an execution unit with another.  With
    *use_smt* the siblings are appended in rounds (all first siblings, then
    all second siblings, ...), so consecutive slices still prefer distinct
    physical cores before doubling up.
    """
    groups = physical_groups(cpus)
    if not use_smt:
        return [g[0] for g in groups]

    pool = []
    depth = max((len(g) for g in groups), default=0)
    for level in range(depth):
        for g in groups:
            if level < len(g):
                pool.append(g[level])
    return pool


# ---------------------------------------------------------------------------
# Selection: --cores "0-11" | "0,2,4" | "8"
# ---------------------------------------------------------------------------

def parse_core_spec(spec):
    """Interpret a ``--cores`` value.

    ``"8"``      -> ``("count", 8)``   — use any 8 cores
    ``"0-11"``   -> ``("list", [0..11])`` — use exactly these logical CPUs
    ``"0,2,4-6"``-> ``("list", [0, 2, 4, 5, 6])``

    A bare integer is a *count*; write ``"8-8"`` to mean the single CPU 8.
    """
    if spec is None:
        return None, None
    if isinstance(spec, int):
        return "count", int(spec)

    text = str(spec).strip()
    if not text:
        return None, None
    try:
        return "count", int(text)
    except ValueError:
        pass
    try:
        ids = _parse_cpu_list(text)
    except ValueError:
        raise ValueError(
            f"cannot parse core spec {spec!r}; expected a count ('8'), "
            f"a range ('0-11') or a list ('0,2,4-6')"
        )
    if not ids:
        raise ValueError(f"core spec {spec!r} selects no CPUs")
    return "list", sorted(set(ids))


def select_cpus(spec=None, use_smt=False):
    """Resolve a ``--cores`` spec against what this process may actually use.

    Returns ``(pool, notes)`` where *pool* is an ordered list of logical CPU
    ids to hand out to workers and *notes* are human-readable strings about
    any clamping (for the caller to log).
    """
    notes = []
    allowed = allowed_cpus()
    allowed_set = set(allowed)
    kind, value = parse_core_spec(spec)

    if kind == "list":
        outside = [c for c in value if c not in allowed_set]
        pool = [c for c in value if c in allowed_set]
        if outside:
            notes.append(
                f"CPUs {outside} are not in this process' affinity mask "
                f"({'SLURM allocation' if under_slurm() else 'sched_getaffinity'}) "
                f"and were dropped"
            )
        if not pool:
            raise RuntimeError(
                f"core spec {spec!r} selects no usable CPU; allowed: {allowed}"
            )
        siblings_used = [g for g in physical_groups(pool) if len(g) > 1]
        if siblings_used and not use_smt:
            notes.append(
                f"{len(siblings_used)} selected core(s) contribute both SMT "
                f"threads; they are treated as separate workers"
            )
        return pool, notes

    pool = cpu_pool(allowed, use_smt)
    if kind == "count":
        if value < 1:
            raise ValueError("--cores count must be >= 1")
        if value > len(pool):
            notes.append(
                f"requested {value} core(s) but only {len(pool)} available"
                + ("" if use_smt else " (add --smt to also use SMT siblings)")
            )
        else:
            pool = pool[:value]
    return pool, notes


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def chunk_cpus(pool, threads_per_worker=1, max_workers=None,
               oversubscribe=False):
    """Split an ordered CPU *pool* into one pinning set per worker.

    Parameters
    ----------
    pool:
        Ordered logical CPU ids, as returned by :func:`select_cpus`.
    threads_per_worker:
        CPUs (and compute threads) given to each relaxation.  1 means one
        thread per structure.
    max_workers:
        Cap on the number of workers, e.g. the number of structures to
        relax.  ``None`` uses every full chunk the pool affords.
    oversubscribe:
        Allow *max_workers* to exceed the number of distinct chunks by
        cycling through them (several workers then share CPUs).  Off by
        default because it usually hurts throughput.

    Returns ``(chunks, notes)``.
    """
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be >= 1")
    pool = list(pool)
    if not pool:
        raise RuntimeError("no usable CPUs found")

    notes = []
    chunks = [pool[i:i + threads_per_worker]
              for i in range(0, len(pool), threads_per_worker)]
    full = [c for c in chunks if len(c) == threads_per_worker]
    if not full:
        notes.append(f"only {len(pool)} CPU(s) available for "
                     f"threads_per_worker={threads_per_worker}; "
                     f"running a single worker on all of them")
        chunks = [pool]
    elif len(full) != len(chunks):
        dropped = len(pool) - len(full) * threads_per_worker
        notes.append(f"{dropped} CPU(s) left idle: not a full "
                     f"{threads_per_worker}-CPU chunk")
        chunks = full

    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_workers <= len(chunks):
            chunks = chunks[:max_workers]
        elif oversubscribe:
            base = list(chunks)
            chunks = [base[i % len(base)] for i in range(max_workers)]
            notes.append(f"oversubscribing: {max_workers} workers share "
                         f"{len(base)} CPU chunk(s)")
        else:
            notes.append(f"asked for {max_workers} worker(s) but the CPU "
                         f"selection only affords {len(chunks)}; "
                         f"structures will queue (add --oversubscribe to force "
                         f"more processes than cores)")
    return chunks, notes


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def available_memory_gb():
    """RAM this job may realistically use, in GiB (``None`` if unknown).

    Under SLURM the allocation is the real ceiling and ``/proc/meminfo`` would
    report the whole node, so the allocation wins when it is set.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return int(per_node) / 1024.0          # SLURM reports MiB
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu:
        try:
            return int(per_cpu) * len(allowed_cpus()) / 1024.0
        except ValueError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    return None


def clamp_workers_by_memory(chunks, mem_per_worker_gb, reserve_gb=0.0):
    """Drop worker slots that would not fit in RAM.

    One relaxation process holds a whole DeepMD model plus its activations, so
    memory — not core count — is usually what bounds concurrency.  Overshooting
    does not merely slow the run down: the node starts swapping and, where swap
    is zram (compressed, and therefore *in* RAM), it can wedge the machine
    outright.  *reserve_gb* is held back for everything that is not a
    relaxation, i.e. USPEX itself and the OS.

    Returns ``(chunks, notes)``; a non-positive *mem_per_worker_gb* disables
    the check.
    """
    notes = []
    if mem_per_worker_gb is None or mem_per_worker_gb <= 0:
        return chunks, notes

    budget = available_memory_gb()
    if budget is None:
        notes.append("cannot read available memory; not clamping worker count")
        return chunks, notes

    usable = budget - max(0.0, reserve_gb)
    affordable = int(usable // mem_per_worker_gb)
    notes.append(
        f"memory: {budget:.1f} GiB available, {reserve_gb:.1f} GiB reserved, "
        f"{mem_per_worker_gb:.1f} GiB per worker -> room for {affordable} "
        f"worker(s), {len(chunks)} requested"
    )
    if affordable >= len(chunks):
        return chunks, notes
    if affordable < 1:
        raise RuntimeError(
            f"not enough memory for a single relaxation: {budget:.1f} GiB "
            f"available, {reserve_gb:.1f} GiB reserved and "
            f"{mem_per_worker_gb:.1f} GiB needed per worker. Lower "
            f"--mem-per-worker-gb/--reserve-gb if that estimate is too "
            f"pessimistic, or free memory."
        )
    notes.append(
        f"clamping {len(chunks)} -> {affordable} worker(s) to stay inside RAM "
        f"(structures queue instead; pass --mem-per-worker-gb 0 to disable)"
    )
    return chunks[:affordable], notes


def describe(chunks=None, threads_per_worker=None):
    """Multi-line summary of the CPU environment, for logging."""
    lines = []
    cpus = allowed_cpus()
    groups = physical_groups(cpus)
    smt = max((len(g) for g in groups), default=1)
    lines.append(f"allowed CPUs      : {len(cpus)} {cpus}")
    lines.append(f"physical cores    : {len(groups)} (SMT depth {smt})")
    if under_slurm():
        lines.append("SLURM             : " +
                     "  ".join(f"{k.replace('SLURM_', '').lower()}={v}"
                               for k, v in slurm_info().items()))
    else:
        lines.append("SLURM             : not under a SLURM job")
    if chunks is not None:
        lines.append(f"workers           : {len(chunks)}"
                     + (f" x {threads_per_worker} thread(s)"
                        if threads_per_worker else ""))
        lines.append(f"pinning           : {[list(c) for c in chunks]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-process runtime configuration
# ---------------------------------------------------------------------------

def set_memory_env(arenas=1, trim_threshold_mb=128):
    """Stop glibc from hoarding freed blocks in per-thread malloc arenas.

    torch's CPU path allocates and frees large tensors every optimizer step.
    By default glibc keeps up to ``8 * ncores`` arenas and returns almost
    nothing to the OS, so RSS ratchets up to several times the live working
    set and never comes back down.  Measured on a 42-atom molecular crystal
    with DPA3 (rcut 6 A, ~143 neighbours/atom): steady RSS **7.7 GiB before,
    2.0 GiB after**, with the transient per-evaluation peak going 8.2 -> 5.1
    GiB.  The live memory was never the problem; the fragmentation was.

    Must be set before the process makes its first allocation, so glibc reads
    it at startup: set it in the parent and let spawned children inherit it.
    An explicit value from the environment is respected.
    """
    os.environ.setdefault("MALLOC_ARENA_MAX", str(arenas))
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_",
                          str(int(trim_threshold_mb) * 1024 * 1024))


def set_thread_env(nthreads):
    """Cap every BLAS/OpenMP thread-pool variable at *nthreads*.

    Call this in the PARENT before spawning workers as well: OpenBLAS reads
    its variable when ``numpy`` is imported, which in a spawned child happens
    before any initializer runs, so the value has to be inherited.
    """
    for var in _THREAD_VARS:
        os.environ[var] = str(nthreads)
    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(nthreads)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = "1"


def configure_cpu_runtime(cores, nthreads, pin=True):
    """Restrict the *current* process to *cores* and cap its thread pools.

    Must be called BEFORE importing torch / deepmd: the thread counts below
    are read by OpenMP / MKL / DeePMD at import time.  Mirrors
    ``scripts/experiments/deepmd_template.py:configure_cpu_runtime`` but
    tolerates a failed pinning attempt (restricted cgroups, containers)
    instead of dying, since the thread caps alone already prevent
    oversubscription.

    Returns a ``(pinned, message)`` tuple.
    """
    cores = sorted(set(int(c) for c in cores))
    pinned = False
    message = "not pinned"

    if pin and cores:
        try:
            os.sched_setaffinity(0, set(cores))
            pinned = True
            message = f"pinned to {sorted(os.sched_getaffinity(0))}"
        except (OSError, AttributeError) as exc:
            message = f"pinning to {cores} failed ({exc}); relying on thread caps"

    set_thread_env(nthreads)

    if pinned:
        # Keep OpenMP/MKL worker threads inside the affinity mask; libgomp and
        # MKL otherwise spread across every CPU on the node.
        cores_csv = ",".join(str(c) for c in cores)
        os.environ["OMP_PROC_BIND"] = "close"
        os.environ["OMP_PLACES"] = f"{{{cores_csv}}}"
        os.environ["GOMP_CPU_AFFINITY"] = cores_csv
        os.environ["KMP_AFFINITY"] = (
            f"granularity=fine,proclist=[{cores_csv}],explicit"
        )

    return pinned, message
