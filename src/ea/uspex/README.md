# `ea.uspex` — USPEX orchestration scripts

USPEX 26 is a complete rewrite and drives structure search very differently
from USPEX 10, so the two workflows are kept in **separate** subpackages.
Both feed candidate structures to a batched **DeepMD** relaxation worker
(`worker.py`) that uses `ea.parallel.create_batch` / `FIRE_parallel` /
`LBFGS_parallel` / `zpe`.

```
ea/uspex/
├── models.py  DeepMD model registry shared by the workers
├── uspex26/   USPEX 26 — one continuous process, ASE (abinitioCode 20)
│   ├── run_uspex26.py   driver (Python): starts USPEX 26, watches
│   │                    Calculation/Calcfold_*, calls worker.py per wave
│   ├── worker.py        batched DeepMD relaxer, GPU-shaped (one dp.eval per
│   │                    optimizer step across the whole wave)
│   ├── worker_cpu.py    per-structure DeepMD relaxer, CPU-shaped (one process
│   │                    + one pinned core per structure)
│   └── run_batch.py     per-Calcfold stub (waits for output.xyz)  ← commandExecutable
└── uspex10/   USPEX 10 (legacy) — repeated `USPEX -r`, whichCluster=1, job.info
    ├── run_uspex.py            per-CalcFolder resume loop
    ├── parallel_run_uspex.py   batched-DeepMD resume loop
    ├── worker.py               batched DeepMD relaxer (CalcFold<N> only)
    └── run_batch.sh            per-CalcFolder stub
```

## USPEX 26

The driver replaces the old `run_uspex26_deepmd.sh`. It is relocatable
(finds `worker.py` and `EA/src` relative to itself) and runs USPEX in the
**workdir**, not in the package directory.

```bash
cd /path/to/your/uspex_run          # holds INPUT.txt, MOL_1, ...
python /path/to/ASE-Evolutionary-Algorithm/src/ea/uspex/uspex26/run_uspex26.py \
       --model deepmd_d3 --device cpu
# or, from anywhere:
python .../uspex26/run_uspex26.py --workdir /path/to/your/uspex_run --device cuda
```

- USPEX 26 binary resolves from `--uspex-exe` → `$USPEX26_EXE` →
  config `uspex26.exe` → hardcoded default. It is set in
  `configs/local_fedora.yaml` (`uspex26.exe`).
- `worker.py` runs in the DeepMD env via `conda run -n <deepmd.conda_env>`
  unless `--worker-python` / `$WORKER_PY` gives an explicit interpreter.
- Molecular runs must use USPEX's ASE interface. In `INPUT.txt` set:

  ```text
  % abinitioCode
  20
  % ENDabinit
  ```

  Code 20 passes structures through the order-preserving extended-XYZ files
  `input.xyz` and `output.xyz`. Do not use USER_CODE/code 99 for `MOL_*`
  runs: its POSCAR conversion groups atoms globally by element, so USPEX can
  read the energy but cannot rebuild the relaxed molecular structure.
- Set `commandExecutable` to this package's `run_batch.py` (copy it next to
  your run or point at it directly). The stub waits for `output.xyz`; the
  batched worker writes it atomically after relaxation.
- `--smoke` abbreviates the optimizer to two FIRE and two LBFGS steps for an
  integration check. It is not a production relaxation setting.

### GPU (`--worker batch`, default) vs CPU (`--worker cpu`)

`worker.py` is built for a GPU: one process evaluates the whole wave in a
single `dp.eval` per optimizer step, so every structure sits on the same step
barrier. On a CPU there is no wide device to fill — the barrier just
serializes, and cores idle as structures converge out of the batch.

`worker_cpu.py` inverts that: **one process, one pinned core and one DeepMD
calculator per structure**, running the same `run_full_optimization` recipe on
a batch of one. Where the GPU worker relaxes 40 structures as one batch, this
launches 40 independent single-core relaxations. Structures beyond the core
count queue up, and a worker picks the next one the moment it finishes (its
model stays loaded), so no core waits on a slow neighbour.

```bash
# whole search, one relaxation per core, cores 0-11
python .../uspex26/run_uspex26.py --worker cpu --cores 0-11 --threads 1

# just relax whatever is pending right now (drop-in for worker.py)
python .../uspex26/worker_cpu.py /path/to/run --cores 8 --model deepmd_d3
```

- `--cores` takes a **count** (`8`), a **range** (`0-11`) or a **list**
  (`0,2,4-6`). Default: every allowed physical core. `--threads N` gives each
  relaxation N CPUs instead of 1; `--smt` also hands out SMT siblings.
- The selection is intersected with `os.sched_getaffinity`, so under **SLURM**
  the pool automatically confines itself to the job's allocation and never
  pins outside it. See `scripts/cluster/uspex26_cpu.sbatch`. Add `--no-pin` if
  your site prefers to own the pinning via `--cpu-bind`.
- In `--worker cpu` mode the driver starts **one** long-lived worker in watch
  mode for the whole search instead of one per wave, so models load once and
  the pool stays warm across generations. It is stopped gracefully (via a stop
  file) once USPEX exits, letting in-flight relaxations publish first.
- Keep `numParallelCalcs` in `INPUT.txt` **>= the number of workers**: USPEX
  only ever has that many structures pending at once, so a smaller value
  leaves cores idle no matter what `--cores` says.
### Memory: what actually bounds concurrency

Memory, not cores, is the binding constraint, and the number to budget is the
**transient peak of one force+stress evaluation**, not the resident set. All
measured for the same 42-atom theophylline cell (2 molecules), `deepmd_d3`
(DPA3, rcut 6 A), one core, `MALLOC_ARENA_MAX=1`:

| cell | neighbours/atom | periodic images | peak RSS | resident after |
|------|-----------------|-----------------|----------|----------------|
| 246 A^3 (USPEX candidate as generated) | 155 | 45 | **4.8 GiB** | 1.3 GiB |
| 400 A^3, compact (`Latticevalues` target) | 95 | 27 | **3.1-3.7 GiB** | 1.3-2.0 GiB |
| 700 A^3 | 54 | 27 | **1.9-2.3 GiB** | 1.3 GiB |

Resident memory really is only ~1.3 GiB. The peak comes from the autograd
backward pass over `nall x nnei` pairs, and it grows with
**neighbours-per-atom**, i.e. with density. Verified consequence: five
single-point calculations, four at a time, fit in an 8 GiB cgroup at 700 A^3
but get OOM-killed at 400 and 246 A^3.

Two controls worth recording so nobody re-derives them:

- the batched evaluator and a plain ASE `DP` calculator use the **same** memory
  to within noise — this is not our optimizer or our batching, and
- `base_deepmd` costs the same as `deepmd_d3`, so the **D3 dispersion term is
  not responsible**.

Two geometry traps make USPEX candidates far more expensive than the target
volume suggests:

- `shrinkVolume` can compress a candidate well below `Latticevalues` (seen: 400
  -> 226-317 A^3, logging `ratio 0.613 ... termination
  time_budget_exhausted`), and density is exactly what drives the peak.
- **Any perpendicular cell width below rcut (6 A) forces extra tilings.** These
  candidates are elongated (e.g. 4.57 x 7.79 x 6.91 A), so the 4.6 A axis makes
  deepmd build 5x3x3 = 45 images and hand the model 1890 atoms instead of 42.
  A compact cell of the *same* volume needs only 27.

`--mem-per-worker-gb` (default 3.5, the 400 A^3 figure) and `--reserve-gb`
(default 2.0, for USPEX itself) clamp the worker count and log the arithmetic;
`--mem-per-worker-gb 0` opts out. On a 16 GiB box `--cores 12` would ask for
~42 GiB and swap the machine to death, especially where swap is zram
(compressed RAM, so swapping consumes the very thing it is meant to free).
- **`MALLOC_ARENA_MAX` matters more than anything else here.** torch's CPU path
  frees a large tensor every step, and glibc's default per-thread arenas hoard
  the freed blocks: RSS ratchets to ~4x the live working set and never returns.
  `cpu_layout.set_memory_env()` (applied automatically, before the first
  allocation) took the dense cell above from **7.7 GiB steady / 8.2 GiB peak
  down to 2.0 / 5.1 GiB** with identical energies. Do not remove it, and set it
  in the parent so spawned children inherit it — glibc reads it once at
  process start.
- Each structure gets its own `cpu_out/<Calcfold>/relax.log` and
  `output.traj`; the parent log keeps a one-line summary per structure.
- The worker is CPU-only by construction: it blanks `CUDA_VISIBLE_DEVICES`
  before torch is imported and refuses to run if the model still lands on a
  GPU. `DP(model, device="cpu")` alone does **not** guarantee this — deepmd's
  torch backend picks its device from `torch.cuda.is_available()` at import
  time, and one CUDA context per core will OOM the card immediately.

See the original walkthrough (INPUT.txt edits, data flow, tuning knobs) in
`/home/vito/uspex_python/USPEX_26_TESTS/test1/README_USPEX26_DEEPMD.md`.

## USPEX 10 (legacy)

Unchanged working scripts, moved here from `scripts/run` and
`scripts/experiments`. They require `whichCluster = 1` (so `submitJob_local.py`
writes `job.info` per CalcFolder for PID tracking) and repeatedly call
`USPEX -r`.

```bash
python .../uspex10/parallel_run_uspex.py --workdir /path/to/run   # batched
python .../uspex10/run_uspex.py          --workdir /path/to/run   # per-folder
```
