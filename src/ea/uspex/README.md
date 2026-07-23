# `ea.uspex` — USPEX orchestration scripts

USPEX 26 is a complete rewrite and drives structure search very differently
from USPEX 10, so the two workflows are kept in **separate** subpackages.
Both feed candidate structures to a batched **DeepMD** relaxation worker
(`worker.py`) that uses `ea.parallel.create_batch` / `FIRE_parallel` /
`LBFGS_parallel` / `zpe`.

```
ea/uspex/
├── uspex26/   USPEX 26 — one continuous process, ASE (abinitioCode 20)
│   ├── run_uspex26.py   driver (Python): starts USPEX 26, watches
│   │                    Calculation/Calcfold_*, calls worker.py per wave
│   ├── worker.py        batched DeepMD relaxer (Calculation/Calcfold_* AND
│   │                    legacy CalcFold<N>)
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
