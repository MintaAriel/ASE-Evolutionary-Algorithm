#!/usr/bin/env python3
"""Per-Calcfold stub for the batched USPEX 26 DeepMD workflow.

USPEX 26 invokes this once per Calcfold (as ``commandExecutable``, pinned to
one core).  The actual relaxation is done ONCE per wave by ``worker.py``,
driven by ``run_uspex26.py``, across all pending Calcfolds at the same time.
This stub only exists so each Calcfold keeps a live PID for USPEX's
job-tracking model: it waits until the batch worker atomically publishes the
interface's completion marker (``output.xyz`` for ASE/code 20 or
``energy.txt`` for legacy USER_CODE/code 99), then exits, freeing the slot.
USPEX polls the job's exit code:
still running -> IN_PROGRESS, exit 0 -> FINISHED, non-zero -> FAILED.  So this
stub must stay alive until the selected interface's completion marker exists,
then exit 0.

IMPORTANT — env scrubbing.  USPEX 26 is a self-extracting bundled-Python app
and launches ``commandExecutable`` with ``PYTHONHOME`` / ``PYTHONPATH`` /
``LD_LIBRARY_PATH`` pointing at its own bundle.  A bare ``python3 run_batch.py``
then inherits that ``PYTHONHOME`` and dies instantly with
``Fatal Python error: Failed to import encodings module`` (exit 1), which USPEX
reads as ``Task failed`` and never waits for the relaxation.  So
``commandExecutable`` in INPUT.txt MUST scrub those vars before python starts::

    % commandExecutable
    /usr/bin/env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH /usr/bin/python3 \
        /home/vito/PythonProjects/ASEProject/EA/src/ea/uspex/uspex26/run_batch.py
    % EndExecutable

(A plain shell stub would not need this, but the whole workflow is kept in
Python — the scrub is a one-line invocation, not a separate script.)
"""

import os
import time
from pathlib import Path


def main():
    cwd = Path.cwd()
    cf = cwd.name
    if (cwd / "input.xyz").is_file():
        marker = "output.xyz"
    elif (cwd / "geom.in").is_file():
        marker = "energy.txt"
    else:
        raise SystemExit(f"{cf}: neither input.xyz nor geom.in is present")

    print(f"[{time.strftime('%H:%M:%S')}] {cf}: waiting for batch worker ({marker})...",
          flush=True)
    while not os.path.isfile(marker):
        time.sleep(2)
    print(f"[{time.strftime('%H:%M:%S')}] {cf}: {marker} present, exiting.", flush=True)


if __name__ == "__main__":
    main()
