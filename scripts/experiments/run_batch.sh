#!/usr/bin/env bash
# Per-CalcFolder stub for the batched DeepMD workflow.
#
# USPEX invokes this once per CalcFolder.  The actual relaxation is done
# by scripts/experiments/uspex_deemdp_parallel.py, called ONCE per
# generation by scripts/run/parallel_run_uspex.py over all CalcFolders
# at the same time.  This stub exists only so each CalcFolder has a
# live PID for USPEX's job-tracking model: it sleeps until the batch
# worker drops energy.txt, then exits, freeing the slot.

set -u

CF="$(basename "$PWD")"
echo "[$(date '+%H:%M:%S')] $CF: waiting for batch worker (energy.txt)..."

while [ ! -f energy.txt ]; do
    sleep 2
done

echo "[$(date '+%H:%M:%S')] $CF: energy.txt present, exiting."
