#!/bin/bash
#SBATCH --job-name=uspex26-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # one relaxation per core (see --cores below)
#SBATCH --time=48:00:00
#SBATCH --output=uspex26-cpu-%j.log
#
# USPEX 26 + per-structure CPU DeepMD relaxation (worker_cpu.py).
#
# Submit FROM the USPEX working directory (the one holding INPUT.txt, MOL_1):
#
#     sbatch --cpus-per-task=16 /path/to/EA/scripts/cluster/uspex26_cpu.sbatch
#
# The whole search runs inside this single job: the driver starts USPEX 26 and
# one long-lived worker pool.  The pool confines itself to the CPUs SLURM
# allocated (it intersects --cores with sched_getaffinity), so it never spills
# onto another job's cores; --cores below just says how many of them to use.
#
# Keep INPUT.txt's numParallelCalcs >= the number of workers, otherwise USPEX
# never has that many structures pending at once and cores sit idle.

set -euo pipefail

# ---- paths (edit these two) ------------------------------------------------
EA_ROOT=${EA_ROOT:-$HOME/ASE-Evolutionary-Algorithm}
CONDA_SH=${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
DRIVER="$EA_ROOT/src/ea/uspex/uspex26/run_uspex26.py"

# ---- environment ----------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "${EA_ENV:-deepmd_env}"      # driver needs pyyaml; worker needs deepmd

export EA_CONFIG=${EA_CONFIG:-sirena}       # picks configs/<name>.yaml
export PYTHONPATH="$EA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# One compute thread per relaxation; worker_cpu re-applies this per process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- how many cores to hand to the relaxation pool ------------------------
# Default: every CPU SLURM gave this task, one relaxation each.
CORES=${CORES:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-4}}}
THREADS=${THREADS:-1}

echo "workdir : $WORKDIR"
echo "EA_ROOT : $EA_ROOT"
echo "config  : $EA_CONFIG"
echo "cores   : $CORES  (threads/relaxation: $THREADS)"
echo "node    : $(hostname)   allocated CPUs: ${SLURM_JOB_CPUS_PER_NODE:-?}"

cd "$WORKDIR"

# --worker-python: this job's interpreter already has deepmd, so skip the
#                  'conda run -n ...' indirection.
# --no-pin:        add it if your site uses --cpu-bind and you want SLURM to
#                  own the pinning instead of the worker.
srun --unbuffered python "$DRIVER" \
    --workdir "$WORKDIR" \
    --worker cpu \
    --cores "$CORES" \
    --threads "$THREADS" \
    --model "${MODEL:-deepmd_d3}" \
    --worker-python "$(command -v python)"

echo "finished at $(date)"
