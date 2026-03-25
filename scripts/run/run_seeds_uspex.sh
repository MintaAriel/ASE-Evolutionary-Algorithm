#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/vito/uspex_matlab/theo_uspex"

for i in $(seq 1 10); do
    DIR="$BASE_DIR/test_$i"
    ( cd "$DIR" && ./run.sh ) &
done
wait

# Wait for all jobs to finish
wait
echo "[INFO] All 10 USPEX jobs finished."
