#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/vito/uspex_matlab/theo_uspex"
SRC_DIR="$BASE_DIR/test_0"
POSCAR_DIR_BASE="/home/vito/uspex_matlab/theo_pyxtal/test_"  # POSCAR_1 in theo_pyxtal_testi

# Number of parallel jobs / tests
N_RUNS=10

for i in $(seq 1 "$N_RUNS"); do
    DEST_DIR="$BASE_DIR/test_$i"

    # Copy test_0 to test_i
    cp -r "$SRC_DIR" "$DEST_DIR"

    # Copy POSCAR_1 into test_i/Seeds
    SEEDS_DIR="$DEST_DIR/Seeds"
    mkdir -p "$SEEDS_DIR"
    POSCAR_FILE="$POSCAR_DIR_BASE$i/POSCARS_1"
    cp "$POSCAR_FILE" "$SEEDS_DIR/"

    # Create run.sh inside each test_i
    RUN_SCRIPT="$DEST_DIR/run.sh"
    cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"  # <-- go to the folder containing run.sh

# Assign core (0-9)
CORE=$((i-1))

LOGFILE="run.log"

# Limit all math threads to 1
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] Running USPEX in \$PWD on core \$CORE"
taskset -c \$CORE conda run -n uspex_matlab ~/uspex_matlab/program/application/archive/USPEX -r "\$PWD" \ 2>&1 | tee "\$LOGFILE"

EOF

    chmod +x "$RUN_SCRIPT"
done

echo "[INFO] test_1..test_10 created with run.sh scripts."
