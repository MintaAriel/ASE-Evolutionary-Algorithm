#!/bin/bash

# Base directory
BASE_DIR="/home/vito/uspex_matlab/theo_pyxtal"
SRC_DIR="$BASE_DIR/test_0"

# Python script
PY_SCRIPT="/home/vito/PythonProjects/ASEProject/EA/scripts/start_e.py"

# Loop to create 10 copies and run in parallel on cores 0-9
for i in $(seq 1 10); do
    DEST_DIR="$BASE_DIR/test_$i"

    # Copy test_0 to test_i
    cp -r "$SRC_DIR" "$DEST_DIR"

    # Assign CPU core (0-9)
    CORE=$((i-1))

    echo "Running script in $DEST_DIR on core $CORE..."

    # Run the Python script on a specific core in background
    taskset -c $CORE conda run -n ase_env python3 "$PY_SCRIPT" "$DEST_DIR" &
done

# Wait for all background jobs to finish
wait

echo "All 10 jobs completed."
