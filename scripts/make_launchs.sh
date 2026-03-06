#!/bin/bash
BASE_DIR="/home/vito/PythonProjects/ASEProject/EA/COMPARISON/TEST3"
DB_DIR="/home/vito/Documents/UMA/Mg4Al8O16_40.db"

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

for i in {0..9}; do
    test_dir="$BASE_DIR/test_$i"
    mkdir -p "$test_dir"
    cp "$DB_DIR" "$test_dir/"
    db_test_dir="$test_dir/Mg4Al8O16_40.db"
    
    # Create job.sh script for this test directory
    cat > "$test_dir/job.sh" << 'EOF'
#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment from miniconda3
source /home/vito/miniconda3/etc/profile.d/conda.sh
conda activate ase_env

# Set temporary directory
export TMPDIR=/home/vito/tmp
mkdir -p "$TMPDIR"

# Limit thread usage for better parallelization
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Extract pair number from directory name (test_0, test_1, etc.)
PAIR_NUMBER=$(basename "$SCRIPT_DIR" | sed 's/test_//')

# Run the Python script with proper arguments
python /home/vito/PythonProjects/ASEProject/EA/COMPARISON/operators_ase.py \
    --count 100 \
    --operator softmut_modes \
    --folder "$SCRIPT_DIR" \
    --db "$SCRIPT_DIR/Mg4Al8O16_40.db" \
    --pair "$PAIR_NUMBER" \
    >> "$SCRIPT_DIR/log" 2>&1

# Optional: Print completion message
echo "Job in $SCRIPT_DIR completed at $(date)" >> "$SCRIPT_DIR/log"
EOF
    
    chmod +x "$test_dir/job.sh"
    echo "Created test directory: $test_dir"
done

# Create launch_all.sh script AFTER the loop
cat > "$BASE_DIR/launch_all.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Maximum number of parallel jobs (default: 10)
MAX_PARALLEL="${1:-10}"

echo "Launching up to $MAX_PARALLEL parallel jobs from $SCRIPT_DIR"

# Count how many test directories we have
NUM_JOBS=$(find "$SCRIPT_DIR" -maxdepth 1 -type d -name 'test_*' | wc -l)
echo "Found $NUM_JOBS test directories"

# Run each job in its own directory with up to $MAX_PARALLEL in parallel
find "$SCRIPT_DIR" -maxdepth 1 -type d -name 'test_*' | sort | \
  xargs -n1 -P "$MAX_PARALLEL" -I{} bash -c '
    dir="{}"
    echo "Starting job in $dir"
    cd "$dir" && ./job.sh
    echo "Job in $dir finished with exit code $?"
  '

echo "All jobs completed!"
echo "Check individual test directories for logs and results."
EOF

chmod +x "$BASE_DIR/launch_all.sh"
echo "Created launch script: $BASE_DIR/launch_all.sh"
echo "Run it with: ./launch_all.sh [number_of_parallel_jobs]"
echo "Example: cd $BASE_DIR && ./launch_all.sh 5"
