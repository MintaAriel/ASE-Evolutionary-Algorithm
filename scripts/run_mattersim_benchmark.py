"""Run MatterSim benchmark across different CPU thread counts.

Usage
-----
    conda activate ase_env
    python scripts/run_mattersim_benchmark.py
    python scripts/run_mattersim_benchmark.py --model /path/to/model.pth --structure /path/to/struct.cif
    python scripts/run_mattersim_benchmark.py --min-cores 1 --max-cores 8 --device cpu
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mattersim_benchmark.benchmark import run_all_benchmarks


def main():
    project_root = Path(__file__).resolve().parent.parent

    default_model = (
        "/home/vito/PythonProjects/ASEProject/container_gpu_2/"
        "models/tuned_mattersim_12.03.2026.pth"
    )
    default_structure = (
        "/home/vito/PythonProjects/ASEProject/container_gpu_2/"
        "structures/1039798.cif"
    )
    default_output = str(project_root / "test" / "matersim_benchmark")

    parser = argparse.ArgumentParser(
        description="Benchmark MatterSim relaxation across CPU thread counts"
    )
    parser.add_argument(
        "--model", default=default_model,
        help="Path to MatterSim checkpoint (.pth)",
    )
    parser.add_argument(
        "--structure", default=default_structure,
        help="Path to input structure (CIF, POSCAR, …)",
    )
    parser.add_argument("--min-cores", type=int, default=1)
    parser.add_argument("--max-cores", type=int, default=15)
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Torch device for MatterSim (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", default=default_output,
        help="Directory for CSV and plots",
    )
    args = parser.parse_args()

    run_all_benchmarks(
        model_path=args.model,
        structure_path=args.structure,
        cores_range=range(args.min_cores, args.max_cores + 1),
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
