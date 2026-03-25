"""Subprocess worker: runs a single MatterSim benchmark and prints JSON."""

import os
import sys
import json
import time
import resource
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--container-root", default=None,
                        help="Path to container_cpu_2 dir. Enables D3 via SIF.")
    parser.add_argument("--input-template", default="input_mattersim_d3.py",
                        help="Template filename inside container_root/templates/.")
    args = parser.parse_args()

    n = args.n_threads

    # Set thread limits *before* importing torch
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

    if args.container_root is None:
        import torch
        torch.set_num_threads(n)

    from ase.io import read
    from ea.analysis.benchmark_mattersim import MatterSimTester

    # Load structure and model
    atoms = read(args.structure)
    tester = MatterSimTester(
        args.model,
        device=args.device,
        container_root=args.container_root,
        input_template=args.input_template,
        n_threads=n,
    )

    # Benchmark
    start = time.perf_counter()
    result = tester.relax(atoms)
    elapsed = time.perf_counter() - start

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    output = {
        "n_threads": n,
        "elapsed_time": round(elapsed, 4),
        "peak_rss_kb": int(peak_rss_kb),
        "peak_rss_mb": round(float(peak_rss_kb) / 1024, 2),
        "initial_energy": float(result["initial_energy"]),
        "final_energy": float(result["final_energy"]),
        "total_steps": int(result["total_steps"]),
        "fire_steps": result.get("fire_steps"),
        "lbfgs_steps": result.get("lbfgs_steps"),
    }

    # Use a marker so the orchestrator can find the JSON line
    # even if libraries printed to stdout during import
    print("__WORKER_JSON__" + json.dumps(output))


if __name__ == "__main__":
    main()
