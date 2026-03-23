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
    args = parser.parse_args()

    n = args.n_threads

    # Set thread limits *before* importing torch
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

    import torch
    torch.set_num_threads(n)

    from ase.io import read
    from mattersim_benchmark.benchmark import MatterSimTester

    # Load structure and model
    atoms = read(args.structure)
    tester = MatterSimTester(args.model, device=args.device)

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
        "fire_steps": int(result["fire_steps"]),
        "lbfgs_steps": int(result["lbfgs_steps"]),
    }

    # Use a marker so the orchestrator can find the JSON line
    # even if libraries printed to stdout during import
    print("__WORKER_JSON__" + json.dumps(output))


if __name__ == "__main__":
    main()
