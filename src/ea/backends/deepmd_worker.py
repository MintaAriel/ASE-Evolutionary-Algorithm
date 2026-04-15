#!/usr/bin/env python3
"""DeepMD worker process — runs inside deepmd_env (no ASE needed).

Reads JSON requests from stdin, computes energy/forces/virial
via ``DeepPot.eval``, and writes JSON responses to stdout.

Usage:
    python deepmd_worker.py <model_path> [--device cpu|cuda]
"""
import sys
import json
import argparse

import numpy as np
from deepmd.infer import DeepPot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the DeepMD model file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for inference (default: cpu)")
    args = parser.parse_args()

    dp = DeepPot(args.model_path)
    type_map = dp.get_type_map()

    print(json.dumps({"status": "ready", "type_map": type_map}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)

            coords = np.array(data["coords"]).reshape(1, -1)
            cells = np.array(data["cell"]).reshape(1, 9)

            symbols = data["symbols"]
            atom_types = [type_map.index(s) for s in symbols]

            energy, forces, virial = dp.eval(coords, cells, atom_types)

            # virial (1,9) -> stress (6,) in Voigt order, divided by volume
            vol = abs(np.linalg.det(np.array(data["cell"]).reshape(3, 3)))
            vir = virial[0].reshape(3, 3)
            stress_voigt = np.array([
                vir[0, 0], vir[1, 1], vir[2, 2],
                vir[1, 2], vir[0, 2], vir[0, 1],
            ]) / vol

            result = {
                "energy": float(energy[0, 0]),
                "forces": forces[0].tolist(),
                "stress": stress_voigt.tolist(),
            }
            print(json.dumps(result), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
