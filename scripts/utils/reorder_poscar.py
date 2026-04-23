#!/usr/bin/env python3

import sys
from ase.io import read, write

def main():
    if len(sys.argv) != 2:
        print("Usage: reorder_poscar.py /path/to/POSCAR")
        sys.exit(1)

    input_path = sys.argv[1]

    # read structure
    atoms = read(input_path)

    # desired order (must match POTCAR!)
    order = ['C', 'O', 'N', 'H']

    # check for unknown elements
    symbols = set(atoms.get_chemical_symbols())
    unknown = symbols - set(order)
    if unknown:
        raise ValueError(f"Found elements not in order list: {unknown}")

    # reorder atoms
    atoms_sorted = atoms[[i for i, a in sorted(
        enumerate(atoms),
        key=lambda x: order.index(x[1].symbol)
    )]]

    # write POSCAR in current directory
    write("POSCAR", atoms_sorted, format="vasp")

    print("Reordered POSCAR written to ./POSCAR")

if __name__ == "__main__":
    main()