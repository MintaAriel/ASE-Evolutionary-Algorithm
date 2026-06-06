from ase.io import read
from ea.io.create_mol import (
    create_mols_from_unit_cell,
    filter_close_molecules_inplane,
    find_inplane_neighbors,
    make_mol_file_unit_cell,
    make_mol_files_inplane_pairs,
)
from ea.structures.ase_creator import NIH2ase, mol2ase2
from ase.visualize import view

CIF_PATH = '/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj/38.cif'
OUT_DIR = '/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj'
OUT_NAME = '38_molecules.mol'
PAIR_PREFIX = '38_pair'
INPLANE_THRESHOLD = 4.0
NEIGHBOR_MAX_DISTANCE = 9.0
NEIGHBOR_MAX_PERP = 2.0

atoms = read(CIF_PATH)
components, G, new_positions = create_mols_from_unit_cell(atoms, cov_radii=1.1)

print(f"Found {len(components)} connected components in {CIF_PATH}")
for i, comp in enumerate(components, 1):
    symbols = [atoms[j].symbol for j in comp]
    print(f"  Molecule {i}: {len(comp)} atoms ({''.join(sorted(symbols))})")

components, new_positions = filter_close_molecules_inplane(
    components, atoms, new_positions, threshold=INPLANE_THRESHOLD,
)
print(f"Kept {len(components)} molecules with in-plane centroid distance < {INPLANE_THRESHOLD} A")

make_mol_file_unit_cell(components, atoms, G, new_positions, OUT_DIR, filename=OUT_NAME)
print(f"Wrote {OUT_DIR}/{OUT_NAME}")

neighbors = find_inplane_neighbors(
    components, atoms, new_positions,
    max_distance=NEIGHBOR_MAX_DISTANCE, max_perp=NEIGHBOR_MAX_PERP,
)
print(f"Found {len(neighbors)} in-plane neighbor offsets (perp < {NEIGHBOR_MAX_PERP} A, r < {NEIGHBOR_MAX_DISTANCE} A)")
for nb in neighbors:
    print(f"  anchor={nb['anchor']} -> neighbor={nb['neighbor']} "
          f"shift={nb['shift_frac']} r={nb['distance']:.3f} perp={nb['perp']:.3f}")

written = make_mol_files_inplane_pairs(
    components, atoms, G, new_positions, neighbors, OUT_DIR, prefix=PAIR_PREFIX,
)
print(f"Wrote {len(written)} pair mol files:")
for fname, _ in written:
    print(f"  {fname}")

a = mol2ase2('/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj/38_molecules.mol')
view(a)
