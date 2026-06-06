from ase.neighborlist import NeighborList
from ase.data import covalent_radii
import networkx as nx
import numpy as np
from ase import Atoms
from rdkit import Chem
from pathlib import Path
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem


def create_mol_from_asu(supercell, cov_radii=1.2):
    atoms_sc = supercell.copy()
    cutoffs = [covalent_radii[a.number] * cov_radii for a in atoms_sc]  # 1.1–1.3 is typical
    nl = NeighborList(cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms_sc)


    G = nx.Graph()

    for i in range(len(atoms_sc)):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            G.add_edge(i, j)

    components = list(nx.connected_components(G))

    mol_indices = max(components, key=len)
    positions = atoms_sc.get_positions()
    cell = atoms_sc.get_cell()

    new_positions = positions.copy()

    visited = set()
    stack = [list(mol_indices)[0]]
    visited.add(stack[0])

    while stack:
        i = stack.pop()
        indices, offsets = nl.get_neighbors(i)

        for j, offset in zip(indices, offsets):
            if j in mol_indices and j not in visited:
                new_positions[j] = new_positions[i] + (positions[j] - positions[i]) + np.dot(offset, cell)
                visited.add(j)
                stack.append(j)

    mol = Atoms(
        symbols=[atoms_sc[i].symbol for i in mol_indices],
        positions=new_positions[list(mol_indices)],
        pbc=False
    )

    return mol, mol_indices, G, new_positions


def make_mol_file(mol_indices, atoms_sc, G, new_positions, out_dir):

    mol_rdkit = Chem.RWMol()

    # Map ASE indices → RDKit indices
    idx_map = {}

    for i in mol_indices:
        atom = atoms_sc[i]
        rd_atom = Chem.Atom(atom.symbol)
        idx = mol_rdkit.AddAtom(rd_atom)
        idx_map[i] = idx

    # Add bonds from graph
    for i in mol_indices:
        for j in G.neighbors(i):
            if j in mol_indices and i < j:
                mol_rdkit.AddBond(idx_map[i], idx_map[j], Chem.BondType.SINGLE)


    conf = Chem.Conformer(len(idx_map))

    for i, rd_i in idx_map.items():
        x, y, z = new_positions[i]
        conf.SetAtomPosition(rd_i, Point3D(x, y, z))

    mol_rdkit.AddConformer(conf)

    Chem.MolToMolFile(mol_rdkit, Path(out_dir) / "molecule.mol")


def create_mols_from_unit_cell(supercell, cov_radii=1.2):
    atoms_sc = supercell.copy()
    cutoffs = [covalent_radii[a.number] * cov_radii for a in atoms_sc]
    nl = NeighborList(cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms_sc)

    G = nx.Graph()
    G.add_nodes_from(range(len(atoms_sc)))

    for i in range(len(atoms_sc)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            G.add_edge(i, j)

    components = [set(c) for c in nx.connected_components(G)]
    positions = atoms_sc.get_positions()
    cell = atoms_sc.get_cell()
    new_positions = positions.copy()

    for comp in components:
        start = next(iter(comp))
        visited = {start}
        stack = [start]
        while stack:
            i = stack.pop()
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if j in comp and j not in visited:
                    new_positions[j] = new_positions[i] + (positions[j] - positions[i]) + np.dot(offset, cell)
                    visited.add(j)
                    stack.append(j)

    return components, G, new_positions


def filter_close_molecules_inplane(components, atoms_sc, new_positions, threshold=4.0, plane_normal=None, search=1):
    cell = np.asarray(atoms_sc.get_cell())
    if plane_normal is None:
        plane_normal = cell[2]
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)

    centroids = [new_positions[list(c)].mean(axis=0) for c in components]
    shifted_positions = new_positions.copy()
    kept = [0]
    rng = range(-search, search + 1)

    for j in range(1, len(components)):
        best = None
        for a in rng:
            for b in rng:
                for k in rng:
                    shift = a * cell[0] + b * cell[1] + k * cell[2]
                    d = (centroids[j] + shift) - centroids[0]
                    d_perp = d - np.dot(d, n) * n
                    perp = float(np.linalg.norm(d_perp))
                    if best is None or perp < best[0]:
                        best = (perp, shift)
        if best is not None and best[0] < threshold:
            kept.append(j)
            for i in components[j]:
                shifted_positions[i] = new_positions[i] + best[1]

    kept_components = [components[i] for i in kept]
    return kept_components, shifted_positions


def molecule_plane_normal(comp, atoms_sc, positions):
    idxs = [i for i in comp if atoms_sc[i].symbol != 'H']
    if len(idxs) < 3:
        idxs = list(comp)
    pts = positions[idxs]
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[-1]


def find_inplane_neighbors(components, atoms_sc, new_positions,
                           max_distance=9.0, max_perp=2.0, search=2):
    cell = np.asarray(atoms_sc.get_cell())
    centroids = [new_positions[list(c)].mean(axis=0) for c in components]
    normals = [molecule_plane_normal(c, atoms_sc, new_positions) for c in components]

    rng = range(-search, search + 1)
    neighbors = []
    for i in range(len(components)):
        n_i = normals[i] / np.linalg.norm(normals[i])
        for j in range(len(components)):
            for a in rng:
                for b in rng:
                    for k in rng:
                        if i == j and a == 0 and b == 0 and k == 0:
                            continue
                        shift = a * cell[0] + b * cell[1] + k * cell[2]
                        d = (centroids[j] + shift) - centroids[i]
                        r = float(np.linalg.norm(d))
                        if r < 0.1 or r > max_distance:
                            continue
                        perp = float(abs(np.dot(d, n_i)))
                        if perp > max_perp:
                            continue
                        inplane = float(np.sqrt(max(0.0, r * r - perp * perp)))
                        neighbors.append({
                            'anchor': i, 'neighbor': j, 'shift_frac': (a, b, k),
                            'shift': shift, 'distance': r,
                            'perp': perp, 'inplane': inplane,
                        })
    return neighbors


def make_mol_file_pair(comp_a, comp_b, atoms_sc, G, pos_a, pos_b, out_dir, filename):
    mol_rdkit = Chem.RWMol()
    idx_map_a, idx_map_b = {}, {}

    for i in comp_a:
        idx_map_a[i] = mol_rdkit.AddAtom(Chem.Atom(atoms_sc[i].symbol))
    for i in comp_b:
        idx_map_b[i] = mol_rdkit.AddAtom(Chem.Atom(atoms_sc[i].symbol))

    for idx_map, comp in [(idx_map_a, comp_a), (idx_map_b, comp_b)]:
        for i in comp:
            for j in G.neighbors(i):
                if j in comp and i < j:
                    mol_rdkit.AddBond(idx_map[i], idx_map[j], Chem.BondType.SINGLE)

    conf = Chem.Conformer(len(idx_map_a) + len(idx_map_b))
    for i, rd_i in idx_map_a.items():
        x, y, z = pos_a[i]
        conf.SetAtomPosition(rd_i, Point3D(x, y, z))
    for i, rd_i in idx_map_b.items():
        x, y, z = pos_b[i]
        conf.SetAtomPosition(rd_i, Point3D(x, y, z))

    mol_rdkit.AddConformer(conf)
    Chem.MolToMolFile(mol_rdkit, str(Path(out_dir) / filename))


def make_mol_files_inplane_pairs(components, atoms_sc, G, new_positions, neighbors,
                                 out_dir, prefix="pair", deduplicate=True):
    written = []
    seen = set()
    for idx, nb in enumerate(neighbors):
        i, j = nb['anchor'], nb['neighbor']
        a, b, k = nb['shift_frac']
        if deduplicate:
            key1 = (i, j, (a, b, k))
            key2 = (j, i, (-a, -b, -k))
            canonical = min(key1, key2)
            if canonical in seen:
                continue
            seen.add(canonical)

        pos_b = new_positions.copy()
        for atom_idx in components[j]:
            pos_b[atom_idx] = new_positions[atom_idx] + nb['shift']

        fname = f"{prefix}_{len(written):02d}_a{i}_n{j}_{a}{b}{k}.mol"
        make_mol_file_pair(components[i], components[j], atoms_sc, G,
                           new_positions, pos_b, out_dir, fname)
        written.append((fname, nb))
    return written


def make_mol_file_unit_cell(components, atoms_sc, G, new_positions, out_dir, filename="unit_cell.mol"):
    mol_rdkit = Chem.RWMol()
    idx_map = {}

    for comp in components:
        for i in comp:
            atom = atoms_sc[i]
            rd_atom = Chem.Atom(atom.symbol)
            idx = mol_rdkit.AddAtom(rd_atom)
            idx_map[i] = idx

    for comp in components:
        for i in comp:
            for j in G.neighbors(i):
                if j in comp and i < j:
                    mol_rdkit.AddBond(idx_map[i], idx_map[j], Chem.BondType.SINGLE)

    conf = Chem.Conformer(len(idx_map))
    for i, rd_i in idx_map.items():
        x, y, z = new_positions[i]
        conf.SetAtomPosition(rd_i, Point3D(x, y, z))

    mol_rdkit.AddConformer(conf)

    Chem.MolToMolFile(mol_rdkit, str(Path(out_dir) / filename))