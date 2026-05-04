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