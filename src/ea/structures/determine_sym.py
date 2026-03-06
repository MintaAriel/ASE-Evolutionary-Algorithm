
# After generating crystal, check symmetry
import pandas as pd
from pymatgen.core import Structure
from ase import Atoms
from ase.ga.data import DataConnection
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from ase.filters import Filter
from ase.visualize import view
from ase.io import read

import numpy as np
from ase import Atoms
from relax import Gulp_relaxation_noadd
from relax import Gulp_relaxation


#
# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/theophilline.db')
# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/carbamazepine/database/carbamazepine.db')
da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/database/theophylline_8.db')
# # da = DataConnection('/home/vito/Downloads/Mg4Al8O16_401.db')



def exp_so3(w):
    '''
    Returns a rotational matrix given an array of the torque in x,y,z in R0
    :param w: skew matrix
    :return: rotational matrix 3x3
    '''
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    W = skew(w/theta)
    return (
        np.eye(3)
        + np.sin(theta) *W
        + (1- np.cos(theta)) * (W @ W)
    )

def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])



# atom = read('/home/vito/PythonProjects/ASEProject/EA/test/struc-gen/LJ/CalcFold/out.cif')
def get_atributes(atoms, tags_index):
    tags = atoms.get_tags()
    mask = np.isin(tags, list(tags_index))
    ASU = atoms.__getitem__(mask)
    print(ASU)

    asu_tags = ASU.get_tags()
    uniq_tags = np.unique(asu_tags)
    mol_per_ASU = len(uniq_tags)

    atoms_per_mol  = int(len(asu_tags)/len(uniq_tags))

    positions = ASU.get_positions().reshape(mol_per_ASU,atoms_per_mol,3)
    R0 = positions.mean(axis=1)
    R0_expanded = R0[:, None, :]  # shape (n-mol, 1, 3)



    all_forces = np.empty(shape=(len(ASU), 3), dtype=float)
    forces = all_forces.reshape(mol_per_ASU, atoms_per_mol, 3)
    gradient_R0 = -np.sum(forces, axis=1)

    r_rel = positions - R0_expanded  # shape (n-mol, 30, 3)

    torque = np.sum(np.cross(r_rel, forces), axis=1)
    displac_grad = np.zeros((mol_per_ASU, 3))
    rot_mat = np.repeat(np.eye(3)[None, :, :], mol_per_ASU, axis=0)

    eta_t = 20e-4
    eta_r = 5e-6

    displac_grad += -eta_t * gradient_R0
    for k in range(mol_per_ASU):
        rot_mat[k] = exp_so3(eta_r * torque[k]) @ rot_mat[k]





    print('forces', forces)
    print('gradient_ro', gradient_R0)

    print('positions', positions)
    print('ro', R0)

    print('torque',torque)
    print('displacement', displac_grad)
    print('rotational matrix',rot_mat)

def define_ASUf(crystal):
    pmg_structure = AseAtomsAdaptor.get_structure(crystal)
    # Assuming you have a pymatgen Structure object
    analyzer = SpacegroupAnalyzer(pmg_structure)
    similar_atoms = analyzer.get_symmetry_dataset().equivalent_atoms


    # print( analyzer.get_symmetry_dataset())


    unique_items = len(np.unique(similar_atoms))


    print( similar_atoms,unique_items)
    tags = crystal.get_tags()
    tags_unique = tags[np.unique(similar_atoms)]
    asu_tags = np.unique(tags_unique)
    print(asu_tags)

    tags_idx = np.unique(similar_atoms)

    print('tags idx', asu_tags)


    # get_atributes(crystal, asu_tags)


    print('spacegroup: ',analyzer.get_symmetry_dataset().number)




    n_molecules = len(np.unique(tags))
    n_atoms =  len(crystal)




    mol_in_asym = n_molecules/(n_atoms/unique_items)
    print(mol_in_asym)

    mask = np.isin(tags, asu_tags)
    asu = crystal.__getitem__(mask)
    print(asu)

    # view(asu)

    return  analyzer.get_symmetry_dataset(), asu

def define_ASU(crystal):

    pmg_structure = AseAtomsAdaptor.get_structure(crystal)
    analyzer = SpacegroupAnalyzer(pmg_structure)

    refined = analyzer.get_refined_structure()
    dataset = analyzer.get_symmetry_dataset()

    refined_ase = AseAtomsAdaptor.get_atoms(refined)

    equiv = dataset.equivalent_atoms
    asu_indices = np.unique(equiv)

    asu = refined_ase[asu_indices]

    return dataset, asu, refined_ase

def build_full_cell(asu_atoms, dataset):

    cell = asu_atoms.cell.array
    Hinv = np.linalg.inv(cell)

    # fractional coordinates
    frac = asu_atoms.get_positions() @ Hinv.T

    rotations = dataset.rotations
    translations = dataset.translations

    all_frac = []

    for R, t in zip(rotations, translations):

        f_new = frac @ R.T + t
        f_new = f_new % 1.0
        all_frac.append(f_new)

    all_frac = np.vstack(all_frac)

    # back to Cartesian
    cart = all_frac @ cell

    full = Atoms(
        symbols=list(asu_atoms.symbols) * len(rotations),
        positions=cart,
        cell=cell,
        pbc=True
    )

    return full

from pymatgen.symmetry.groups import SpaceGroup

def get_full_sym_cell( ASU):
    '''
    computes the full cell using symmetric operations of its spacegroup.
    :param ASU_new: ASE atoms cell with the asymmetric cell
    :return: full cell
    '''

    sg = SpaceGroup.from_int_number(6)

    ASU_positions = ASU.get_scaled_positions()

    all_positions = []

    ops = list(sg.symmetry_ops)
    for i, op in enumerate(ops):
        op = ops[i]
        W = op.rotation_matrix
        t = op.translation_vector

        pos_sym = ASU_positions @ W.T + t
        pos_sym %= 1
        all_positions.append(pos_sym)

    new_positions = np.concatenate(all_positions, axis=0)
    new_symbols = list(ASU.symbols) *2

    print(len(new_symbols), len(new_positions))
    new_crystal = Atoms(cell=ASU.cell, scaled_positions=new_positions, symbols=new_symbols)

    return new_crystal


for i in range(2,10):
    print(i)
    atom = da.get_atoms(i)
    dataset, asu = define_ASUf(atom)
    # full = build_full_cell(asu, dataset)
    # full = get_full_sym_cell(asu)
    # view(asu)
    # view(full)








