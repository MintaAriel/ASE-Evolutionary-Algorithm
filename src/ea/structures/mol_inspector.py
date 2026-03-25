from ase.io import read, write
from ase.build.supercells import make_supercell
import numpy as np
import os
from ase.visualize import view
from ase.ga.data import DataConnection
from ase.filters import Filter
import pandas as pd
from ase_creator import mol2ase


class Molecule_inspector():
    def __init__(self,conections_dir, Mol_dir, bond_change, tag):
        self.connections = conections_dir
        self.connections_df = self.open_connections(self.connections)
        self.Mol_dir = Mol_dir
        self.bond_change = bond_change
        self.tag = tag
        self.orig_dis_df = self.original_inter_dis()

    def original_inter_dis(self):
        convertor = mol2ase(self.Mol_dir)
        original_mol = convertor.read()
        orig_dis_df = self.connections_df.copy(deep=True)

        orig_dis_df['distance'] = [
            original_mol.get_distance(i - 1, j - 1, mic=True)
            for i, j in zip(orig_dis_df['atom1'], orig_dis_df['atom2'])
        ]

        return orig_dis_df


    def open_connections(self, connections_dir):

        """Read connection file into DataFrame."""
        # Read with space/tab delimiter, skip initial whitespace
        df = pd.read_csv(connections_dir,
                         sep='\s+',  # Handles spaces/tabs
                         header=None,            # No header in file
                         names=['connect', 'atom1', 'atom2'])
        return df


    def make_supercell_(self, atom, a, b, c):

        transformation_matrix = np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ])
        super_cell_general = make_supercell(atom, transformation_matrix)
        return super_cell_general

    def get_inter_distances(self, atom):
        single_molecule = (atom.get_tags() == self.tag)
        hydrogen_filter = Filter(atom, mask=single_molecule)

        filtered_atom = atom.__getitem__(single_molecule)


        inter_dis_df = self.connections_df.copy(deep=True)

        inter_dis_df['distance'] = [
            filtered_atom.get_distance(i-1, j-1, mic=True)
            for i, j in zip(inter_dis_df['atom1'], inter_dis_df['atom2'])
        ]


        # print(inter_dis_df)
        # print(filtered_atom)
        return inter_dis_df

    def bond_inspection(self, atom):

        inter_dis_df = self.get_inter_distances(atom)
        # print(self.orig_dis_df)
        # print(inter_dis_df)

        bond_change = self.orig_dis_df['distance'] - inter_dis_df['distance']

        any_gt_03 = (bond_change.abs() > 0.3).any()
        # Get indices where condition is True
        indices = bond_change[bond_change.abs() > 0.3].index.tolist()

        #print(f"Any > {self.bond_change}: {any_gt_03}")
        if any_gt_03 == True:
            print(f"Bonds {indices} changed more than {self.bond_change} A \nThe Structure may be damaged")

        return any_gt_03
        # print(bond_change)


inspector = Molecule_inspector(conections_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/gen_track/conections',
                               Mol_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/gen_track/MOL_1',
                               bond_change=0.4,
                               tag=2)


# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/Zahra_nicotinamide/nicotinamide_zahra.db')
# atom = da.get_atoms(54)

# da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/Genesis/try/Nicotinamide_2/nicotinamide_1/nicotinamide_zahra.db')
# for i in range(50,100):
#     atom = da.get_atoms(i)
#     chek_molecule = inspector.bond_inspection(atom)
#     if chek_molecule == True:
#         print(i)
#         print(atom.info['key_value_pairs']['origin'])
#
#     else:
#         if atom.info['key_value_pairs']['origin'] == 'StrainMutation':
#             print('THIS IS A GOOD STRAIN MUT', i)


