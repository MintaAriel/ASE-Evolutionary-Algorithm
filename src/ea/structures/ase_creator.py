from ase.io import read
from io import StringIO
from ase import Atoms
import os
import pandas as pd
import json
from ase.data import chemical_symbols
from ase.visualize import view
import numpy as np


class mol2ase():
    def __init__(self, MOL_path):
        self.molecule_path = MOL_path

    def read(self):
        df = pd.read_csv(self.molecule_path,
                         skiprows=2,
                         sep='\s+',
                         header=None,
                         names=['atom_type', 'x', 'y', 'z',
                                'i', 'j', 'k', 'm', 'charge'])

        atoms = [value[0:2].replace("_", "") for value in df['atom_type'].tolist()]
        labels = []
        for index, symbol in enumerate(atoms):
            label = df['atom_type'].tolist()[index].replace(symbol, '', 1)
            labels.append(label)


        symbols = ''.join(atoms)

        matrix_data = df.to_numpy()

        positions = matrix_data[:,1:4]
        charge = np.nan_to_num(matrix_data[:,-1].astype(float) , nan=0.0)


        urea = Atoms(symbols=symbols, positions=positions, charges=charge)
        folder_path = os.path.dirname(self.molecule_path)

        with open(f'{folder_path}/labels.json', 'w') as f:
            json.dump(labels, f)
        return urea

class poscar2ase_molecule():
    def __init__(self, Molfile_dir, poscar_dir):
        self.Mol_file = Molfile_dir
        self.poscar_dir = poscar_dir
        self.df = self.read_mol(self.Mol_file)
        self.df_grouped = self.df.groupby('atom_type')
        self.Mol_poscar = None

    def read_mol(self, Mol):
        df = pd.read_csv(Mol,
                         skiprows=2,
                         sep='\s+',
                         header=None,
                         names=['atom_type', 'x', 'y', 'z',
                                'i', 'j', 'k', 'm', 'charge'])

        atoms = [value[0:2].replace("_", "") for value in df['atom_type'].tolist()]
        labels = []
        for index, symbol in enumerate(atoms):
            label = df['atom_type'].tolist()[index].replace(symbol, '', 1)
            labels.append(label)

        df['atom_type'] = atoms

        return df


    def create_molecule(self):
        if self.Mol_poscar == None:
            with open(self.poscar_dir, 'r') as file:
                mol = file.readlines()
        else:
            mol = self.Mol_poscar

        poscar = ''.join(mol)
        atoms_sym = mol[5]
        atoms_count = mol[6]


        symbols = atoms_sym.split()
        count = atoms_count.split()
        count_int = np.array([int(item) for item in count])
        n = 4
        single_mol_index = count_int/n


        df_new_index = pd.DataFrame()
        tags = []
        atom_groups = []
        symbol = []
        charges = []
        inside_index= []

        atom_index = 0

        for index, value in enumerate(single_mol_index):
            charge = self.df_grouped.get_group(symbols[index])['charge'].tolist()
            ins_index = self.df_grouped.get_group(symbols[index])['charge'].index.tolist()

            for i in range(n):
                low_index = atom_index
                atom_index += value
                top_index = atom_index
                tags.append(i)
                atom_groups.append([i for i in range(int(low_index), int(top_index))])
                symbol.append(symbols[index])
                charges.append(charge)
                inside_index.append(ins_index)


        df_new_index['tags'] = tags
        df_new_index['atom_groups'] = atom_groups
        df_new_index['symbol'] = symbol
        df_new_index['charges'] = charges
        df_new_index['inside_index'] = inside_index


        expanded_df = df_new_index.explode(['atom_groups', 'charges', 'inside_index'])

        # Reset index for clean ordering
        expanded_df = expanded_df.reset_index(drop=True)
        sorted_df = expanded_df.sort_values(['tags', 'inside_index'])

        # Reset index if desired
        sorted_df = sorted_df.reset_index(drop=True)

        # print(tabulate(df_new_index, headers='keys', tablefmt='psql', showindex=False))
        # print(tabulate(sorted_df, headers='keys', tablefmt='psql', showindex=True))

        sorted_index = sorted_df['atom_groups'].tolist()
        sorted_tags = sorted_df['tags'].tolist()
        sorted_charges = sorted_df['charges'].tolist()

        atom = read(StringIO(poscar), format='vasp')
        #view(atom)
        positions = atom.get_positions()
        sorted_positions = positions[sorted_index]

        atom.set_positions(sorted_positions)
        atom.set_tags(sorted_tags)
        atom.set_initial_charges(sorted_charges)
        atom.set_chemical_symbols(''.join(sorted_df['symbol'].tolist()))

        print(atom)
        view(atom)



# converter = poscar2ase_molecule(Molfile_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/poscar2ase_molcrys/MOL_1',
#                     poscar_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/poscar2ase_molcrys/best_uspex.vasp')
#
# converter.create_molecule()




# urea_mol = mol2ase( MOL_path='/home/vito/PythonProjects/ASEProject/CARLO/Carbamazepine/MOL_1')
# urea = urea_mol.read()
# view(urea)

def NIH2ase():


    with open('/home/vito/PythonProjects/ASEProject/CARLO/Carbamazepine/Conformer3D_COMPOUND_CID_2554.json', 'r') as f:
        data = json.load(f)

    # If your JSON is already a string
    # data = json.loads(json_string)

    # Extract the atoms data from the first compound
    atoms_data = data['PC_Compounds'][0]

    print(atoms_data.keys())
    positions_keys = ['x', 'y', 'z']

    xyz = atoms_data['coords'][0]['conformers'][0]
    positions = {key: xyz[key] for key in positions_keys if key in xyz}

    element = [chemical_symbols[i] for i in atoms_data['atoms']['element']]
    coords = pd.DataFrame(positions)
    bonds = pd.DataFrame(atoms_data['bonds'])

    coords.insert(loc=0, column='atom', value=element)

    num_bonds = len(atoms_data['bonds']['order'])

    connections = ''
    for bond in range(num_bonds):
        connections += 'connect  ' + str(atoms_data['bonds']['aid1'][bond]) + ' ' + str(
            atoms_data['bonds']['aid2'][bond]) + '\n'

    custom_header = f"Carbamazepine\nNumber of atoms: {len(coords)}\n"

    # with open('MOL_1', 'w') as f:
    #     f.write(custom_header)
    #     coords.to_csv(f, index=False, header=False, sep='\t')
    #
    # with open('connections', 'w') as f:
    #     f.write(connections)
    #
    bonds.to_json('bonds.json',
                  orient='records',
                  indent=4)

    print(atoms_data['bonds'])
    print(coords)
    print(connections)
