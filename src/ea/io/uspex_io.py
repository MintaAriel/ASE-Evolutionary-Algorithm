import pandas as pd
from ase.io import read
import os
from io import StringIO
import numpy as np
import sqlite3


def convert_to_list(x):
    # Remove the brackets and split by commas
    elements = x.strip('[]').split(',')

    # Convert to integer while ignoring empty strings
    int_list = [int(e) for e in elements if e]

    # Create a NumPy array from the list
    np_array = np.array(int_list)

    return np_array

# Opening Individuals
def indiv_to_df(indiv_dir, uspex_ver='10.5'):
    with open(indiv_dir, "r", encoding="utf-8") as file:
        content = file.readlines()
        if uspex_ver == '10.5':
            content.pop(1)
        content_str = ''.join(content)

    # Replace spaces inside brackets with commas
    converted_str = ''
    inside_brackets = False
    for char in content_str:
        if char == '[':
            inside_brackets = True
            converted_str += char
        elif char == ']':
            inside_brackets = False
            converted_str += char
        elif inside_brackets and char == ' ':
            converted_str += ','
        else:
            converted_str += char

    # Use StringIO to simulate a file object
    data_io = StringIO(converted_str)

    #Columns we will use from results
    cols_to_use = ['Gen','ID', 'Origin', 'Enthalpy', 'Volume', 'SYMM' ]
    #optional columns [Composition Density   Fitness   KPOINTS  Q_entr A_order S_order]

    # Read the data into a DataFrame, handling whitespace more flexibly
    df = pd.read_csv(data_io, sep='\s+', skiprows=0, usecols=cols_to_use)

    df['Enthalpy'] = df['Enthalpy'].replace(100000, np.nan)

    if 'Composition' in cols_to_use:
        df['Composition'] = df['Composition'].apply(lambda x: convert_to_list(x))
        # df['N_atoms'] = df['Composition'].apply(lambda x: np.sum(x))
        # df['Energy/N_atoms'] = df['Enthalpy'] / df['N_atoms']
    if 'KPOINTS' in cols_to_use:
        df['KPOINTS'] = df['KPOINTS'].apply(lambda x: convert_to_list(x))

    df.rename(columns={'Gen': 'generation', 'Enthalpy':'energy', 'Origin':'operator',
                       'SYMM':'symmetry', 'ID':'id', 'Volume': 'volume'}, inplace=True)

    return (df)


class best_POSCAR():
    def __init__(self, best_poscar_dir):
        self.poscar_dir = best_poscar_dir

    def get_best(self):
        # atom = read(os.path.expanduser("~/uspex-python/results1/best_structure_POSCAR"))

        # view(atom)

        data = {}
        with open(self.poscar_dir, 'r') as file:
            content = file.read()
            lines_list = content.splitlines()


            poscar_line = [i for i, line in enumerate(lines_list) if 'EA' in line]


            poscar_line.append(len(lines_list))

        print(len(lines_list))
        print(poscar_line)
        for idx,line in enumerate(poscar_line):
            if idx != len(poscar_line)-1:

                print(line)
                print(poscar_line[idx + 1])
                print(idx)
                data[f'{lines_list[line].split()[0]}'] = '\n'.join(lines_list[line:poscar_line[idx + 1]])

        return data

    def join_poscar(self):
        path = '/home/vito/PythonProjects/ASEProject/EA/new_minimum/ALL_BEST'
        seeds_dir = os.listdir(path)
        all_seeds = None
        for dir in seeds_dir:
            seeds_dir = os.path.join(path, dir)
            best = read(seeds_dir, format="vasp")
            all_seeds += best

        with open("ALL_BEST.vasp", "w") as f:
            f.write(all_seeds)

class parse_tests():
    def __init__(self, tests_dir, out_dir='.'):
        self.tests_dir = tests_dir
        self.out_dir = out_dir

    def individuals(self, type='best', db=False):
        if type == 'best':
            work_dir = os.path.join(self.tests_dir, 'BESTIndividuals')
        elif type == 'all':
            work_dir = os.path.join(self.tests_dir, 'Individuals')
        elements = os.listdir(work_dir)
        dfs = []
        for file in elements:
            df_test_dir = os.path.join(work_dir,file)
            #to split the number of the test from the folder "BESTIndividuals_test_5"
            test_n = file.split('_')[-1]

            df_test = indiv_to_df(df_test_dir)
            df_test['run'] = int(test_n)
            dfs.append(df_test)

        df = pd.concat(dfs, ignore_index=True)
        if db:
            conn = sqlite3.connect(os.path.join(self.out_dir,f'{type}_Individuals.db'))
            df.to_sql('results', conn, if_exists='replace', index=False)
            conn.close()

        return df

    def best_gathered_poscars(self):
        work_dir = os.path.join(self.tests_dir, 'BESTgatheredPOSCARS')
        elements = os.listdir(work_dir)
        for file in elements:
            print(file.split('_')[-1])
        ...
    def gathered_poscars(self):
        work_dir = os.path.join(self.tests_dir, 'gatheredPOSCARS')
        ...

    def gathered_pos_unrelaxed(self):
        work_dir = os.path.join(self.tests_dir, 'gatheredPOSCARS_unrelaxed')
        ...


