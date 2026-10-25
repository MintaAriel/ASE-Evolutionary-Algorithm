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

# USPEX output formats understood by this module.
# '10.5'  : classic Matlab USPEX (Gen ID Origin Composition Enthalpy ... header + units line,
#           POSCAR blocks headed by "EA<id> ...")
# '2126'  : python USPEX (2026 release, a.k.a. USPEX26): lowercase header
#           (generation number num_atoms_all energy cell_volume space_group origin parents ...),
#           POSCAR blocks headed by "number=<id>, energy=..."
USPEX_NEW_VERSIONS = ('2126', '2026', '26')


def _is_new_uspex(uspex_ver):
    return str(uspex_ver) in USPEX_NEW_VERSIONS


def _brackets_to_commas(content_str):
    # Replace spaces inside brackets with commas so read_csv sees one token per list
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
    return converted_str


# Opening Individuals
def indiv_to_df(indiv_dir, uspex_ver='10.5'):
    """Parse an USPEX Individuals/BESTIndividuals file into a DataFrame.

    uspex_ver: '10.5' (default, Matlab USPEX) or '2126' (python USPEX 2026; '2026' and '26'
    are accepted as aliases). For '2126' the extra columns 'composition', 'natoms' and
    'parents' (np.array of parent ids, empty for random/first-generation structures) are returned.
    """
    if _is_new_uspex(uspex_ver):
        return _indiv_to_df_2126(indiv_dir)

    with open(indiv_dir, "r", encoding="utf-8") as file:
        content = file.readlines()
        if uspex_ver == '10.5':
            content.pop(1)
        content_str = ''.join(content)

    converted_str = _brackets_to_commas(content_str)

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


def _indiv_to_df_2126(indiv_dir):
    with open(indiv_dir, "r", encoding="utf-8") as file:
        content_str = file.read()

    # ", " inside brackets ([67, 7]) -> "[67,,7]"; convert_to_list ignores the empty tokens
    data_io = StringIO(_brackets_to_commas(content_str))
    df = pd.read_csv(data_io, sep=r'\s+')

    df.rename(columns={'number': 'id', 'Generation': 'generation', 'origin': 'operator', 'space_group': 'symmetry',
                       'cell_volume': 'volume', 'num_atoms_all': 'composition'}, inplace=True)

    df['composition'] = df['composition'].apply(convert_to_list)
    df['natoms'] = df['composition'].apply(lambda x: int(np.sum(x)))
    df['parents'] = df['parents'].apply(convert_to_list)
    df['energy'] = df['energy'].replace(100000, np.nan)

    return df


def get_structure_from_id(poscar_dir, id_structures : list [1], all=False, uspex_ver='10.5'):
    """Return {id: POSCAR string} for the requested ids of a gatheredPOSCARS file.

    uspex_ver='2126' reads the python-USPEX format whose blocks start with "number=<id>, ...";
    there the keys are always int ids and `all=True` returns every structure in the file.
    """
    if _is_new_uspex(uspex_ver):
        return _get_structure_from_id_2126(poscar_dir, id_structures, all=all)

    id_names = {f"EA{v} " for v in id_structures}

    data = {}
    with open(poscar_dir, 'r') as file:
        content = file.read()
        lines_list = content.splitlines()
        if all:
            poscar_line = [i for i, line in enumerate(lines_list) if 'EA' in line]
            poscar_line.append(len(lines_list))

            for idx, line in enumerate(poscar_line):
                if idx != len(poscar_line) - 1:
                    data[f'{lines_list[line].split()[0]}'] = '\n'.join(lines_list[line:poscar_line[idx + 1]])

        else:
            poscar_line_ini = []
            poscar_line_end = []
            added_structure = 0

            for i, line in enumerate(lines_list):
                if any(name in line for name in id_names):
                    poscar_line_ini.append(i)
                    added_structure = 1

                if added_structure == 1 and 'EA' in lines_list[i+1]:
                    poscar_line_end.append(i+1)
                    added_structure = 0

            for idx, id in enumerate(sorted(id_structures)):
                data[id] =  '\n'.join(lines_list[poscar_line_ini[idx]:poscar_line_end[idx]])

    return data

def _get_structure_from_id_2126(poscar_dir, id_structures, all=False):
    wanted = None if all else {int(v) for v in id_structures}

    data = {}
    with open(poscar_dir, 'r') as file:
        lines_list = file.read().splitlines()

    starts = [i for i, line in enumerate(lines_list) if line.startswith('number=')]
    starts.append(len(lines_list))
    for ini, end in zip(starts[:-1], starts[1:]):
        id_ = int(lines_list[ini].split(',')[0].split('=')[1])
        if wanted is None or id_ in wanted:
            data[id_] = '\n'.join(lines_list[ini:end])

    if wanted is not None:
        missing = wanted - set(data)
        if missing:
            raise KeyError(f'ids not found in {poscar_dir}: {sorted(missing)}')
    return data


def poscar_to_atoms(poscar_str):
    """Convert one POSCAR block returned by get_structure_from_id into ase.Atoms."""
    return read(StringIO(poscar_str), format='vasp')


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
        self.folder_name =  os.path.basename(self.tests_dir)

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
            print('saving:',os.path.join(self.out_dir,f'{type}_{self.folder_name}.db'))
            conn = sqlite3.connect(os.path.join(self.out_dir,f'{type}_{self.folder_name}.db'))
            df.to_sql('results', conn, if_exists='replace', index=False)
            conn.close()

        return df

    def best_gathered_poscars(self):
        work_dir = os.path.join(self.tests_dir, 'BESTgatheredPOSCARS')
        elements = os.listdir(work_dir)
        for file in elements:
            print(file.split('_')[-1])
        ...
    def gathered_poscars(self, test_n: int = 1, structures_id :list = [1]):
        work_dir = os.path.join(self.tests_dir, 'gatheredPOSCARS', f'gatheredPOSCARS_test_{test_n}')
        poscars_dictionary = get_structure_from_id(work_dir, structures_id)
        return poscars_dictionary
        ...

    def gathered_pos_unrelaxed(self):
        work_dir = os.path.join(self.tests_dir, 'gatheredPOSCARS_unrelaxed')
        ...


