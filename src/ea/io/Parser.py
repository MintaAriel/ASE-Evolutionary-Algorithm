

import numpy as np
from pymatgen.io.cif import CifParser
import nglview as nv

import pandas as pd
from io import StringIO
from tabulate import tabulate

with open("/home/brian/USPEX_work/EX02/results3/symmetrized_structures.cif", "r") as f:
    cif_str = f.read()
    print(type(cif_str))

structure_number = []
symmetry_number = []

for index,line, in enumerate(cif_str.splitlines()):
    if line.strip().startswith("data_findsym-STRUC-"):
        structure_number.append(int(line.strip().split("-")[-1]))
        print(index)
    if line.strip().startswith("_symmetry_Int_Tables_number"):
        symmetry_number.append(int(line.strip().split()[-1]))


if len(symmetry_number) == len(structure_number):
    df_idsym = pd.DataFrame({'ID':structure_number, 'Symm': symmetry_number})
    print(df_idsym)
else:
    raise ValueError(f"Length mismatch: {len(structure_number)} IDs vs {len(symmetry_number)} symmetries.")

#Filtering certain symm
df_id63 = df_idsym[df_idsym['Symm']==63]
print(df_id63)

#Opening Individuals

with open("/home/brian/USPEX_work/EX02/results3/Individuals", "r", encoding="utf-8") as file:
    content = file.readlines()
    content.pop(1)
    content_str =  ''.join(content)


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

# Read the data into a DataFrame, handling whitespace more flexibly
df = pd.read_csv(data_io, delim_whitespace=True, skiprows=0)

# Display the DataFrame
print(df)


def convert_to_list(x):
    # Remove the brackets and split by commas
    elements = x.strip('[]').split(',')

    # Convert to integer while ignoring empty strings
    int_list = [int(e) for e in elements if e]

    # Create a NumPy array from the list
    np_array = np.array(int_list)

    return np_array


df['Composition'] = df['Composition'].apply(lambda x: convert_to_list(x))
df['KPOINTS'] = df['KPOINTS'].apply(lambda x: convert_to_list(x))
df['N_atoms'] = df['Composition'].apply(lambda x: np.sum(x))
df['Energy/N_atoms'] = df['Enthalpy'] / df['N_atoms']
print(df)

#Data filtering
df_id63 = df[df['SYMM']==63]
print(df_id63)
print(tabulate(df_id63, headers='keys'))