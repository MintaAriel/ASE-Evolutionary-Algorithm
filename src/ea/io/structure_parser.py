import pandas as pd
import os
from io import StringIO
import numpy as np



# Opening simmetryzed_structures.cif
def ciff_to_df(cif_dir):
    with open(cif_dir, "r") as g:
        dif_str = g.read().splitlines()

    indices_start = []

    for index, line, in enumerate(dif_str):
        if line.strip().startswith("data_findsym-STRUC-"):
            indices_start.append(index)

    indices_start.append(len(dif_str))

    cif_files = {}
    cif = []
    # Cif will be stored as a list
    for i in range(len(indices_start) - 1):
        cif.append(dif_str[indices_start[i]:indices_start[i + 1]])

    cif_files['CIF'] = cif

    df = pd.DataFrame(cif_files)
    return df


# Opening Individuals
def indiv_to_df(indiv_dir):
    with open(indiv_dir, "r", encoding="utf-8") as file:
        content = file.readlines()
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

    # Read the data into a DataFrame, handling whitespace more flexibly
    df = pd.read_csv(data_io, delim_whitespace=True, skiprows=0)

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
    return (df)


cif_dir = r"/home/brian/USPEX_work/EX02/results3/symmetrized_structures.cif"
indiv_dir = r"/home/brian/USPEX_work/EX02/results3/Individuals"

ciff_df = ciff_to_df(cif_dir)
ind_df = indiv_to_df(indiv_dir)
structures = pd.concat([ciff_df, ind_df], axis=1)
structures.to_csv('structures.csv', index=False)
structures

df_63 = structures[structures['SYMM'] == 63]


def export_cif(df, crystal, folder_name):
    # create a folder
    path = rf'/home/brian/USPEX_work/CIF/{folder_name}'
    try:
        os.mkdir(path)
        print("Folder %s created!" % path)
    except FileExistsError:
        print("Folder %s already exists" % path)

    # create each .cif file
    for index, v in df_63.iterrows():
        cif_name = f"{crystal}_{v['ID']}"
        plain_text = '\n'.join(v['CIF'][1:])
        with open(rf"{path}/{cif_name}.cif", "w") as f:
            f.write(plain_text)


export_cif(df_63, 'Spinel', 63)