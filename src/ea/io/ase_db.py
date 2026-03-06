import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os
from ase.ga.data import DataConnection
from ase.visualize import view
class Read_results():
    def __init__(self, directory):
        self.dir = directory
        self.conn = sqlite3.connect(self.dir)
        self.df = self.get_db()

    def get_db(self):
        table_query = ("SELECT * "
                        "FROM results " 
                        "WHERE energy > 654.9; ")
        df = pd.read_sql_query(table_query, self.conn)
        return df

class population_vis():
    def __init__(self, db_name, struc_ids):
        self.db_name = db_name
        self.struc_ids = struc_ids
        self.da = DataConnection(self.db_name)

    def open_structures(self):
        structures = [self.da.get_atoms(i) for i in self.struc_ids]
        return structures

ase_pyxtal = Read_results('/home/brian/PycharmProjects/ASEProject/EA/Visualization/ase_pyxtal_cor.db')
df = ase_pyxtal.get_db()
df = df[df['run'] < 6]
df_best_run = df.drop_duplicates(subset=['run', 'energy'])
best = list(zip(df_best_run['run'], df_best_run['id'], df_best_run['energy']))
print(best)

experiments_fol = '/home/brian/PycharmProjects/ASEProject'
experiment = 'runs_pyxtal'
db_name = 'Mg4Al8O16_40.db'

for run,id,energy in best:
    dir = os.path.join(experiments_fol, experiment, f"run_{run:03d}" , db_name)
    print(dir)

id_counts = df_best_run.groupby('run')['id'].apply(list)
data = {}

for index, row in id_counts.to_frame().iterrows():
    print(index)
    dir = os.path.join(experiments_fol, experiment, f"run_{index:03d}", db_name)
    best_st = population_vis(dir, row[0]).open_structures()
    data[f"run_{index:03d}"] = best_st
    print(dir)
    print(row[0])
    print(best_st)

print(id_counts)
print(df_best_run)

print(data['run_001'][2].info)
