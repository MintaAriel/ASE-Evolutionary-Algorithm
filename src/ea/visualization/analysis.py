from ase import visualize
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import json

class Read_results():
    def __init__(self, directory):
        self.dir = directory
        self.conn = sqlite3.connect(self.dir)
        self.df = self.get_db()

    def get_db(self):
        table_query = ("SELECT id, "
                       "json_extract(key_value_pairs, '$.generation') AS generation,"
                       "json_extract(key_value_pairs, '$.raw_score') AS energy,"
                       "json_extract(key_value_pairs, '$.origin') AS operator "
                       "FROM systems WHERE json_extract(key_value_pairs, '$.relaxed') = 1;")
        df = pd.read_sql_query(table_query, self.conn)
        return df

    # Opening Individuals
    def uspex_to_df(self, indiv_dir):
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

    def best_struc_graph(self):
        max_energy = df[df['operator'].notna()].groupby('generation')['energy'].idxmax()
        best_df = df.loc[max_energy]
        print(best_df)
        plt.scatter(best_df['generation'], -best_df['energy'])
        best_energies = df[df['operator'].notna()].groupby('generation')['energy'].max()
        best_so_far = [max(best_energies.tolist()[:i + 1]) for i in range(len(best_energies.tolist()))]
        print(best_so_far)
        plt.step(best_df['generation'], -np.array(best_so_far), where='post')
        plt.xlim(0,)
        plt.xlabel('Generation number')
        plt.ylabel('Energy')
        plt.show()

db = Read_results('/home/brian/PycharmProjects/ASEProject/GA/Sym40_1/GA1_sym_40.db')
df = db.get_db()
print(df)
uspex1 = db.uspex_to_df(r'/home/brian/PycharmProjects/ASEProject/GA/Visualization/BESTIndividuals' )
print(uspex1)
db.best_struc_graph()




