import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

gfnff_uspex = '/home/vito/PythonProjects/ASEProject/EA/results/THP/individuals/all_theophylline_uspex.db'
deepmd_d3_uspex = '/home/vito/PythonProjects/ASEProject/EA/results/THP/relax_cpu/energies_summary_uspex.json'

gfnff_thp3 = '/home/vito/PythonProjects/ASEProject/EA/results/THP/individuals/all_collected_theophylline_3.db'
deepmd_d3_thp3 = '/home/vito/PythonProjects/ASEProject/EA/results/THP/relax_cpu/energies_summary_thp3.json'


def plot_gnff_deepmd(gfnff_path, deepmd_path):

    conn = sqlite3.connect(gfnff_path)

    df1 = pd.read_sql_query("""
        SELECT *
        FROM results
        WHERE run = 26 AND generation = 1
    """, conn)

    print(df1)

    conn.close()


    df = pd.read_json(deepmd_path)

    uspex_deempmd = df.iloc[:, 0].tolist()[0]


    uspex_gfnff = df1['energy'].tolist()


    y= np.array(uspex_gfnff)
    x= np.array(uspex_deempmd)

    print(np.sort(x))
    print(np.sort(y))

    # Mask: keep only pairs where BOTH x and y are valid numbers
    mask = (~np.isnan(x)) & (~np.isnan(y))

    # Apply mask to both arrays
    x_clean = x[mask]
    y_clean = y[mask]

    # Plot
    plt.scatter(x_clean, y_clean)



plot_gnff_deepmd(gfnff_uspex, deepmd_d3_uspex)
plot_gnff_deepmd(gfnff_thp3, deepmd_d3_thp3)

plt.ylabel("Energy, eV (gnfnf)")
plt.xlabel("Energy, eV (deepmd D3 trained)")
plt.title("Theophylline with 4 molecules")
plt.show()