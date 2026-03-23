import os.path
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ea.io.uspex_io import get_structure_from_id
from typing import Dict
from io import StringIO

class AnalyzeTest():
    def __init__(self, db, program):
        '''
        :param db: .db of BESTIndividuals obtained by ea.io.uspex_io.parse_tests class
        :param program: 'uspex' or 'ase' depending on which program was used for the ea
        '''
        self.db = db
        self.con = sqlite3.connect(self.db)
        self.program = program.lower()
        self.df, self.low = self.sql_to_df()
        self.success_df = None
        self.df['cummin_energy'] = self.df.groupby('run')['energy'].transform('cummin')
        self.max_gen = self.df['generation'].max()
        self.test_max = self.df['run'].max()


    def sql_to_df(self):
        df = pd.read_sql('SELECT * FROM results', self.con)
        if self.program == 'ase':
            df['energy'] = df['energy']
            df['generation'] = df['generation'] + 1
        else:
            None

        low = df['energy'].nsmallest(3).tolist()

        return df, low

    def mean_energy(self):
        df_mean = self.df.groupby('generation')['cummin_energy'].mean()

        return df_mean

    def success_rate(self, global_min):

        PROGRAM_RULES = {
            "uspex": lambda x: np.isclose(x, float(global_min), atol=1e-3),
            "ase": lambda x: x < float(global_min),
        }

        self.df["success"] = self.df["cummin_energy"].apply(PROGRAM_RULES[self.program])

        first_success_generations = self.df.groupby('run')['success'].transform('idxmax').drop_duplicates()
        first_success_data = self.df.loc[first_success_generations, ['run', 'generation', 'success']].sort_values(
            by='generation')
        self.success_df =first_success_data[first_success_data['success'] == True]

        return first_success_data

class SuccessResults():

    def __init__(self, dict_results):
        """
     :param dict_results: It is a dictionary with AnalyzeTest object of each test
                 """
        self.dict_results = dict_results
        self.success = {}
        self.max_gens = {}
        self.test_max = []

        for k,v in self.dict_results.items():
            # print(v.success_df)
            self.success[k] = v.success_df
            self.max_gens[k] = int(v.max_gen)
            self.test_max.append(int(v.test_max))

        self.n_test = max(self.test_max)


    def plot_best(self):
        for k,v in self.success.items():
            plt.step(v.index, v.cummin(), where='post', label=k)
        plt.xlim(0, )
        plt.ylim(0,)
        plt.xlabel('Generation number')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()

    def plot_success(self):
        print(self.test_max)
        for k,v in self.success.items():

            df = v.groupby('generation')['success'].sum()
            df = df.to_frame()
            df['cumulative_sum'] = df['success'].cumsum()
            df.loc[self.max_gens[k]] = [np.nan, df['cumulative_sum'].iloc[-1]]
            df.loc[0, ['success', 'cumulative_sum']] = 0
            df = df.sort_index()
            plt.step(df.index, df['cumulative_sum']/self.n_test, where='post', label=k)

        plt.xlim(0, max(self.max_gens.values()))
        plt.ylim(0,)
        plt.xlabel('Generation number')
        plt.ylabel('Success rate')
        plt.title(f'Success Rate for {self.n_test} Tests')
        plt.legend()
        plt.show()

    def plot_success_hist(self):
        for k, v in self.success.items():
            v = v[v['success'] ==1]
            sns.histplot(v, x="generation", element='step')
        plt.xlim(0, 60)
        plt.ylim(0,)
        plt.xlabel('Generation number')
        plt.ylabel('Discovered global minimum')
        plt.legend()
        plt.show()

class AllIndivuals():
    def __init__(self, db_dir):
        self.db = db_dir
        self.con = sqlite3.connect(self.db)
        self.df = pd.read_sql('SELECT * FROM results', self.con)

        # Perform a clean of duplicate structures by energy, volume and symmetry
        self.df_clean = (
            self.df
            .dropna(subset=["energy"])
            .loc[lambda x: x["operator"] != "keptBest"]
            .drop_duplicates(subset=["energy", "volume", "symmetry"])
            .reset_index(drop=True)
        )

    def df_d(self):
        print(self.df_clean)
        plt.hist(self.df_clean['energy'],bins=100, edgecolor='black')
        plt.show()

    def energy_per_generation(self,generation=1, parameter='energy', plot=False):
        df_filtered = (self.df[self.df ['generation'] == generation]
                        .sort_values(by=parameter, ascending=True))

        nan_percentage = df_filtered['energy'].isna().mean() * 100
        print(f'Percentage of non-valid structures (Fitness = 10 000): {nan_percentage}%')
        if plot:
            plt.hist(df_filtered['symmetry'], bins=100, edgecolor='black')
            plt.show()



        return df_filtered


    def operator_percent(self, generation=5, parameter='Random', plot=False):

        percentages = (
                self.df.assign(is_random=self.df['operator'] == parameter)
                .groupby('generation')['is_random']
                .mean() * 100
        )

        print(percentages)
        if plot:
            plt.plot(percentages[1:generation])
            plt.title(f'{parameter} operator percent per generation')
            plt.xlabel('generation')
            plt.ylabel('%')
            plt.show()

        return percentages

    def mean_energy(self, top_generation=10, plot=False):
        mean_energy = (self.df
                       .groupby('generation')['energy']
                       .mean())

        if plot:
            plt.plot(mean_energy[:top_generation])
            plt.title(f'mean energy per generation')
            plt.xlabel('generation')
            plt.ylabel('Energy, eV')
            plt.show()

        return mean_energy

    def get_lowets(self, n=100):
        df_low = (
            self.df_clean
            .sort_values(by="energy", ascending=True)
            .drop_duplicates(subset=["energy", "symmetry"])
            .iloc[:n]
            .copy()
        )
        return df_low

    def get_lowest_poscar(self, n: int=100 , gatheredPOSCARS_dir='.', out_dir='.', name='POSCARS'):
        df = self.get_lowets(n=n)
        structures_id = (
            df.sort_values(["run", "id"])
            .groupby("run")["id"]
            .apply(list)
            .to_dict()
        )
        best = {}
        for k,v in structures_id.items():
            poscar_dir= os.path.join(gatheredPOSCARS_dir, f'gatheredPOSCARS_test_{k}')
            # print(type(f))
            collected_poscar = get_structure_from_id(poscar_dir= poscar_dir, id_structures= v)
            best[k] = collected_poscar

        runs = df['run'].tolist()
        ids = df['id'].tolist()
        energies = df['energy'].tolist()

        All_poscars = ''

        for i in range(len(ids)):
            poscar = best[runs[i]][ids[i]]
            comment = f' | test={runs[i]} | energy={energies[i]}'

            lines = poscar.splitlines()
            lines[0] = lines[0] + comment
            All_poscars += "\n".join(lines) + '\n'


        with open(os.path.join(out_dir,name), "w") as f:
            f.write(All_poscars)

        return df


class CompareRuns:
    def __init__(self, dict_results: Dict[any, AllIndivuals]):
        """
        :param dict_results: Dictionary with AllIndivuals objects
        """
        self.dict_results = dict_results

    def mean_energy(self, generation : int = 10):
        energy = {}
        for k,v in self.dict_results.items():
            energy[k] = v.mean_energy(top_generation=generation)
            plt.plot(energy[k], label=k)
        plt.title(f'mean energy per generation')
        plt.xlabel('generation')
        plt.ylabel('Energy, eV')
        plt.legend()
        plt.show()

        return energy

    def operator_percent(self, top_gen=10, parameters='Random'):
        runs = {}
        for k,v in self.dict_results.items():
            runs[k] = v.operator_percent(parameter=parameters,  generation=top_gen)
            plt.plot(runs[k], label=k)
        plt.legend()
        plt.show()

    def energy_per_generation(self, parameter='energy'):
        runs = {}

        # Collect all energies
        all_energies = []
        for k, v in self.dict_results.items():
            data = v.energy_per_generation()
            runs[k] = data
            all_energies.extend(data[parameter])

        # Define common bins based on global min/max
        bins = np.linspace(min(all_energies), max(all_energies), 100)

        # Plot histograms with same bins and transparency
        for k in runs:
            plt.hist(
                runs[k][parameter],
                bins=bins,
                alpha=0.25,  # transparency
                edgecolor='black',
                label=k
            )

        plt.legend()
        plt.xlabel(f"{parameter}")
        plt.ylabel("Count")
        plt.title(f"{parameter} Distribution per Generation")
        plt.show()













