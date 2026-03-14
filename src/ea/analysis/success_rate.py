import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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






