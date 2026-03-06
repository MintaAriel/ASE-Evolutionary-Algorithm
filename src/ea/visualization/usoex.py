import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import seaborn as sns


class best_100():
    def __init__(self, db, program):
        self.db = db
        self.con = sqlite3.connect(self.db)
        self.program = program.lower()
        self.df = self.sql_to_df()
        self.success_df = None

    def sql_to_df(self):
        df = pd.read_sql('SELECT * FROM results',self.con)
        if self.program == 'ase':
            df['energy'] = df['energy']
            df['generation'] = df['generation']+1
        else:
            None

        return df


    def mean_energy(self):
        self.df['cummin_energy'] = self.df.groupby('run')['energy'].transform('cummin')
        if self.program == 'uspex':
            self.df['success'] = self.df['cummin_energy'].apply(lambda x: 1 if np.isclose(x, -655.062 , atol=1e-3)  else 0)
        elif self.program == 'ase':
            self.df['success'] = self.df['cummin_energy'].apply(lambda x: 1 if x < -655.062 else 0)

        df_mean = self.df.groupby('generation')['cummin_energy'].mean()
        low = self.df['energy'].min()
        print(f'The lowest values in {low}')

        self.success_df = self.succes_rate()

        return df_mean

    def plot_best(self, dic_results):
        for k,v in dic_results.items():
            plt.step(v.index, v.cummin(), where='post', label=k)
        plt.xlim(0, )
        plt.xlabel('Generation number')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()

    def succes_rate(self):
        first_success_generations = self.df.groupby('run')['success'].transform('idxmax').drop_duplicates()
        first_success_data = self.df.loc[first_success_generations, ['run', 'generation', 'success']].sort_values(by='run')

        return first_success_data

    def plot_success(self, dic_results):
        for k,v in dic_results.items():
            df = v.groupby('generation')['success'].sum()
            df = df.to_frame()
            df['cumulative_sum'] = df['success'].cumsum()
            df.loc[59] = [np.nan, df['cumulative_sum'].iloc[-1]]
            plt.step(df.index, df['cumulative_sum'], where='post', label=k)
        plt.xlim(0, 60)
        plt.ylim(0,100)
        plt.xlabel('Generation number')
        plt.ylabel('Success rate')
        plt.legend()
        plt.show()

    def plot_success_hist(self, dic_results):
        for k, v in dic_results.items():
            v = v[v['success'] ==1]
            sns.histplot(v, x="generation", element='step')
        plt.xlim(0, 60)
        plt.ylim(0,)
        plt.xlabel('Generation number')
        plt.ylabel('Discovered global minimum')
        plt.legend()
        plt.show()



uspex_mat = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/uspex_mat_2.db', 'uspex')
ase = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/ase_2.db', 'ase')
uspex_python_my = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/uspex_python.db', 'uspex')
ase_pyxtal_cor = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/pyxtal_correct.db', 'ase')
uspex_python = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/uspex_python_pyxtal.db','uspex')
#ase_pyxtalRanGen = best_100('/home/brian/PycharmProjects/ASEProject/GA/Visualization/ase_pyxtal(RanGen).db', 'ase')
uspexpy2 = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/uspexpy2.db','uspex')
ase_3 = best_100('/home/vito/PythonProjects/ASEProject/EA/Visualization/runs_ase_3.db', 'ase')

data =  {'uspex_mat':uspex_mat.mean_energy(),
        'ase':-ase.mean_energy(),
         'ase_pyxtal_cor': -ase_pyxtal_cor.mean_energy(),
         'uspex_pyxtal':uspex_python.mean_energy(),
         'uspex_pyxtal_my':uspex_python_my.mean_energy(),
         'uspexpy2':uspexpy2.mean_energy(),
         'ase3': ase_3.mean_energy()}


def print_gen( df, gen):
    print(tabulate(df[(df.run == gen)], headers='keys', tablefmt='psql'))


print_gen(uspex_mat.df, 3)


# 'uspex_mat': uspex_mat.success_df,
# 'ase':ase.success_df,
success = {'ase_pyxtal_cor': ase_pyxtal_cor.success_df,
           'uspexpy1': uspex_python.success_df,
           'uspexpy2': uspex_python_my.success_df,
           'uspexpy3':uspexpy2.success_df,
           'ase3': ase_3.success_df}

print(success)

print([(k,str(v['success'].sum())) for k,v in success.items()])
#uspex_mat.plot_success_hist(success)
uspex_mat.plot_success(success)

#print('this is the success \n',ase_pyxtal.success_df['success'].value_counts())
df1 = ase.success_df.groupby('generation')['success'].sum()
df1 = df1.to_frame()
df1['cumulative_sum'] = df1['success'].cumsum()
print(df1)


#uspex_mat.plot_best(data)




