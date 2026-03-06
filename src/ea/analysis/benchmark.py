import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os
import matplotlib
import numpy as np
from tabulate import tabulate
import seaborn as sns
from ase.parallel import parallel_function
from ase.ga.data import DataConnection
from ase.visualize import view



# da = DataConnection('/home/vito/Documents/UMA/Mg4Al8O16_40.db')
# a = da.get_atoms(45)
# view(a)

class compare_uspex():

    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.df = pd.DataFrame()
        self.parents = []

    def get_df(self, indiv_dir):
        with open(indiv_dir, "r", encoding="utf-8") as file:
            content = file.readlines()
            #content.pop(1)
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
        df = pd.read_csv(data_io, sep='\s+', skiprows=0)
        return df

    def get_tests(self, operator):
        if operator == 'heredity':
            indices = [i for i in range(1,11)]
            for i in indices:
                ind_dir = os.path.join(self.test_dir, f'test_{i}', 'results1/Individuals')
                df = self.get_df(ind_dir)
                energies = df[(df['generation'] == 2) & (df['origin'] == 'Heredity')]['energy']
                parents = df[df['generation'] == 1]['energy']
                self.df[f'{i}'] = energies.tolist()
                self.parents.append(parents.tolist())

        elif operator == 'softmut':
            indices = [i for i in range(2,22,2)]
            experiment  = {}
            for i in indices:
                ind_dir = os.path.join(self.test_dir, f'test_{i}', 'results1/Individuals')
                df = self.get_df(ind_dir)
                energies = df[(df['generation'] == 2) & (df['origin'] == 'SoftMutation')]['energy']
                parent = df[df['generation'] == 1]['energy']
                experiment[f'{i}'] = energies.tolist()
                self.parents.append(parent.tolist())

            self.df = pd.DataFrame.from_dict(experiment, orient='index').T
            self.df.columns = [f'{i}' for i in range(1,11)]
        else:
            print('not valid operator')





class compare_ase():
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.df = pd.DataFrame()

    def csv2df(self, dir):
        df = pd.read_csv(dir)
        return df.iloc[:, 1]

    def get_tests(self, operator):
        if operator == 'heredity':
            for i in range(1,11):
                result_dir = os.path.join(self.test_dir, f'test_{i-1}','heredity/progress_log.csv')
                energies = self.csv2df(result_dir)
                print(energies.tolist())
                self.df[f'{i}'] = energies.tolist()
        elif operator == 'softmut':
            experiment = {}
            for i in range(1,11):
                result_dir = os.path.join(self.test_dir, f'test_{i-1}','softmut_modes/progress_log.csv')
                energies = self.csv2df(result_dir)
                print(energies.tolist())
                experiment[f'{i}'] = energies.tolist()

            self.df = pd.DataFrame.from_dict(experiment, orient='index').T



def get_modal_class(counts, bins):
    modal_bin_index = counts.argmax()  # or np.argmax(counts)

    # 3. GET the corresponding bin EDGES (the modal class interval)
    modal_class_lower = bins[modal_bin_index]
    modal_class_upper = bins[modal_bin_index + 1]
    result = [counts.max(), modal_class_lower, modal_class_upper]

    return result

def compare_ase_uspex(test_n, df1, df2, operator, parents):
    plt.rcParams["font.family"] = "serif"
    matplotlib.rcParams.update({
        'font.size': 16,
        'figure.figsize': (8, 6)
    })

    ase = df1[f'{test_n}'].tolist()
    uspex = df2[f'{test_n}'].tolist()
    all_data = ase+uspex
    global_min = min(all_data)
    global_max = max(all_data)

    num_bins = 50
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)

    counts1, bins1, _ = plt.hist(ase, bins=bin_edges, alpha=0.5, label='ASE',
             edgecolor='black', color='darkblue')
    counts2, bins2, _ = plt.hist(uspex, bins=bin_edges, alpha=0.5, label='Uspex',
             edgecolor='black', color='green')

    ase_mean = df1[f'{test_n}'].mean()
    uspex_mean = df2[f'{test_n}'].mean()
    print(f'Pair{test_n}')
    print(f'The ase mean is: {ase_mean:.3f} \nThe Uspex mean is: {uspex_mean:.3f}')

    modal_ase = get_modal_class(counts1, bins1)
    modal_uspex = get_modal_class(counts2, bins2)

    print(f'The max count in ASE histogram is {modal_ase[0]}, bin: [{modal_ase[1]:.3f}, {modal_ase[2]:.3f}]')
    print(f'The max count in Uspex histogram is {modal_uspex[0]}, bin: [{modal_uspex[1]:.3f}, {modal_uspex[2]:.3f}]')


    if operator == 'heredity':
        plt.axvline(parents[test_n-1][0], color='r', label='parents')
        plt.axvline(parents[test_n-1][1], color='r')
    elif operator == 'softmut':
        plt.axvline(parents[test_n - 1][0], color='r', label='parents')

    plt.legend()
    plt.title(f'Pair_{test_n}')
    plt.ylabel('Count')
    plt.xlabel('Energy')
    #plt.savefig(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/results_test2/pair_{test_n}')
    plt.show()

def compare_ase_uspex_sum(test_n, df1, df2, operator, parents):
    plt.rcParams["font.family"] = "serif"
    matplotlib.rcParams.update({
        'font.size': 16,
        'figure.figsize': (8, 6)
    })

    ase = []
    uspex = []
    for i in range(1,11):
        ase += df1[f'{i}'].tolist()
        uspex += df2[f'{i}'].tolist()


    print(ase)
    print(len(ase))

    all_data = ase + uspex
    global_min = min(all_data)
    global_max = max(all_data)

    num_bins = 50
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)

    # counts1, bins1, _ = plt.hist(ase, bins=bin_edges, alpha=0.5, label='ASE',
    #                              edgecolor='black', color='darkblue')
    # counts2, bins2, _ = plt.hist(uspex, bins=bin_edges, alpha=0.5, label='Uspex',
    #                              edgecolor='black', color='green')

    sns.histplot(ase, kde=True, bins=bin_edges, color='darkblue', edgecolor='black', alpha=0.6, label='ASE')
    sns.histplot(uspex, kde=True, bins=bin_edges, color='red', edgecolor='black', alpha=0.6, label='USPEX')

    plt.ylabel('Count')
    plt.xlabel('Energy')
    plt.legend()

    if operator == 'heredity':
        plt.title('Heredity')

    elif operator == 'softmut':
        plt.title('Soft Mutation')

    plt.savefig(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/results_test2/{operator}_1000')
    plt.show()



def compare_heredity():
    exp_uspex = compare_uspex('/home/vito/uspex_python/TEST2')
    exp_uspex.get_tests(operator='heredity')
    uspex = exp_uspex.df

    exp_ase = compare_ase('/home/vito/PythonProjects/ASEProject/EA/COMPARISON/TEST2')
    exp_ase.get_tests(operator='heredity')
    ase = exp_ase.df

    print('THIS ARE THE PARENTS \n',exp_uspex.parents)
    print('This is ase \n', ase, '\n', 'This is USPEX \n', uspex)



    # for i in range(1, 11):
    #     compare_ase_uspex(i, ase, uspex, operator='heredity', parents=exp_uspex.parents)
        #plt.show()
    compare_ase_uspex_sum(1, ase, uspex, operator='heredity', parents=exp_uspex.parents)

def compare_softmut():
    exp_uspex = compare_uspex('/home/vito/uspex_python/TEST3')
    exp_uspex.get_tests('softmut')
    uspex = exp_uspex.df
    print(uspex.iloc[:10])
    print(uspex.info())

    exp_ase = compare_ase('/home/vito/PythonProjects/ASEProject/EA/COMPARISON/TEST3')
    exp_ase.get_tests('softmut')
    ase = exp_ase.df
    ase.replace(0, np.nan, inplace=True)

    for i in range(1, 11):
        compare_ase_uspex(i, ase.iloc[:50],uspex.iloc[:50],  operator='softmut', parents=exp_uspex.parents)
    #compare_ase_uspex_sum(1, ase,uspex,  operator='softmut', parents=exp_uspex.parents)


    print(tabulate(ase, headers='keys', tablefmt='psql'))

    print(ase.info())

#compare_heredity()
compare_softmut()