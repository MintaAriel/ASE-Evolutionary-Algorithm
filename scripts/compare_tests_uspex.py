import os.path
from ea.io.uspex_io import parse_tests
from ea.analysis.success_rate import SuccessResults, AnalyzeTest
from tabulate import tabulate
from pathlib import Path

d = parse_tests('/home/vito/PythonProjects/ASEProject/EA/test/collected_theophylline',
                out_dir='/home/vito/PythonProjects/ASEProject/EA/test/collected_thp_all')
# df = d.individuals()
# print(df)

script_dir = Path(__file__).parent
project_root = script_dir.parent

spinel_results_dir = os.path.join(project_root,'results', 'THP', 'best_individuals')


# spinel_folders = {'ase':['ase_2.db', 'uspex'],
#                   'uspex_python_my':['uspex_python.db', 'uspex'],
#                   'ase_3':['runs_ase_3.db', 'ase'],}

thp_folder = {'uspex':['theophylline_full.db','uspex'],
              'pyxtal':['theophylline_seeds_1.db','uspex'],
              'pyxtal/reaxff':['theophylline_seeds_2.db','uspex'],
              'pyxtal/reaxff small':['theophylline_seeds_3.db','uspex']}


def read_results(dic_experiments, task='read'):
    '''
    :param dic_experiments: Dictionary with db.name and program used
    Example:
    spinel_folders = {'ase':['ase_2.db', 'uspex'],
                    'uspex_python_my':['uspex_python.db', 'uspex'],
                    'ase_3':['runs_ase_3.db', 'ase'],}
    :param task: read: Use this to see the global minimum of each experiment
                compare: Once you know with global minimum you are using to compare the parallel tests
                use it as an input for AnalyzeTests.success_rate(global_min=your minimum)
    :return: matplotlib graphic with success rate
    '''
    spinel_results = {}
    for k,v in dic_experiments.items():
        db_path = os.path.join(spinel_results_dir, v[0])
        spinel_results[k] = AnalyzeTest(db=db_path, program=v[1])
        print(f'3 Lowest values in {k}',spinel_results[k].low)
        if task == 'compare':

            spinel_results[k].success_rate(global_min=-457.567)
            print(spinel_results[k].success_df)

    if task == 'compare':
        benchmark = SuccessResults(spinel_results)
        benchmark.plot_success()
        # benchmark.plot_success_hist()
        benchmark.plot_best()
read_results(dic_experiments=thp_folder, task='compare')

# 'uspex_mat': uspex_mat.success_df,
# 'ase':ase.success_df,
# success = {'ase_pyxtal_cor': ase_pyxtal_cor.success_df,
#            'uspexpy1': uspex_python.success_df,
#            'uspexpy2': uspex_python_my.success_df,
#            'uspexpy3':uspexpy2.success_df,
#            'ase3': ase_3.success_df}
#
# print(success)
#
# print([(k,str(v['success'].sum())) for k,v in success.items()])
# #uspex_mat.plot_success_hist(success)
# uspex_mat.plot_success(success)

#print('this is the success \n',ase_pyxtal.success_df['success'].value_counts())



#uspex_mat.plot_best(data)
