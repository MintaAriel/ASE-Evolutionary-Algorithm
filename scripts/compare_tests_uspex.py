import os.path
from ea.io.uspex_io import parse_tests
from ea.analysis.success_rate import SuccessResults, AnalyzeTest
from tabulate import tabulate
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent

def create_db(folder, type ):

    # folder = 'theophylline_seeds_0'

    results_dir = os.path.join(project_root,'results','THP','tests', folder)

    if type == 'all':
        out_dir = os.path.join(project_root,'results','THP','individuals')

    elif type == 'best':
        out_dir = os.path.join(project_root,'results','THP','best_individuals')

    collect = parse_tests(tests_dir=results_dir,
                    out_dir=out_dir)
    collect.individuals(type=type, db=True)

tests = ['theophylline_uspex','collected_theophylline_0','collected_theophylline_1',
         'collected_theophylline_2','collected_theophylline_3']

# for test in tests:
#     create_db(test, type='best')
#     create_db(test, type='all')




spinel_results_dir = os.path.join(project_root,'results', 'THP', 'best_individuals')



thp_folder = {'uspex':['best_theophylline_uspex.db','uspex'],
              'pyxtal':['best_collected_theophylline_1.db','uspex'],
              'pyxtal/reaxff':['best_collected_theophylline_2.db','uspex'],
              'pyxtal/reaxff small':['best_collected_theophylline_3.db','uspex']}


def read_results(dic_experiments, task='read'):
    '''
    :param dic_experiments: Dictionary with db.name (collected BESTIndividuals) and program used
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
        # benchmark.plot_best()

# read_results(dic_experiments=thp_folder, task='compare')

