from ea.analysis.operators_ase import MolCrystalOperatorTest

test = MolCrystalOperatorTest(db_path='/home/vito/uspex_matlab/theo_pyxtal/test_2/theophilline.db',
                              out_dir='/home/vito/PythonProjects/ASEProject/EA/test/collected_theophylline')

# Quick timing comparison (no GULP)
test.run_comparison(parent_pairs=[(2, 3), (4, 5)], n_trials=50)
print(test.summary())
test.plot_time_distribution(save=True)

# Or test a single operator with relaxation
df = test.test_operator_with_relaxation('heredity', mom_id=1, dad_id=2, n_trials=20)