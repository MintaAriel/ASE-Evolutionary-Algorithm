from ea.core.mutation import mutate_reaxff, mutate_reaxff_small2

operator = mutate_reaxff_small2(work_dir='/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1',
                         connection_dir='/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections')

operator.mutate(n_structures=70, poscar_name='1_POSCARS', keep_traj=True)

