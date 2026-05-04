from pygulp.molecule import fix_mol_gradient
from ase.ga.data import DataConnection
from ase.io.trajectory import Trajectory
from ..structures import create_seeds
import os
import shutil

def rearange_mut(work_dir, connections_dir ,atom, lenghts ):
    optimizer = fix_mol_gradient.GradientDescentGULP(atom, work_dir=work_dir, connections=connections_dir,
                                                     library='reaxff')
    # optimizer.asu.cell[0][0] = lenghts[0]
    # optimizer.asu.cell[1][1] = lenghts[1]
    # optimizer.asu.cell[2][2] = lenghts[2]
    optimizer.asu.set_cell([8.211, 5.4624, 65.38, 90, 87.78, 90])
    try:
        new = optimizer.run(steps=150, traj=True)
        best = optimizer.best_structure
    except Exception as e:
        print(e)
        print(f'structure not suitable')
        # best = atom

    return optimizer.best_structure


class mutate_reaxff():
    def __init__(self, work_dir, connection_dir):
        self.work_dir= work_dir
        self.connection_dir = connection_dir
        self.db_dir = os.path.join(work_dir, 'theophilline.db')
        self.da = DataConnection(self.db_dir)
        self.traj_name = 'all_relaxed.trajectory'



    def mutate(self, n_structures, poscar_name='POSCARS_1'):
        traj_dir = os.path.join(self.work_dir, self.traj_name)
        trajectory = Trajectory(traj_dir, 'w')
        for i in range(2,int(2+n_structures)):
            atom = self.da.get_atoms(i)
            optimizer = fix_mol_gradient.GradientDescentGULP(atom, work_dir=self.work_dir, connections=self.connection_dir, library='reaxff')

            try:
                new  = optimizer.run(steps =50,  traj=True)
                best = optimizer.best_structure
            except Exception as e:
                print(e)
                print(f'structure {i} not suitable')
                best = atom

            trajectory.write(best)
        trajectory.close()

        seeds = create_seeds.write_POSCAR_traj(traj=traj_dir,
                                     out_dir=self.work_dir)
        seeds.poscar_name = poscar_name
        seeds.create(n_structures)

        os.remove(traj_dir)
        shutil.rmtree(os.path.join(self.work_dir, 'CalcFold'))

class mutate_reaxff_small():
    def __init__(self, work_dir, connection_dir):
        self.work_dir= work_dir
        self.connection_dir = connection_dir
        self.db_dir = os.path.join(work_dir, 'theophilline.db')
        self.da = DataConnection(self.db_dir)
        self.traj_name = 'all_relaxed.trajectory'



    def mutate(self, n_structures, poscar_name='POSCARS_1', factor=0.85):
        traj_dir = os.path.join(self.work_dir, self.traj_name)
        trajectory = Trajectory(traj_dir, 'w')
        for i in range(2,int(2+n_structures)):
            atom = self.da.get_atoms(i)
            optimizer = fix_mol_gradient.GradientDescentGULP(atom, work_dir=self.work_dir, connections=self.connection_dir, library='reaxff')

            try:
                optimizer.asu.cell *= factor
                new  = optimizer.run(steps =50,  traj=True)
                best = optimizer.best_structure
            except Exception as e:
                print(e)
                print(f'structure {i} not suitable')
                best = atom

            trajectory.write(best)
        trajectory.close()

        seeds = create_seeds.write_POSCAR_traj(traj=traj_dir,
                                     out_dir=self.work_dir)
        seeds.poscar_name = poscar_name
        seeds.create(n_structures)

        os.remove(traj_dir)
        shutil.rmtree(os.path.join(self.work_dir, 'CalcFold'))

class mutate_reaxff_small2():
    def __init__(self, work_dir, connection_dir):
        self.work_dir= work_dir
        self.connection_dir = connection_dir
        self.db_dir = os.path.join(work_dir, 'theophilline.db')
        self.da = DataConnection(self.db_dir)
        self.traj_name = 'all_relaxed.trajectory'

    def mutate(self, n_structures, poscar_name='POSCARS_1', keep_traj=False):
        traj_dir = os.path.join(self.work_dir, self.traj_name)
        trajectory = Trajectory(traj_dir, 'w')

        for i in range(2, int(2 + n_structures)):
            atom = self.da.get_atoms(i)

            try:
                optimizer = fix_mol_gradient.GradientDescentGULP(
                    atom,
                    work_dir=self.work_dir,
                    connections=self.connection_dir,
                    library='reaxff'
                )

                optimizer.asu.cell *= 0.85  # try modifying cell
                new = optimizer.run(steps=50, traj=True)
                best = optimizer.best_structure

            except Exception as e:
                print(e)
                print(f'structure {i} not suitable')

                # fallback: run without modifying cell
                try:
                    optimizer = fix_mol_gradient.GradientDescentGULP(
                        atom,
                        work_dir=self.work_dir,
                        connections=self.connection_dir,
                        library='reaxff'
                    )
                    new = optimizer.run(steps=50, traj=True)
                    best = optimizer.best_structure

                except Exception as e2:
                    print(e2)
                    best = atom  # final fallback

            trajectory.write(best)

        trajectory.close()

        seeds = create_seeds.write_POSCAR_traj(traj=traj_dir,
                                     out_dir=self.work_dir)
        seeds.poscar_name = poscar_name
        seeds.create(n_structures)

        if keep_traj == True:
            pass
        else:
            os.remove(traj_dir)
        shutil.rmtree(os.path.join(self.work_dir, 'CalcFold'))

