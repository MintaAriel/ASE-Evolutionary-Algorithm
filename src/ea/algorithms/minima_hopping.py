from ase.optimize.minimahopping import MinimaHopping, MHPlot
from ase.ga.data import DataConnection
from ase.calculators.gulp import GULP, GULPOptimizer #Conditions
import os
import re
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, LBFGS
os.environ["GULP_LIB"] = "/home/brian/USPEX_105/application/archive/test/templates/GULP_201/Specific"


from ase import Atoms

#We have to create a customize Gulp class singe we need to define a pressure and
#gulp return 'Total lattice enthalphy' instead of 'Total lattice energy' which GULP class cannot parse
class MyGULP(GULP):
    def read_results(self):
        super().read_results()  # try parent parsing first

        got_path = os.path.join(self.directory, self.label + '.got')
        with open(got_path) as fd:
            lines = fd.readlines()

        for line in lines:
            m = re.match(r'\s*Total lattice enthalpy\s*=\s*(\S+)\s*eV', line)
            if m:
                self.results['energy'] = float(m.group(1))
                #print(self.results['energy'])


class population_vis():
    def __init__(self, db_name):
        self.db_name = db_name
        self.da = DataConnection(self.db_name)

    def open_structures(self, num_prev):
        slab = self.da.get_atoms(num_prev)
        return slab



class global_opt_MH():

    def __init__(self, atom):
        self.atom = atom

    def optimization(self):
        calc = MyGULP(
            keywords='gradient nosymmetry conp pressure \n pressure 0.002 GPa',
            options=[open("/home/brian/PycharmProjects/PythonProject/global_opt/ginput_1").read().strip()],
            library=None
        )

        self.atom.calc = calc
        print("Before MH:", 'momenta' in self.atom.arrays)
        hop = MinimaHopping(self.atom, Ediff0=1, T0=1000)
        hop(totalsteps=30)
        #self.atom.info['key_value_pairs']['raw_score'] = -self.atom.get_potential_energy()
        #calc.read_results()
        print("After MH:", 'momenta' in hop._atoms.arrays)


def read_traj(path):
    traj = Trajectory(path)
    atoms = traj[-2]
    print(atoms.info)
    print(atoms.arrays)
    print(atoms.calc)
    #print(atoms.get_potential_energy())



class BFGS_opt():
    def __init__(self, atom):
        self.atom = atom
        self.calc = MyGULP(
            keywords='gradient nosymmetry conp pressure '
                     '\npressure 100 GPa',
            options=[open("/home/brian/PycharmProjects/PythonProject/global_opt/ginput_1").read().strip()],
            library=None
        )
        self.atom.calc = self.calc

    def optimize_BFGS(self):
        dyn = BFGS(self.atom, trajectory='test_bfgs.traj')
        dyn.run(fmax=0.05)

    def optimize_LBFGS(self):
        dyn = LBFGS(self.atom, trajectory='test_lbfgs.traj')
        dyn.run(fmax=0.05)

atom1 = population_vis('/home/brian/PycharmProjects/PythonProject/GA/prueba6.db').open_structures(3)


#bf = BFGS_opt(atom1)
#bf.optimize_BFGS()

#opt = global_opt_MH(atom1)
#opt.optimization()

plot = MHPlot('/home/brian/PycharmProjects/PythonProject/global_opt')
plot.get_figure()

#read_traj('/home/brian/PycharmProjects/PythonProject/global_opt/qn00000.traj')