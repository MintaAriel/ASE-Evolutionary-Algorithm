import os
import re
import sqlite3
from ase.ga.data import DataConnection
from ase.calculators.gulp import GULP, GULPOptimizer #Conditions
from ase.db import connect
import spglib

os.environ["ASE_GULP_COMMAND"]= os.path.expanduser("~/miniconda/envs/ase_env/bin/gulp < PREFIX.gin > PREFIX.got")

#We have to create a customize Gulp class singe we need to define a pressure and
#gulp return 'Total lattice enthalphy' instead of 'Total lattice energy' which GULP class cannot parse
class MyGULP(GULP):
    def read_results(self):
        super().read_results()  # try parent parsing first

        got_path = os.path.join(self.directory, self.label + '.got')
        print(got_path)
        with open(got_path) as fd:
            lines = fd.readlines()

        for line in lines:
            m = re.match(r'\s*Total lattice enthalpy\s*=\s*(\S+)\s*eV', line)
            if m:
                self.results['energy'] = float(m.group(1))
                #print(self.results['energy'])

class Gulp_relaxation:

    def __init__(self, atom):
        self.atom = atom

    def get_symetry(self, atom):
        sg = spglib.get_spacegroup((atom.get_cell(), atom.get_scaled_positions(),
                                    atom.get_atomic_numbers()),
                                   symprec=1e-5)
        if sg is None:
            sg_no = 1
        else:
            sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
        return sg_no

    def use_gulp(self):
        calc = MyGULP(keywords='opti conjugate nosymmetry conp \npressure 100 GPa',
                      # goutput file parameters from USPEX
                      options=[open("/home/brian/PycharmProjects/ASEProject/EA/Programms/GULP/ginput_3").read().strip()],
                      # maybe Optional Parameters
                      library=None,)

        calc2 = MyGULP(keywords='opti conjugate nosymmetry conp \npressure 100 GPa',
                      # goutput file parameters from USPEX
                      options=[open("/home/brian/PycharmProjects/ASEProject/EA/Programms/GULP/ginput_4").read().strip()],
                      # maybe Optional Parameters
                      library=None,)

        calc.directory = '/home/brian/PycharmProjects/ASEProject/EA/Programms/GULP/gulp_try'
        calc2.directory = '/home/brian/PycharmProjects/ASEProject/EA/Programms/GULP/gulp_try'
        print(calc.directory)


        GULPOptimizer(self.atom, calc).run()
        print(self.atom)
        self.atom.calc = None
        print(self.atom)
        GULPOptimizer(self.atom, calc2).run()

        self.atom.info['key_value_pairs']['spacegroup'] = self.get_symetry(self.atom)
        self.atom.info['key_value_pairs']['raw_score'] = -calc.get_potential_energy()


        return self.atom



#fist_gen = Gulp_relaxation('/home/brian/PycharmProjects/ASEProject/GA/sym70.db').relax_generation()
'''
        try:
            GULPOptimizer(atom, calc)
            atom.info['key_value_pairs']['spacegroup'] = self.get_symetry(atom)
            atom.info['key_value_pairs']['raw_score'] = -atom.info['energy']
            self.da.add_relaxed_step(atom)

        except Exception as e:
            print('Structure not suitable for optimization')
            self.da.kill_candidate(atom.info['confid'])
'''