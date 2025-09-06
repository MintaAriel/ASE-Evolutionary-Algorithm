import os
from ase.calculators.gulp import GULP, GULPOptimizer #Conditions
from ase.ga.data import DataConnection


os.environ["GULP_LIB"] = "/home/brian/USPEX_105/application/archive/test/templates/GULP_201/Specific"
da = DataConnection('/home/brian/PycharmProjects/PythonProject/GA/spinel1.db')

slab = da.get_atoms(3)
#Since GULP has no built-in functions to read custom input files, it is possible
#to put all this parameters in options
calc = GULP(keywords='opti conjugate nosymmetry conp', #goutput file parameters from USPEX
            options=[open("/home/brian/USPEX_work/EX02/Specific/ginput_1").read().strip()],  # maybe Optional Parameters
            library=None,)

print(calc.parameters)# <-- shows all parameters



# Attach the calculator to the atoms
slab.calc = calc
energy = slab.get_potential_energy()
print("Energy:", energy, "eV")
print(slab)
calc.atoms = slab
# If you want to write the GULP input to file (optional)
calc.write_input(slab)


