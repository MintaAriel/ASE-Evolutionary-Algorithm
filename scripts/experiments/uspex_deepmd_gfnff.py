from ea.utils.config import load_config
from ase.atoms import Atoms
from ase.io import read, write
from ase.visualize import view
import shutil
from pathlib import Path
# from pygulp.relaxation.relax import Gulp_relaxation_noadd
from deepmd_template import DeepMDRelaxation, RelaxConfig
import os
import time
from ase.calculators.gulp import GULP, GULPOptimizer
from ase.vibrations import Vibrations
import re


def get_folder_id():
    folder_name = os.path.basename(os.getcwd())
    match = re.search(r'(\d+)$', folder_name)  # integer at the end
    return int(match.group(1)) if match else None

def phonons_at_gamma(out_dir, atoms):
    vib_dir =  Path(out_dir) / 'vib'
    vib = Vibrations(atoms, name= vib_dir)
    vib.run()
    #vib.summary()
    energy = vib.get_zero_point_energy()
    print('ZPE:', energy )

    return energy




gulp_input = (f"opti gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
              f"gfnff_scale 0.80 1.343 0.727 1.0 1.41455 \n"
              f"maths mrrr"
              f"pressure 0 GPa"
              )
options = (
    "output movie cif gfnff.cif\n"
    "maxcycle 500"
)


print("CWD at start:", os.getcwd())
#
# relax = Gulp_relaxation_noadd(path=os.getcwd(),
#                               library=None,
#                               gulp_keywords=gulp_input,
#                               gulp_options=options)


cfg = load_config()

gulp_exe = cfg["executables"]["gulp_dir"]

calculator_gulp = GULP(keywords=gulp_input,
                    # goutput file parameters from USPEX
                    options=[options.strip()],
                    # maybe Optional Parameters
                    library=None)

calculator_gulp.directory = os.getcwd()

id = get_folder_id()
start_core = 0
threads = 2
if id is not None:
    calculator_gulp.command = f'taskset -c {int((id * threads) + start_core)} {gulp_exe} <gulp.gin> gulp.got'
    config = RelaxConfig(threads=2, cores=[id*threads + start_core, id*threads-1 +start_core])
else:
    calculator_gulp.command = f'{gulp_exe} <gulp.gin> gulp.got'

    config = RelaxConfig(threads=2)


def pick_input() -> str:
    """Using this comand in USPEX
    % abinitioCode
    99
    % ENDabinit

    When creating the calcl folder, there will be a poscar
    file named geom.in, this function reads the path of the
    file and transforms it into an ase ASE atoms object
    """
    if Path("geom.in").is_file():
        return read(Path("geom.in"), format='vasp')

    raise FileNotFoundError("Нет входного файла: geom.in")

crystal = pick_input()


ms = cfg['deepmd']
MODELS = ms['models_path']


print(crystal)

try:
        #Gfnff relaxation till 0.01 min Force
    # crystal.calc = calculator_gulp
    GULPOptimizer(crystal, calculator_gulp).run()
    gulp_energy = crystal.get_potential_energy()
    print('Energy at the end of gfnff', gulp_energy)
    print(crystal)

    #DeepMD relaxation till 0.001

    print('Starting DeePMD')
    deep = DeepMDRelaxation(config)
    deep.model_key = 'deepmd_d3'
    calc_deep = deep.build_calculator(models_dir=Path(MODELS),
                                 device='cpu')

    crystal.calc = calc_deep
    results_deeep = deep.run(crystal, Path(os.getcwd()))
    relaxed_atoms = crystal
    energy_deep = results_deeep['energy']
    print('Energy at the end of deepmd  ', energy_deep)
    print(crystal)

    #Phonons contribution
    ZPE = phonons_at_gamma(os.getcwd(), crystal)
    energy = energy_deep + ZPE

except Exception as e:
    print(f"Relaxation failed: {e}")
    print("Falling back to input structure with energy = 0")
    relaxed_atoms = crystal
    energy = 0

print("CWD before write:", Path(os.getcwd()))
print('saving as geom.out')

write('geom.out', relaxed_atoms, format='vasp', direct=True)

print('structure saved')

with open('energy.txt', 'w') as f:
    f.write(f"{energy}\n")