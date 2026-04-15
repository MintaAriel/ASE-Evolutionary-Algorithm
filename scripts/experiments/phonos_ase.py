import numpy as np
from deepmd_template import DeepMDRelaxation, RelaxConfig
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pathlib import Path
import time
from ase.io import read
from ase.vibrations import Vibrations

config = RelaxConfig()

deep = DeepMDRelaxation(config)
deep.model_key = 'deepmd_d3'
calc = deep.build_calculator(models_dir=Path('/home/vito/PythonProjects/ASEProject/container_gpu_2/models'),
                             device='cuda',
                             threads=2)

cif_path = '/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/128707/deepmd_d3_in/final.cif'

atom = read(cif_path)
atom.calc =calc

vib = Vibrations(atom)
vib.run()
vib.summary()

vib.write_mode(211)