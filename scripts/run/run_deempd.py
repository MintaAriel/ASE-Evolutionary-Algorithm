from ea.backends.deepmd_client import DeepMDClient
from ea.utils.config import load_config
from ase.io import read

cfg = load_config()
atom = read('/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/128707/base_deepmd/final.cif')

atom.calc = DeepMDClient(config=cfg)

energy = atom.get_potential_energy()
print(energy)

atom.calc.close()
