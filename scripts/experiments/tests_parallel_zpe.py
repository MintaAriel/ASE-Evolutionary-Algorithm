# ---------------------------------------------------------------- demo main


from ase.io import read
from ea.parallel.zpe import make_deepmd_evaluator, ParallelVibrations
from ea.utils.config import load_config
from pathlib import Path
from ase.io.trajectory import Trajectory
import time


cfg = load_config()

MODELS = {
    'base_deepmd': 'dpa3_12.03.2026.pth',
    'deepmd_d3': 'dpa3-d3_torch.pth',
    'deepmd_d4': 'dpa3-d4.pth',
    'deepmd_d3_abs': 'dpa3-d3_abs_torch.pth',
    'deepmd_d3_mbj': 'dpa3-d3-cpu_mbj.pth',
    'deepmd_d3_mbj_abs': 'dpa3-d3-cpu_mbj_abs.pth',
}

def resolve_model(models_dir: Path, key: str) -> tuple[Path, str]:
    path = Path(models_dir) / Path(MODELS[key])
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path, key

path,_ = resolve_model(cfg['deepmd']['models_path'], 'deepmd_d3')

str18 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_18_POSCARS')
str19 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_19_POSCARS')
batch = Trajectory('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1 (Copy)/output.traj')[:2]

from deepmd.calculator import DP  # noqa: F401  (imported lazily)
calc = DP(model=path)
evaluator = make_deepmd_evaluator(calc)

start = time.perf_counter()

pv = ParallelVibrations(batch, batch_evaluator=evaluator,
                        delta=0.01, nfree=2)
pv.run()
pv.summary()
print('ZPE per structure (eV):', pv.get_zero_point_energies())

end = time.perf_counter()
print(f"Elapsed time: {end - start} seconds")