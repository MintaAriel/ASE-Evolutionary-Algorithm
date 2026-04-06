from ea.io.uspex_io import get_structure_from_id
from io import StringIO
from ase.io import read
from ea.utils.config import load_config
from ea.analysis.benchmark_mattersim import MatterSimTester
from pathlib import Path
import json

cfg = load_config()
pths = cfg['paths']
ms = cfg['mattersim']

project_root = Path(__file__).resolve().parents[2]
results_dir = Path(project_root / pths['results_dir']).resolve()

# Output directory for this experiment
output_dir = results_dir / 'THP' / 'relax_first_generation'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be stored in: {output_dir}")

tests = [#'collected_theophylline_0',
         #'collected_theophylline_1',
         #'collected_theophylline_2',
         'theophylline_uspex']

test_to_try = [26, 7, 37, 19, 40]

# Use GPU container with deepmd_template and deepmd_d3 method
tester = MatterSimTester(
    model_path=ms['model_path'],
    device='cuda',
    container_root=ms['container_root_gpu'],
    input_template='deepmd_template.py',
    n_threads=ms.get('n_threads', 4),
    sif_name='ml-relax-gpu.sif',
    method='deepmd_d3',
    model_key='base_deepmd',
    params={"fire_steps": 100, "lbfgs_stages": [0.03], "lbfgs_steps": 100},
)

all_energies = {}
for name in tests:
    all_energies[name] = {}
    for number in test_to_try:
        poscar_dir = (results_dir / 'THP' / 'tests' / name /
                      'gatheredPOSCARS_unrelaxed' / f'gatheredPOSCARS_unrelaxed_test_{number}'
                      )
        print(f"\n--- {name} / test_{number} ---")
        print(f"POSCAR dir: {poscar_dir}")

        if not poscar_dir.exists():
            print(f"  Skipping, path does not exist: {poscar_dir}")
            continue

        poscar = get_structure_from_id(poscar_dir=poscar_dir,
                                       id_structures=[i for i in range(1, 41)])

        energies = []
        # Per-test output directory
        test_output_dir = output_dir / name / f'test_{number}'

        for k, v in poscar.items():
            crystal = read(StringIO(v), format='vasp')
            struct_dir = test_output_dir / f'structure_{k}'

            try:
                result = tester.relax(crystal, outdir=struct_dir)
                energy = result['final_energy']
                print(f"  Structure {k}: energy = {energy:.4f} eV")
            except Exception as e:
                print(f"  Structure {k}: Relaxation failed: {e}")
                energy = 0

            energies.append(energy)

            traj_file = struct_dir / 'opt.traj'
            if traj_file.exists():
                traj_file.unlink()

        all_energies[name][number] = energies

# Save summary
summary_path = output_dir / 'energies_summary.json'
with open(summary_path, 'w') as f:
    json.dump(all_energies, f, indent=2)
print(f"\nEnergy summary saved to: {summary_path}")
