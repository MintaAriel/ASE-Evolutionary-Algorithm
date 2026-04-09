from ea.io.uspex_io import get_structure_from_id
from io import StringIO
from ase.io import read
from ea.utils.config import load_config
from ea.analysis.benchmark_mattersim import MatterSimTester
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

cfg = load_config()
pths = cfg['paths']
ms = cfg['mattersim']

project_root = Path(__file__).resolve().parents[2]
results_dir = Path(project_root / pths['results_dir']).resolve()

# Output directory for this experiment
output_dir = results_dir / 'THP' / 'relax_cpu'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be stored in: {output_dir}")

tests = [#'collected_theophylline_0',
         #'collected_theophylline_1',
         #'collected_theophylline_2',
         'theophylline_uspex']

test_to_try = [26]
structures_id = list(range(1, 11))
# CPU core affinity per structure: structure_id -> list of core IDs
core_affinity = {
    k: [2*(k-1), 2*(k-1) + 1]
    for k in structures_id
}

# Use CPU container with deepmd_template and deepmd_d3 method
tester = MatterSimTester(
    model_path=ms['model_path'],
    device='cpu',
    container_root=ms['container_root_cpu'],
    input_template='deepmd_template.py',
    n_threads=ms.get('n_threads', 4),
    sif_name='ml-relax-cpu.sif',
    method='deepmd_d3',
    model_key='base_deepmd',
    #params={"fire_steps": 100, "lbfgs_stages": [0.03], "lbfgs_steps": 100},
)

def relax_one(tester, poscar_str, struct_dir, cpu_affinity):
    """Relax a single structure (runs in a worker process)."""
    crystal = read(StringIO(poscar_str), format='vasp')
    result = tester.relax(crystal, outdir=struct_dir, cpu_affinity=cpu_affinity)
    traj_file = struct_dir / 'opt.traj'
    if traj_file.exists():
        traj_file.unlink()
    return result['final_energy']


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
                                       id_structures=structures_id)

        test_output_dir = output_dir / name / f'test_{number}'

        # Submit all structures in parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=len(poscar)) as pool:
            for k, v in poscar.items():
                struct_dir = test_output_dir / f'structure_{k}'
                affinity = core_affinity.get(k)
                future = pool.submit(relax_one, tester, v, struct_dir, affinity)
                futures[future] = k
                time.sleep(1)  # ← delay between submissions


            energies = {}
            for future in as_completed(futures):
                k = futures[future]
                try:
                    energy = future.result()
                    print(f"  Structure {k}: energy = {energy:.4f} eV")
                except Exception as e:
                    print(f"  Structure {k}: Relaxation failed: {e}")
                    energy = 0
                energies[k] = energy

        # Collect in original order
        all_energies[name][number] = [energies[k] for k in sorted(energies)]

# Save summary
summary_path = output_dir / 'energies_summary.json'
with open(summary_path, 'w') as f:
    json.dump(all_energies, f, indent=2)
print(f"\nEnergy summary saved to: {summary_path}")
