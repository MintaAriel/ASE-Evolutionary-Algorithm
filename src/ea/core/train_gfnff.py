from ase.io import read, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from matplotlib.figure import Figure
from pathlib import Path
import numpy as np


def relax_traj(relax, traj_path, out_path, index=':'):
    """Relax every structure in traj_path with GULP and write the optimized ones to out_path.

    relax is a pygulp relaxation object exposing use_gulp(atoms), e.g. Gulp_relaxation_noadd.
    Returns a list of (structure index, energy in eV) for the successful relaxations.
    """
    traj_path = Path(traj_path)
    structures = read(traj_path, index=index)
    energies = []

    with Trajectory(out_path, 'w') as out_traj:
        for i, atoms in enumerate(structures):
            print(f'[{i + 1}/{len(structures)}] relaxing {traj_path.name}')
            new_crys = relax.use_gulp(atoms)

            # use_gulp returns the input structure when GULP fails, so check the results
            results = new_crys.calc.results if new_crys.calc is not None else {}
            if 'energy' not in results:
                print(f'  structure {i} failed, skipping')
                continue

            energy = results['energy']
            n_molecules = len(np.unique(new_crys.get_tags()))
            new_crys.info['source_index'] = i
            new_crys.info['energy_per_molecule'] = energy / n_molecules

            # Store the energy without keeping the GULP calculator attached
            new_crys.calc = SinglePointCalculator(new_crys, energy=energy)
            out_traj.write(new_crys)
            energies.append((i, energy))
            print(f'  E = {energy:.4f} eV')

    return energies


def save_energy_distribution(energies, out_prefix, bins=20):
    """Save the energies to out_prefix.csv and a histogram to out_prefix.png."""
    out_prefix = Path(out_prefix)
    data = np.array(energies)

    np.savetxt(out_prefix.with_suffix('.csv'), data, delimiter=',',
               header='source_index,energy_eV', fmt=['%d', '%.6f'], comments='')

    # Figure without pyplot works headless and doesn't change the caller's backend
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(data[:, 1], bins=bins, edgecolor='black')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Count')
    ax.set_title(f'{out_prefix.name} (n = {len(data)})')
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.png'), dpi=200)
