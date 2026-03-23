import os
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.data import atomic_numbers
from ase import Atoms
from pygulp.relaxation.relax import Gulp_relaxation
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Operator_comparator():
    def __init__(self, blocks, molecular=False):
        self.block = blocks
        self.molecular = molecular
        self.atom_num = [atomic_numbers[name] for (name, number) in self.block]
        self.n_top = sum(value for element, value in self.block)
        self.slab = Atoms('', pbc=True)
        self.blmin = closest_distances_generator(self.atom_num, 0.1)
        self.cellbounds = CellBounds(
                    bounds={
                        'phi': [20, 160],
                        'chi': [20, 160],
                        'psi': [20, 160],
                        'a': [2, 40],
                        'b': [2, 40],
                        'c': [2, 40],
                    }
                )
        self.blmin_soft = closest_distances_generator(self.atom_num, 0.1)


    def heredity(self, mom, dad):
        self.slab.set_cell(mom.cell)
        # self.blmin = None
        pairing = CutAndSplicePairing(
        self.slab,
        self.n_top,
        self.blmin,
        p1=1.0,
        p2=0.0,
        minfrac=0.15,
        number_of_variable_cell_vectors=3,
        cellbounds=self.cellbounds,
        use_tags=self.molecular,)

        baby  = pairing.cross(mom, dad)
        return baby

    def softmutation_used_modes(self, mom, json_dir):
        mom.info['confid'] = 1
        sfmut = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0], used_modes_file=json_dir, use_tags=self.molecular)
        baby = sfmut.mutate(mom)
        return baby

    def softmut_no_modes(self,mom):
        mom.info['confid'] = 1
        sfmut = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0], used_modes_file=None, use_tags=self.molecular)
        baby = sfmut.mutate(mom)
        return baby


def save_progress(df, filename='progress_log.csv'):
    """Save/append DataFrame with timestamp to track progress"""
    # Check if file exists
    if os.path.exists(filename):
        # Append without writing header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(filename, mode='w', header=True, index=False)


def make_stats(operator,moms, dads, folder, db, n_experiments, index):

    da = DataConnection(db)
    experiment = Operator_comparator([('Mg', 4), ('Al', 8), ('O', 16)])
    df = pd.DataFrame()
    os.mkdir(f'{folder}/{operator}')

    a_norelax = da.get_atoms(moms)
    b_norelax = da.get_atoms(dads)

    relax_parent_dir = Gulp_relaxation(f'/{folder}/parents')
    a = relax_parent_dir.use_gulp_no_add(a_norelax)[0]
    b = relax_parent_dir.use_gulp_no_add(b_norelax)[0]



    energies = []
    os.mkdir(f'{folder}/{operator}/Pair_{index + 1}')

    for j in range(n_experiments):
        c = None
        count = 0
        while c is None:
            try:
                if operator == 'softmut_modes':
                    c = experiment.softmutation_used_modes(a, f'{folder}/{operator}/Pair_{index+1}/used_modes.json')
                elif operator == 'softmut_nomodes':
                    c = experiment.softmut_no_modes(a)
                elif operator == 'heredity':
                    c = experiment.heredity(a, b)
                else:
                    print('The operator is not correct')
            except Exception as error:
                print('Error when creating offspring', error)
                count += 1

            if count == 3:
                break


        if count == 3:
            break

        relaxation = Gulp_relaxation(f'/{folder}/{operator}/Pair_{index+1}/{j+1}')
        try:
            energy = relaxation.use_gulp_no_add(c)

        except Exception as error:
            energy = [0,0]
            print('There was a mistake in GULP')

        energies.append(energy[1])
        print(f'{j + 1} la energia es {energy[1]}')



    df[f'Test_{index}'] = energies
    df.to_csv(f'{folder}/{operator}/progress_log.csv', mode='w', header=True, index=True)


# BEST_ASE = read('/home/vito/PythonProjects/ASEProject/EA/new_minimum/ase/BEST_ASE.vasp')
# relaxation = Gulp_relaxation(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Heredity/ASE_1gulp')
# relaxation.use_gulp_no_add(BEST_ASE)


class MolCrystalOperatorTest:
    """Compare genetic operators on molecular crystal structures.

    Reads from a database created by start_e.py containing ASE Atoms
    with tags identifying molecular units. All operators use use_tags=True.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database created by start_e.py.
    out_dir : str, optional
        Output directory for results. Defaults to directory of db_path.
    """

    def __init__(self, db_path, out_dir=None):
        self.db_path = db_path
        self.da = DataConnection(db_path)
        self.slab = self.da.get_slab()
        self.out_dir = out_dir or os.path.dirname(db_path)
        os.makedirs(self.out_dir, exist_ok=True)
        self.atoms_to_opt = self.da.get_atom_numbers_to_optimize()
        self.n_top =  sum(value for element, value in self.atoms_to_opt.items())
        self.atom_numbers = [atomic_numbers[name] for name, _ in self.atoms_to_opt.items()]
        print(self.atoms_to_opt)
        print(self.atom_numbers)
        print(self.n_top)
        self.blmin = closest_distances_generator(
            atom_numbers=self.atom_numbers,
            ratio_of_covalent_radii=0.5
        )
        self.blmin_soft = closest_distances_generator(
            atom_numbers=self.atom_numbers,
            ratio_of_covalent_radii=0.1
        )
        self.cellbounds = CellBounds(
            bounds={
                'phi': [20, 150],
                'chi': [20, 150],
                'psi': [20, 150],
                'a': [2, 50],
                'b': [2, 50],
                'c': [2, 50],
            }
        )
        self.results_df = None

    def get_parent(self, parent_id):
        """Get an Atoms object from the DB with confid set."""
        a = self.da.get_atoms(parent_id)
        a.info['confid'] = parent_id
        return a

    def get_all_ids(self):
        """Return list of all candidate IDs in the database."""
        return [r.id for r in self.da.c.select()]

    def test_operator(self, operator_name, mom_id, dad_id=None, n_trials=100):
        """Test a single operator n_trials times, tracking time and success.

        Parameters
        ----------
        operator_name : str
            'heredity', 'softmut_modes', 'softmut_nomodes', or 'strainmut'
        mom_id : int
            DB id for the first parent.
        dad_id : int or None
            DB id for the second parent (required for heredity).
        n_trials : int
            Number of repetitions.

        Returns
        -------
        pd.DataFrame
            Columns: trial, operator, mom, dad, success, time_s
        """
        mom = self.get_parent(mom_id)
        dad = self.get_parent(dad_id) if dad_id else None

        if operator_name == 'heredity':
            op = CutAndSplicePairing(
                self.slab, self.n_top, self.blmin,
                p1=1.0, p2=0.0, minfrac=0.15,
                number_of_variable_cell_vectors=3,
                cellbounds=self.cellbounds,
                use_tags=True)
        elif operator_name == 'softmut_modes':
            modes_file = os.path.join(self.out_dir, 'test_used_modes.json')
            op = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0],
                              used_modes_file=modes_file, use_tags=True)
        elif operator_name == 'softmut_nomodes':
            op = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0],
                              used_modes_file=None, use_tags=True)
        elif operator_name == 'strainmut':
            op = StrainMutation(self.blmin, stddev=0.7,
                                cellbounds=self.cellbounds,
                                number_of_variable_cell_vectors=3,
                                use_tags=True)
            op.update_scaling_volume([mom], w_adapt=0.5, n_adapt=4)
        else:
            raise ValueError(f'Unknown operator: {operator_name}')

        records = []
        for i in range(n_trials):
            t0 = time.time()
            try:
                if operator_name == 'heredity':
                    baby = op.cross(mom, dad)
                else:
                    baby = op.mutate(mom)
                elapsed = time.time() - t0
                success = baby is not None
            except Exception as e:
                elapsed = time.time() - t0
                success = False
                print(f'  Trial {i+1}: {e}')

            records.append({
                'trial': i + 1,
                'operator': operator_name,
                'mom': mom_id,
                'dad': dad_id,
                'success': success,
                'time_s': round(elapsed, 6),
            })

        df = pd.DataFrame(records)
        rate = df['success'].mean() * 100
        avg_t = df['time_s'].mean()
        print(f'  {operator_name}: {n_trials} trials, '
              f'success={rate:.1f}%, avg_time={avg_t:.4f}s')
        return df

    def test_operator_with_relaxation(self, operator_name, mom_id, dad_id=None,
                                      n_trials=100, relax_parents=True):
        """Test operator with optional GULP relaxation of parents and offspring.

        Returns a DataFrame with an additional 'energy' column.
        """
        mom = self.get_parent(mom_id)
        dad = self.get_parent(dad_id) if dad_id else None

        relax_dir = os.path.join(self.out_dir, operator_name, f'pair_{mom_id}_{dad_id}')
        os.makedirs(relax_dir, exist_ok=True)

        if relax_parents:
            parent_relax = Gulp_relaxation(os.path.join(relax_dir, 'parents'))
            mom = parent_relax.use_gulp_no_add(mom)[0]
            if dad is not None:
                dad = parent_relax.use_gulp_no_add(dad)[0]
            mom.info['confid'] = mom_id
            if dad is not None:
                dad.info['confid'] = dad_id

        if operator_name == 'heredity':
            op = CutAndSplicePairing(
                self.slab, self.n_top, self.blmin,
                p1=1.0, p2=0.0, minfrac=0.15,
                number_of_variable_cell_vectors=3,
                cellbounds=self.cellbounds,
                use_tags=True)
        elif operator_name == 'softmut_modes':
            modes_file = os.path.join(relax_dir, 'used_modes.json')
            op = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0],
                              used_modes_file=modes_file, use_tags=True)
        elif operator_name == 'softmut_nomodes':
            op = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0],
                              used_modes_file=None, use_tags=True)
        elif operator_name == 'strainmut':
            op = StrainMutation(self.blmin, stddev=0.7,
                                cellbounds=self.cellbounds,
                                number_of_variable_cell_vectors=3,
                                use_tags=True)
            op.update_scaling_volume([mom], w_adapt=0.5, n_adapt=4)
        else:
            raise ValueError(f'Unknown operator: {operator_name}')

        records = []
        for i in range(n_trials):
            energy = None
            t0 = time.time()
            try:
                if operator_name == 'heredity':
                    baby = op.cross(mom, dad)
                else:
                    baby = op.mutate(mom)
                elapsed = time.time() - t0
                success = baby is not None

                if success:
                    try:
                        trial_relax = Gulp_relaxation(
                            os.path.join(relax_dir, f'trial_{i+1}'))
                        result = trial_relax.use_gulp_no_add(baby)
                        energy = result[1]
                    except Exception:
                        energy = None
            except Exception as e:
                elapsed = time.time() - t0
                success = False
                print(f'  Trial {i+1}: {e}')

            records.append({
                'trial': i + 1,
                'operator': operator_name,
                'mom': mom_id,
                'dad': dad_id,
                'success': success,
                'time_s': round(elapsed, 6),
                'energy': energy,
            })
            print(f'  {i+1}/{n_trials} energy={energy}')

        df = pd.DataFrame(records)
        save_progress(df, os.path.join(self.out_dir, 'operator_relaxed_progress.csv'))
        return df

    def run_comparison(self, parent_pairs, n_trials=100,
                       operators=('heredity', 'softmut_modes',
                                  'softmut_nomodes', 'strainmut')):
        """Run all operators for the given parent pairs.

        Parameters
        ----------
        parent_pairs : list of (int, int)
            List of (mom_id, dad_id) tuples.
        n_trials : int
            Trials per operator per pair.
        operators : tuple of str
            Which operators to test.
        """
        dfs = []
        for pair_idx, (mom_id, dad_id) in enumerate(parent_pairs):
            print(f'\n=== Pair {pair_idx + 1}: mom={mom_id}, dad={dad_id} ===')
            for op_name in operators:
                print(f'  Running {op_name}...')
                did = dad_id if op_name == 'heredity' else None
                df = self.test_operator(op_name, mom_id, did, n_trials)
                df['pair'] = pair_idx + 1
                dfs.append(df)
                save_progress(df, os.path.join(self.out_dir, 'operator_progress.csv'))

        self.results_df = pd.concat(dfs, ignore_index=True)
        return self.results_df

    def summary(self):
        """Summary statistics grouped by operator."""
        if self.results_df is None:
            raise ValueError('No results. Call run_comparison first.')
        return self.results_df.groupby('operator').agg(
            total_trials=('trial', 'count'),
            success_rate=('success', 'mean'),
            mean_time_s=('time_s', 'mean'),
            median_time_s=('time_s', 'median'),
            std_time_s=('time_s', 'std'),
            max_time_s=('time_s', 'max'),
        ).round(4)

    def save_results(self, filename='operator_comparison.csv'):
        if self.results_df is None:
            raise ValueError('No results.')
        path = os.path.join(self.out_dir, filename)
        self.results_df.to_csv(path, index=False)
        print(f'Saved to {path}')

    def plot_time_distribution(self, save=False):
        """Box plot of timing + bar chart of success rates."""
        if self.results_df is None:
            raise ValueError('No results.')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        self.results_df.boxplot(column='time_s', by='operator', ax=axes[0])
        axes[0].set_ylabel('Time (s)')
        axes[0].set_xlabel('Operator')
        axes[0].set_title('Time per Operation')

        summary = self.summary()
        summary['success_rate'].plot(kind='bar', ax=axes[1],
                                     color='steelblue', edgecolor='black')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_xlabel('Operator')
        axes[1].set_title('Success Rate')
        axes[1].set_ylim(0, 1.05)

        plt.suptitle('Molecular Crystal Operator Comparison')
        plt.tight_layout()

        if save:
            path = os.path.join(self.out_dir, 'operator_comparison.png')
            plt.savefig(path, dpi=150)
            print(f'Saved to {path}')
        plt.show()

    def plot_time_per_pair(self, save=False):
        """Timing distribution grouped by operator and pair."""
        if self.results_df is None:
            raise ValueError('No results.')

        operators = self.results_df['operator'].unique()
        n_ops = len(operators)
        fig, axes = plt.subplots(1, n_ops, figsize=(5 * n_ops, 5), sharey=True)
        if n_ops == 1:
            axes = [axes]

        for ax, op in zip(axes, operators):
            subset = self.results_df[self.results_df['operator'] == op]
            subset.boxplot(column='time_s', by='pair', ax=ax)
            ax.set_title(op)
            ax.set_xlabel('Pair')

        axes[0].set_ylabel('Time (s)')
        plt.suptitle('Time per Pair by Operator')
        plt.tight_layout()

        if save:
            path = os.path.join(self.out_dir, 'time_per_pair.png')
            plt.savefig(path, dpi=150)
            print(f'Saved to {path}')
        plt.show()

    def plot_energy_distribution(self, save=False):
        """Histogram of energies per operator (requires relaxed results)."""
        if self.results_df is None or 'energy' not in self.results_df.columns:
            raise ValueError('No energy data. Use test_operator_with_relaxation.')

        df = self.results_df.dropna(subset=['energy'])
        operators = df['operator'].unique()

        fig, ax = plt.subplots(figsize=(10, 6))
        for op in operators:
            subset = df[df['operator'] == op]['energy']
            ax.hist(subset, bins=30, alpha=0.5, edgecolor='black', label=op)

        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title('Offspring Energy Distribution by Operator')
        ax.legend()
        plt.tight_layout()

        if save:
            path = os.path.join(self.out_dir, 'energy_distribution.png')
            plt.savefig(path, dpi=150)
            print(f'Saved to {path}')
        plt.show()

