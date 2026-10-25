"""CPU-only checks of INPUT recipes and actual optimizer/model switching."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms

from ea.io.read_input import parse_uspex26_workflow
from ea.uspex.uspex26 import worker


class WorkflowTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'INPUT.txt'
            path.write_text(text)
            return parse_uspex26_workflow(path)

    def test_sections_keep_stage_order_steps_models_and_conda(self):
        result = self.parse('''
*****
* MODELS *
*****
0.2: /models/fast.pt2
0.001: /models/accurate.pt2
*****
* STAGES *
*****
0.1 : FIRE : 500
0.03 : LBFGS : 1200
0.01 : LBFGS : 1200
0.005 : LBFGS : 1200
0.002 : LBFGS : 1200
0.001 : LBFGS : 1200
deepmd_dpa4 : CONDA
10 : numParallelCalcs
''')
        self.assertEqual(result['models'], [(.2, '/models/fast.pt2'),
                                            (.001, '/models/accurate.pt2')])
        self.assertEqual(result['stages'], [(.1, 'FIRE', 500),
                         *[(f, 'LBFGS', 1200) for f in worker.LBFGS_STAGES]])
        self.assertEqual(result['conda'], 'deepmd_dpa4')

    def test_blocks_and_numeric_parameter_after_stage_section(self):
        for text in ('% STAGES\n0.05 : lbfgs : 7\n% EndSTAGES\n',
                     '* STAGES *\n0.05 : LBFGS : 7\n10 : numParallelCalcs\n'):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text)['stages'], [(.05, 'LBFGS', 7)])

    def test_omitted_stages_and_input_use_defaults(self):
        self.assertEqual(self.parse('10 : numParallelCalcs\n')['stages'], [])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(worker.load_optimization_profile(tmp), {})
            (Path(tmp) / 'INPUT.txt').write_text('* MODELS *\n0.2: /fast.pt2\n')
            self.assertEqual(worker.load_optimization_profile(tmp), {})

    def test_bad_stages_are_rejected(self):
        for row in ('0.1 : BFGS : 10', '0.1 : FIRE : 0',
                    '0 : LBFGS : 3', 'nan : FIRE : 3',
                    '0.1 : LBFGS : 1.5', '0.1 : FIRE'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.parse('* STAGES *\n' + row + '\n')

    def test_profile_preserves_steps_and_smoke_explicitly_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / 'INPUT.txt').write_text(
                '* STAGES *\n0.08 : FIRE : 17\n0.004 : LBFGS : 29\n')
            self.assertEqual(worker.load_optimization_profile(tmp), {
                'optimization_stages': [(.08, 'FIRE', 17), (.004, 'LBFGS', 29)]})
            self.assertEqual(worker.load_optimization_profile(tmp, smoke=True), {
                'optimization_stages': [(.1, 'FIRE', 2), (.03, 'LBFGS', 2)]})


class OptimizationTests(unittest.TestCase):
    @staticmethod
    def atom(x):
        return Atoms('H', positions=[[x, 0, 0]], cell=[5, 5, 5], pbc=True)

    @staticmethod
    def harmonic(atoms_list, calculator):
        # Each model has its own stiffness and energy offset.
        stiffness, offset = calculator
        positions = [atoms.positions for atoms in atoms_list]
        return (np.array([offset + .5 * stiffness * np.square(p).sum()
                          for p in positions]),
                [-stiffness * p for p in positions],
                [np.zeros(6) for _ in atoms_list])

    def test_real_optimizers_obey_recipe_step_limits_and_final_energy(self):
        created = []

        def record(cls):
            def make(*args, **kwargs):
                optimizer = cls(*args, **kwargs)
                created.append(optimizer)
                return optimizer
            return make

        stages = [(1e-12, 'FIRE', 2), (1e-13, 'LBFGS', 3)]
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(worker, 'batch_calculator_deepmd', self.harmonic), \
                patch.object(worker, 'ParallelFIRE', side_effect=record(worker.ParallelFIRE)), \
                patch.object(worker, 'ParallelLBFGS', side_effect=record(worker.ParallelLBFGS)):
            atoms, energies = worker.run_full_optimization(
                [self.atom(1)], (1., 0.), tmp, optimization_stages=stages)
        self.assertEqual([opt.nsteps_done for opt in created], [2, 3])
        self.assertEqual([opt.max_steps for opt in created], [2, 3])
        self.assertEqual([opt.fmax for opt in created], [1e-12, 1e-13])
        self.assertLess(atoms[0].positions[0, 0], 1.)
        self.assertAlmostEqual(energies[0], .5 * np.square(atoms[0].positions).sum())
        self.assertAlmostEqual(atoms[0].get_potential_energy(), energies[0])

    def test_default_recipe_is_independent_of_model_thresholds(self):
        created = []

        def record(cls, name):
            def make(*args, **kwargs):
                created.append((name, kwargs['fmax'], kwargs['max_steps']))
                return cls(*args, **kwargs)
            return make

        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(worker, 'batch_calculator_deepmd', self.harmonic), \
                patch.object(worker, 'ParallelFIRE', side_effect=record(worker.ParallelFIRE, 'FIRE')), \
                patch.object(worker, 'ParallelLBFGS', side_effect=record(worker.ParallelLBFGS, 'LBFGS')):
            worker.run_full_optimization([self.atom(0)],
                                         [(.2, (1., 0.)), (.001, (2., 10.))], tmp)
        self.assertEqual(created, [('FIRE', .1, 500),
                         *[('LBFGS', f, 1200) for f in worker.LBFGS_STAGES]])

    def test_switching_is_per_structure_strict_and_persists_between_stages(self):
        models = [(.2, (1., 0.)), (.001, (2., 10.))]
        evaluator = worker.ModelSwitchingEvaluator(models, 3)
        opt = worker.ParallelLBFGS([self.atom(.1), self.atom(.2), self.atom(.5)],
                                  batch_evaluator=evaluator, logfile=None)
        evaluator.bind(opt.states)
        with patch.object(worker, 'batch_calculator_deepmd', self.harmonic), \
                contextlib.redirect_stdout(io.StringIO()):
            energies, forces, _ = evaluator([s.atoms for s in opt.states])
            self.assertEqual(evaluator.indices, [1, 0, 0])
            self.assertAlmostEqual(energies[0], 10.01)
            self.assertAlmostEqual(forces[0][0, 0], -.2)
            # Only a subset remains active; model indices must not shift.
            evaluator([opt.states[2].atoms])
            self.assertEqual(evaluator.indices, [1, 0, 0])
            following = worker.ParallelFIRE([s.atoms for s in opt.states],
                                            batch_evaluator=evaluator, logfile=None)
            evaluator.bind(following.states)
            following.states[0].atoms.positions[0, 0] = 1.
            evaluator([s.atoms for s in following.states])
            self.assertEqual(evaluator.indices, [1, 0, 0])

    def test_cell_forces_prevent_premature_switch(self):
        evaluator = worker.ModelSwitchingEvaluator([(.2, 'fast'), (.001, 'accurate')], 1)
        opt = worker.ParallelFIRE([self.atom(0)], batch_evaluator=evaluator, logfile=None)
        evaluator.bind(opt.states)

        def stress_only(atoms, calc):
            self.assertEqual(calc, 'fast')
            return [0.], [np.zeros((1, 3))], [np.array([.1, .1, .1, 0, 0, 0])]

        with patch.object(worker, 'batch_calculator_deepmd', stress_only):
            evaluator([opt.states[0].atoms])
        self.assertEqual(evaluator.indices, [0])

    def test_switch_resets_lbfgs_history_and_rechecks_new_model_convergence(self):
        evaluator = worker.ModelSwitchingEvaluator([(.2, (.1, 0.)), (.001, (1., 10.))], 1)
        opt = worker.ParallelLBFGS([self.atom(1)], batch_evaluator=evaluator,
                                  fmax=.15, logfile=None)
        evaluator.bind(opt.states)
        state = opt.states[0]
        state.iteration = 1
        state.s, state.y, state.rho = [np.ones(12)], [np.ones(12)], [1.]
        state.r0, state.f0 = np.zeros(12), np.zeros(12)
        with patch.object(worker, 'batch_calculator_deepmd', self.harmonic), \
                contextlib.redirect_stdout(io.StringIO()):
            opt.step()
        self.assertEqual(evaluator.indices, [1])
        self.assertFalse(state.converged)  # accurate model fmax=1 > .15
        self.assertEqual(state.s, [])
        self.assertEqual(state.iteration, 1)  # first step on the new model


if __name__ == '__main__':
    unittest.main()
