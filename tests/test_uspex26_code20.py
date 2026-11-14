import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.io import read, write

from ea.uspex.uspex26.run_uspex26 import count_pending
from ea.uspex.uspex26.worker import (
    ASE_MODE,
    calcfolder_mode,
    discover_calcfolders,
    write_calcfolder_result,
)
from ea.uspex.uspex26.calcfolders import VERSION_ENV, detect_uspex_version
from ea.uspex.uspex26.worker_gpu import build_task


class Code20WorkerTests(unittest.TestCase):
    def test_release_detection_and_explicit_wrapper_override(self):
        self.assertEqual(detect_uspex_version('/bin/uspex_2622'), '2622')
        self.assertEqual(detect_uspex_version('/bin/uspex_2616.sh'), '2616')
        self.assertIsNone(detect_uspex_version('/bin/uspex'))
        self.assertEqual(detect_uspex_version('/bin/wrapper', '26.2.2'), '2622')
        with self.assertRaises(ValueError):
            detect_uspex_version('/bin/wrapper', 'invalid')

    def test_2622_attempts_are_discovered_and_published_independently(self):
        atoms = Atoms('CH', positions=[[0, 0, 0], [1, 0, 0]])
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {VERSION_ENV: '2622'}):
            workdir = Path(tmp)
            folders = []
            for system in (1, 2):
                cf = workdir / 'Calculation' / 'Relaxation' / f'Calcfold_{system}_1' / 'try_1'
                cf.mkdir(parents=True)
                write(cf / 'input.xyz', atoms, format='extxyz')
                folders.append(cf)
            self.assertEqual([cf for _, cf in discover_calcfolders(workdir)], folders)
            self.assertEqual(count_pending(workdir), 2)
            tasks = [build_task(cf, ASE_MODE, workdir, {}, False) for cf in folders]
            self.assertEqual(len({task['out_dir'] for task in tasks}), 2)
            for cf, task in zip(folders, tasks):
                self.assertEqual(Path(task['out_dir']).relative_to(workdir),
                                 Path('gpu_out') / cf.parent.name / 'try_1')
            write_calcfolder_result(folders[0], atoms, -10., ASE_MODE)
            self.assertTrue((folders[0] / 'output.xyz').is_file())
            self.assertEqual([cf for _, cf in discover_calcfolders(workdir)], folders[1:])
            self.assertEqual(count_pending(workdir), 1)

    def test_2622_selects_only_latest_retry_numerically(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {VERSION_ENV: '2622'}):
            workdir = Path(tmp)
            parent = workdir / 'Calculation' / 'Relaxation' / 'Calcfold_1_1'
            for attempt in (1, 2, 10):
                cf = parent / f'try_{attempt}'
                cf.mkdir(parents=True)
                (cf / 'input.xyz').write_text('input')
            self.assertEqual([cf for _, cf in discover_calcfolders(workdir)], [parent / 'try_10'])
            (parent / 'try_10' / 'output.xyz').touch()
            self.assertEqual(discover_calcfolders(workdir), [])

    def test_older_release_retains_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {VERSION_ENV: '2616'}):
            workdir = Path(tmp)
            flat = workdir / 'Calculation' / 'Calcfold_1_1'
            nested = workdir / 'Calculation' / 'Relaxation' / 'Calcfold_2_1' / 'try_1'
            for cf in (flat, nested):
                cf.mkdir(parents=True)
                (cf / 'input.xyz').write_text('input')
            self.assertEqual([cf for _, cf in discover_calcfolders(workdir)], [flat])
            self.assertEqual(count_pending(workdir), 1)

    def test_output_xyz_preserves_atom_order_and_exposes_energy(self):
        atoms = Atoms(
            symbols=["C", "N", "C", "H", "O", "N"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            calcfolder = Path(tmp) / "Calcfold_1_1"
            calcfolder.mkdir()
            energy = -123.456789
            write_calcfolder_result(calcfolder, atoms, energy, ASE_MODE)

            output = calcfolder / "output.xyz"
            self.assertTrue(output.is_file())
            self.assertFalse((calcfolder / ".output.xyz.tmp").exists())
            self.assertIn(f"energy={energy}", output.read_text().splitlines()[1])

            result = read(output, format="extxyz")
            self.assertEqual(result.get_chemical_symbols(), atoms.get_chemical_symbols())
            self.assertAlmostEqual(result.get_potential_energy(), energy)

    def test_code20_folder_is_pending_only_until_output_is_published(self):
        atoms = Atoms("CNCH", positions=[[0, 0, 0], [1, 0, 0],
                                          [2, 0, 0], [3, 0, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            calcfolder = workdir / "Calculation" / "Calcfold_1_1"
            calcfolder.mkdir(parents=True)
            write(calcfolder / "input.xyz", atoms, format="extxyz")

            self.assertEqual(calcfolder_mode(calcfolder), ASE_MODE)
            self.assertEqual(len(discover_calcfolders(workdir)), 1)
            self.assertEqual(count_pending(workdir), 1)

            write_calcfolder_result(calcfolder, atoms, -10.0, ASE_MODE)
            self.assertEqual(discover_calcfolders(workdir), [])
            self.assertEqual(count_pending(workdir), 0)


if __name__ == "__main__":
    unittest.main()
