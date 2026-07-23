import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from ea.uspex.uspex26.run_uspex26 import count_pending
from ea.uspex.uspex26.worker import (
    ASE_MODE,
    calcfolder_mode,
    discover_calcfolders,
    write_calcfolder_result,
)


class Code20WorkerTests(unittest.TestCase):
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
