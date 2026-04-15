from phonopy import load
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

# Load from YAML (everything inside)
# phonon = load("/home/vito/PythonProjects/ASEProject/EA/test/phonopy/python/phonopy_params_relaxed2.yaml")
phonon = load('/home/vito/PythonProjects/ASEProject/EA/test/phonopy/python/phonopy_params_cuda.yaml')

# 1. Mesh (ALWAYS required first)
phonon.run_mesh([20, 20, 20], with_eigenvectors=True)

# 2. DOS
# phonon.run_total_dos()
# phonon.plot_total_dos(with_tight_frequency_range=False).show()
band_paths = [
    [[0, 0, 0], [0.5, 0, 0]],
    [[0.5, 0, 0], [0.5, 0.5, 0]],
    [[0.5, 0.5, 0], [0, 0, 0]]
]
# phonon.run_band_structure(band_paths, with_eigenvectors=True)
# phonon.plot_band_structure().show()
phonon.symmetry
phonon.auto_band_structure()
phonon.plot_band_structure().show()

freqs = phonon._mesh.frequencies
eigenvectors = phonon._mesh.eigenvectors

print(freqs.min())

print(freqs)
print(eigenvectors)

from ase.visualize import view


def phonopy_atoms_to_ase(patoms: PhonopyAtoms):
    """Convert PhonopyAtoms -> ASE Atoms."""
    from ase import Atoms
    return Atoms(
        symbols=patoms.symbols,
        cell=patoms.cell,
        scaled_positions=patoms.scaled_positions,
        pbc=True,
    )




q_index = 0
mode_index = 30
#
# mode = eigenvectors[q_index][mode_index]
# # mode = mode.real.reshape(-1, 3)
# print(mode)
# mode = mode.real.reshape(-1, 3)
# print(mode)
# atoms = phonopy_atoms_to_ase(phonon.supercell)
#
# images = []
#
# for t in np.linspace(0, 2*np.pi, 40):
#     disp = np.cos(t) * mode
#     img = atoms.copy()
#     img.positions += 0.2 * disp
#     images.append(img)
#
# # write("phonon_mode.traj", images)
# view(images)


# # 3. (Optional) Band structure
# phonon.run_band_structure(
#     paths=[[[0,0,0],[0.5,0,0]]],  # simple example path
#     npoints=50
# )
#
# # 4. Now plotting works
# phonon.plot_band_structure_and_dos()