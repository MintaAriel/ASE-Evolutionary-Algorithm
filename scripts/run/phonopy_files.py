from phonopy import load
from phonopy.file_IO import parse_FORCE_SETS
import numpy as np


# Load structure + displacements from YAML
phonon = load("/home/vito/PythonProjects/ASEProject/EA/test/phonopy/examples/phonopy.yaml")

# Parse FORCE_SETS
dataset = parse_FORCE_SETS(filename="/home/vito/PythonProjects/ASEProject/EA/test/phonopy/examples/FORCE_SETS")

# Inject dataset
phonon.dataset = dataset

# Build force constants
phonon.produce_force_constants()

print(phonon.force_constants.shape)

phonon.run_mesh([20, 20, 20], with_eigenvectors=True)

phonon.run_total_dos()
# phonon.plot_total_dos(with_tight_frequency_range=False).show()

mesh = phonon.get_mesh_dict()

frequencies = mesh["frequencies"]   # shape: (Nq, Nmodes)
weights = mesh["weights"]           # shape: (Nq,)

hbar = 6.582119569e-16  # eV·s

zpe = 0.0

for i in range(len(weights)):
    w_q = weights[i]
    freqs = frequencies[i]

    # Remove imaginary frequencies
    freqs = freqs[freqs > 0]

    zpe += w_q * np.sum(freqs)

zpe *= 0.5 * hbar
zpe /= np.sum(weights)  # normalize

print("ZPE (eV per unit cell):", zpe)

print('ZPE via DOS', phonon.run_total_dos())