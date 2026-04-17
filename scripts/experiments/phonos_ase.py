import numpy as np
from deepmd_template import DeepMDRelaxation, RelaxConfig
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pathlib import Path
import time
from ase.io import read
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo
from ase.build import make_supercell

config = RelaxConfig()

deep = DeepMDRelaxation(config)
deep.model_key = 'deepmd_d3'
calc = deep.build_calculator(models_dir=Path('/home/vito/PythonProjects/ASEProject/container_gpu_2/models'),
                             device='cuda',
                             threads=2)

cif_path = '/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/128707/deepmd_d3_in/final.cif'

atom = read(cif_path)
atom.calc =calc
out_dir = Path('/home/vito/PythonProjects/ASEProject/EA/test/phonopy')

def phonons_at_gamma():

    vib_dir =  out_dir / 'vib'
    vib = Vibrations(atom, name= vib_dir)
    vib.run()
    vib.summary()

    energy = vib.get_zero_point_energy()

    print('ZPE:', energy )

    #vib.write_mode(211)

def phonons_BZ():
    # S = np.diag([2, 2, 2])  # 2x2x2 supercell
    # supercell = make_supercell(atom, S)
    # print("supercell =", supercell)
    delta = 0.01
    kpts = (4, 4, 4)
    npts = 500
    width = 5e-3
    temperatures = np.arange(0, 1001, 50)
    vib_dir = out_dir / 'vib_phonons'

    E0 = atom.get_potential_energy()

    ph = Phonons(atom, calc, supercell=(1, 1, 1), delta=delta, name=vib_dir)
    ph.run()
    ph.read(acoustic=True)

    dosdata = ph.get_dos(kpts=kpts).sample_grid(npts=npts, width=width)
    energies = dosdata.get_energies()
    dos = dosdata.get_weights()

    # ZPE from raw DOS integral
    zpe_raw = 0.5 * np.trapezoid(dos * energies, energies)

    # ZPE from DOS integral after removing non-positive energies
    mask = energies > 1e-12
    energies_pos = energies[mask]
    dos_pos = dos[mask]
    zpe_pos = 0.5 * np.trapezoid(dos_pos * energies_pos, energies_pos)

    # CrystalThermo for the whole simulation cell
    thermo = CrystalThermo(
        phonon_energies=energies_pos,
        phonon_DOS=dos_pos,
        potentialenergy=E0,
        formula_units=1,
    )

    # ZPE from CrystalThermo at very small temperature
    zpe_ct = thermo.get_internal_energy(temperature=1e-12) - E0

    print(f"Static energy (eV):            {E0:.10f}")
    print(f"ZPE raw integral (eV):         {zpe_raw:.10f}")
    print(f"ZPE positive-only integral:    {zpe_pos:.10f}")
    print(f"ZPE CrystalThermo (eV):        {zpe_ct:.10f}")
    print()
    print("# T(K)   F(eV)")

    for T in temperatures:
        if T < 1e-12:
            F = E0 + zpe_ct
        else:
            F = thermo.get_helmholtz_energy(T)
        print(f"{T:8.2f}  {F: .10f}")

    ph.clean()


phonons_at_gamma()
