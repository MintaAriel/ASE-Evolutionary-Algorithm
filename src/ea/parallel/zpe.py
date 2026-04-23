from ase.io import read
from ase.vibrations import Vibrations

str18 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_18_POSCARS')
str19 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_19_POSCARS')

test_dir = '/home/vito/PythonProjects/ASEProject/EA/test/phonopy/parallel'

batch = [str18, str19]

def phonons_at_gamma():

    vib_dir =  out_dir / 'vib'
    if vib_dir.exists() and vib_dir.is_dir():
        shutil.rmtree(vib_dir)

    vib = Vibrations(atom, name= vib_dir)
    vib.run()
    vib.summary()

    energy = vib.get_zero_point_energy()

    print('ZPE:', energy )
