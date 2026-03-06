import os.path

from ase.ga.data import DataConnection
from ase.io import read, write
from ase.build import sort
from multipart import file_path


class write_POSCAR():
    def __init__(self, db, out_dir):
        self.db = db
        self.da = DataConnection(self.db)
        self.out_dir = out_dir
        self.poscar_name = 'POSCAR'

    def create(self, n_first):
        file_path = os.path.join(self.out_dir, self.poscar_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"POSCAR has been deleted.")
        else:
            print(f"POSCAR does not exist.")


        temp_poscar = os.path.join(self.out_dir, 'poscar')
        for i in range(n_first):
            crystal = self.da.get_atoms(i+2)
            crystal_sorted = sort(crystal)
            # write(temp_poscar, crystal_sorted, format='vasp')
            # write(temp_poscar, crystal_sorted, format='vasp', direct=False)
            write(temp_poscar, crystal_sorted,format="vasp", vasp5=True,  direct=True)

            with open(temp_poscar, 'r') as f:
                content = f.read()

            with open(os.path.join(self.out_dir, self.poscar_name), 'a') as master_file:
                master_file.write(content)
                # master_file.write(content + "\n\n")

            print(f"Structure {i+1} saved into POSCAR")

        os.remove(temp_poscar)




