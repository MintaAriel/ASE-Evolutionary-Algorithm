from ase.ga.data import DataConnection
from ase import Atoms
from ase.visualize import view
from relax import Gulp_relaxation
from ase.io import write
from ase.io.cif import write_cif

class population_vis():
    def __init__(self, db_name):
        self.db_name = db_name
        self.da = DataConnection(self.db_name)

    def open_one_struc(self, i):
        slab = self.da.get_atoms(i)
        return slab

test1 =population_vis('/home/brian/PycharmProjects/ASEProject/runs_pyxtal/run_001/Mg4Al8O16_40.db')

view(test1.open_one_struc(3593))
view(test1.open_one_struc(3655))
view(test1.open_one_struc(3656))
view(test1.open_one_struc(3740))
#view(test1.open_one_struc(3699))

# start = test1.open_one_struc(692)
# view(final)
#
#
# final2 = Gulp_relaxation(start.copy()).use_gulp()
#
# # view(start)
# # view(final2)
# write_cif('final2.cif', final2)
#
# print(final)
# print(final2)
# view(final2)
