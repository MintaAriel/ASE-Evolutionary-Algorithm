from ase.ga.data import DataConnection
from ase.visualize import view
from ase.io import write, read
import os

da = DataConnection('/home/vito/Documents/UMA/Mg4Al8O16_40.db')
# for i in range(2,23):
#     a = da.get_atoms(i)
#     write(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Seeds2/{i}.vasp', a   , format='vasp')
# #

# mom = [i for i in range(45,65,2)]
# dad = [i for i in range(46,66,2)]

mom = [i for i in range(2,22,2)]
dad = [i for i in range(3,23,2)]
print(mom,dad)

#path = '/home/vito/PythonProjects/ASEProject/EA/new_minimum/ALL_BEST'

# for index, i in enumerate(mom):
#     print(i, dad[index])
#     parents = [i, dad[index]]
#     all_seeds = ''
#     for parent in parents:
#         print(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Seeds2/{parent}.vasp')
#         with open(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Seeds2/{parent}.vasp', 'r') as f:
#             #best = read(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Seeds/{parent}.vasp', format="vasp")
#             best = f.read()
#             print(best)
#             all_seeds += best
#
#     print(all_seeds)
#     with open(f"/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Seeds/heredity2/Pair_{index+1}.vasp", "w") as f:
#          f.write(all_seeds)