
with open("/home/brian/USPEX_work/EX02/results3/symmetrized_structures.cif", "r") as g:
    dif_str = g.read().splitlines()

structure_number = []
symmetry_number = []
indices_start= []
for index, line, in enumerate(dif_str):
    if line.strip().startswith("data_findsym-STRUC-"):
        structure_number.append(int(line.strip().split("-")[-1]))
        indices_start.append(index)
    if line.strip().startswith("_symmetry_Int_Tables_number"):
        symmetry_number.append(int(line.strip().split()[-1]))

indices_start.append(len(dif_str))

cif_files = {}
#cif_files['ID'] =
cif = []


for i in range(len(indices_start) - 1):
    print(indices_start[i],indices_start[i+1])
    cif.append(dif_str[indices_start[i]:indices_start[i+1]])

print(cif[-1])
print(len(cif))


print(len(indices_start))
print(indices_start)
print(len(structure_number))