def parse_connections(input_file, output_file="connections"):
    bonds = set()

    with open(input_file, "r") as f:
        lines = f.readlines()

    atom_index = 0

    for line in lines:
        parts = line.split()

        # Only process valid atom lines
        # Must have at least 8 columns
        if len(parts) < 8:
            continue

        # First column must be a chemical symbol (letter)
        if not parts[0].isalpha():
            continue

        atom_index += 1  # 1-based indexing

        neighbors = parts[4:5]

        for n in neighbors:
            try:
                j = int(n)
            except ValueError:
                continue

            if j != 0:
                a = min(atom_index, j)
                b = max(atom_index, j)
                bonds.add((a, b))

    with open(output_file, "w") as out:
        for a, b in sorted(bonds):
            out.write(f"connect    {a}    {b}\n")

def parse_connections2(input_file, output_file="connections"):
    with open(input_file, "r") as f:
        lines = f.readlines()

    counts_idx = None
    for i, line in enumerate(lines):
        if 'V2000' in line:
            counts_idx = i
            break
    if counts_idx is None:
        raise ValueError(f"Not a V2000 MOL file: {input_file}")

    n_atoms = int(lines[counts_idx][0:3])
    n_bonds = int(lines[counts_idx][3:6])

    bond_block_start = counts_idx + 1 + n_atoms
    bond_lines = lines[bond_block_start:bond_block_start + n_bonds]

    bonds = set()
    for line in bond_lines:
        a1 = int(line[0:3])
        a2 = int(line[3:6])
        a = min(a1, a2)
        b = max(a1, a2)
        bonds.add((a, b))

    with open(output_file, "w") as out:
        for a, b in sorted(bonds):
            out.write(f"connect    {a}    {b}\n")


# Usage
# parse_connections("/home/vito/PythonProjects/ASEProject/EA/data/theophylline/MOL_1")