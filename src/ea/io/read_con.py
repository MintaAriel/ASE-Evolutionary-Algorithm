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

# Usage
parse_connections("/home/vito/PythonProjects/ASEProject/EA/data/theophylline/MOL_1")