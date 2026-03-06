#!/usr/bin/env python3
"""
Parser for evolutionary algorithm input files with custom format.
Supports:
- Block entries: % key ... % Endkey with value on the next line.
- Colon-separated: value : key (value may contain parentheses, commas).
- Ignores section headers (lines starting with '*') and empty lines.
- Automatically converts values to int, float, or leaves as string.
"""

import os
import re

from pymatgen.analysis.fragmenter import open_ring


def parse_value(val_str):
    """
    Convert a string to int, float, or return stripped string.
    """
    v = val_str.strip()
    # Try integer
    try:
        return int(v)
    except ValueError:
        pass
    # Try float
    try:
        return float(v)
    except ValueError:
        pass
    # Otherwise return stripped string
    return v


def parse_input_file(filepath):
    """
    Parse the input file and return a dictionary of parameters.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    params = {}
    params.setdefault('firstGenPOSCAR', 0)


    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        i += 1

        # Skip empty lines and section headers (starting with '*')
        if not line or line.startswith('*'):
            continue

        # Handle block start: % key (but not % End...)
        if line.startswith('%') and not line.startswith('% End'):
            # Extract key after '% '
            key = line[2:].strip()  # remove '% '

            # Find the next non‑empty, non‑header line that is not an end marker
            while i < n:
                val_line = lines[i].strip()
                i += 1
                if not val_line or val_line.startswith('*'):
                    continue
                # Stop if we hit the corresponding end marker
                if val_line.startswith('% End'):
                    break
                # Otherwise this is the value line
                params[key] = parse_value(val_line)
                break
            # Continue to next line (the end marker will be skipped automatically)
            continue

        # Handle colon-separated lines: value : key
        if ':' in line:
            # Split on first colon only
            parts = line.split(':', 1)
            if len(parts) == 2:
                val_str, key_str = parts
                key = key_str.strip()
                # value may have trailing comments after colon? not in sample, but safe
                # Remove any inline comment after colon (starting with '#')
                if '#' in key:
                    key = key.split('#', 1)[0].strip()
                params[key] = parse_value(val_str)
                continue

        # If we get here, the line format is unrecognised – warn but ignore
        print(f"Warning: Skipping unrecognised line: {line}")

    splits_str = params['splits']

    parts = splits_str.split(',')

    splits_list = []
    for value in parts:
        split_values = value.split('_')
        split = int(split_values[0])
        split_ratio = int(split_values[1])
        splits_list.append((split, split_ratio))


    split_dict = {(x,): y for x, y in splits_list}

    params['splits'] = split_dict


    return params


