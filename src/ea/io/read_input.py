#!/usr/bin/env python3
"""
Parser for evolutionary algorithm input files with custom format.
Supports:
- Block entries: % key ... % Endkey with value on the next line.
- Colon-separated: value : key (value may contain parentheses, commas).
- Ignores section headers (lines starting with '*') and empty lines.
- Automatically converts values to int, float, or leaves as string.

``parse_uspex26_workflow`` reads the extra MODELS / CONDA settings that the
USPEX 26 DeepMD workers accept in a USPEX ``INPUT.txt``.
"""

import os
import re


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


# ---------------------------------------------------------------------------
# USPEX 26 worker settings (MODELS / CONDA) embedded in a USPEX INPUT.txt
# ---------------------------------------------------------------------------

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_MODEL_LINE_RE = re.compile(rf"^({_FLOAT})\s*:\s*(.+?)\s*$")
_IDENTIFIER_RE = re.compile(r"^\w+$")


def _is_header(line):
    return line.startswith('*')


def _header_name(line):
    """``*   MODELS   *`` -> ``'MODELS'``; decorative ``*****`` -> ``''``."""
    return line.strip('* \t').strip()


def _block_values(lines, name):
    """Value lines of a ``% <name> ... % End<name>`` block, or None if absent."""
    start_re = re.compile(rf"^%\s*{name}\s*$", re.IGNORECASE)
    end_re = re.compile(r"^%\s*End", re.IGNORECASE)
    for i, line in enumerate(lines):
        if start_re.match(line):
            values = []
            for value in lines[i + 1:]:
                if end_re.match(value):
                    return values
                if value and not _is_header(value):
                    values.append(value)
            raise ValueError(f"block '% {name}' has no closing '% End' line")
    return None


def _parse_model_entry(line, model_keys):
    """``0.2: /path/model.pt2`` -> ``(0.2, '/path/model.pt2')`` or None.

    USPEX's own ``value : key`` lines (``10 : numParallelCalcs``) share this
    shape, so a bare identifier is only accepted when it is a known model key.
    """
    m = _MODEL_LINE_RE.match(line)
    if not m:
        return None
    fmax, model = float(m.group(1)), m.group(2)
    if _IDENTIFIER_RE.match(model) and model not in model_keys:
        return None
    return fmax, model


def parse_uspex26_workflow(filepath, model_keys=()):
    """Read the DeepMD worker settings from a USPEX 26 ``INPUT.txt``.

    Two optional settings are recognised:

    MODELS -- staged relaxation, one model per stage, written either under a
    ``*  MODELS  *`` section header or as a ``% MODELS ... % EndMODELS`` block::

        ******************************************
        *               MODELS                   *
        ******************************************
        0.2:   /path/to/fast_compressed_model.pt2
        0.001: /path/to/accurate_model.pt2

    Each line is ``<target fmax> : <model>``; a structure is relaxed with the
    first model down to its fmax, then with the next one, and so on.  The
    model is a file path or a key of ``ea.uspex.models.MODELS``.

    CONDA -- conda environment that runs the workers, as ``<env> : CONDA``
    or a ``% CONDA ... % EndCONDA`` block.

    Returns ``{'models': [(fmax, model), ...], 'conda': env}``; ``models`` is
    an empty list and ``conda`` None when the INPUT does not set them.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with open(filepath, 'r', errors='replace') as f:
        lines = [line.strip() for line in f]
    model_keys = set(model_keys)

    models = []
    block = _block_values(lines, 'MODELS')
    if block is not None:
        for line in block:
            entry = _parse_model_entry(line, model_keys)
            if entry is None:
                raise ValueError(f"bad MODELS line {line!r}; "
                                 "expected '<fmax> : <model>'")
            models.append(entry)
    else:
        in_section = False
        for line in lines:
            if _is_header(line):
                name = _header_name(line)
                if name:
                    in_section = name.upper() == 'MODELS'
                continue
            if not in_section or not line:
                continue
            entry = _parse_model_entry(line, model_keys)
            if entry is None:
                # First ordinary USPEX line after the model list ends it.
                in_section = False
                continue
            models.append(entry)

    for fmax, model in models:
        if fmax <= 0:
            raise ValueError(f"MODELS fmax must be positive: {fmax} : {model}")
    targets = [fmax for fmax, _ in models]
    if any(b >= a for a, b in zip(targets, targets[1:])):
        raise ValueError(
            "MODELS must be listed from the loosest to the tightest fmax "
            f"(strictly decreasing), got {targets}")

    conda = None
    block = _block_values(lines, 'CONDA')
    if block:
        conda = block[0].split()[0]
    else:
        for line in lines:
            if ':' in line and not _is_header(line):
                value, key = (part.strip() for part in line.split(':', 1))
                if key.split('#', 1)[0].strip().upper() == 'CONDA' and value:
                    conda = value
                    break

    return {'models': models, 'conda': conda}
