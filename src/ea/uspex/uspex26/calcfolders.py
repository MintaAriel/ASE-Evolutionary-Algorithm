"""Version-specific USPEX calculation folders, without ASE dependencies."""

import os
import re
from pathlib import Path


ASE_MODE = "ase"
USER_CODE_MODE = "user_code"
VERSION_ENV = "USPEX26_VERSION"
_CALC_RE = re.compile(r"CalcFold(\d+)$")
_CALC26_RE = re.compile(r"Calcfold_(\d+)_(.+)$")
_TRY_RE = re.compile(r"try_(\d+)$")


def detect_uspex_version(executable, override=None):
    """Use an explicit release, or the release in e.g. ``uspex_2622``."""
    if override is not None:
        version = str(override).replace(".", "")
        if not re.fullmatch(r"26\d{2}", version):
            raise ValueError("USPEX version must be a release such as 2622")
        return version
    match = re.search(r"(?<!\d)(26\d{2})(?!\d)", Path(executable).name)
    return match.group(1) if match else None


def calcfolder_mode(cf):
    """The ready input protocol; ASE preserves molecular atom order."""
    cf = Path(cf)
    for name, mode in (("input.xyz", ASE_MODE), ("geom.in", USER_CODE_MODE)):
        path = cf / name
        if path.is_file() and path.stat().st_size > 0:
            return mode
    return None


def _pending(cf):
    mode = calcfolder_mode(cf)
    if mode == ASE_MODE:
        return not (cf / "output.xyz").is_file()
    if mode == USER_CODE_MODE:
        return not (cf / "energy.txt").is_file()
    return False


def calcfolder_name(cf):
    """Unique output path for a system, step and optional retry attempt."""
    cf = Path(cf)
    if _TRY_RE.fullmatch(cf.name) and _CALC26_RE.fullmatch(cf.parent.name):
        return str(Path(cf.parent.name) / cf.name)
    return cf.name


def discover_calcfolders(workdir, version=None):
    """Return pending ``(sort_key, execution_directory)`` pairs.

    2622 puts relaxation attempts in
    ``Calculation/Relaxation/Calcfold_<system>_<step>/try_<n>``.
    Only the newest attempt may publish a result. Older releases retain
    their flat layout. The launcher passes its selected release to workers
    through USPEX26_VERSION; standalone workers can set that variable too.
    """
    workdir = Path(workdir)
    version = str(version if version is not None else os.environ.get(VERSION_ENV, ""))
    out = []
    for cf in workdir.iterdir():
        match = _CALC_RE.fullmatch(cf.name)
        if cf.is_dir() and match and _pending(cf):
            out.append(((0, int(match.group(1)), "", 0), cf))

    for base in ("Calculation", "CalculationTemp"):
        bases = [workdir / base]
        if version == "2622":
            bases.append(workdir / base / "Relaxation")
        for base_dir in bases:
            if not base_dir.is_dir():
                continue
            for cf in base_dir.iterdir():
                match = _CALC26_RE.fullmatch(cf.name)
                if not cf.is_dir() or not match:
                    continue
                execution_dir, attempt = cf, 0
                if version == "2622":
                    tries = [(int(m.group(1)), p) for p in cf.iterdir()
                             if p.is_dir() and (m := _TRY_RE.fullmatch(p.name))]
                    if tries:
                        attempt, execution_dir = max(tries)
                if _pending(execution_dir):
                    out.append(((1, int(match.group(1)), match.group(2), attempt),
                                execution_dir))

    out.sort(key=lambda item: item[0])
    return out
