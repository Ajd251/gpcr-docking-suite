"""Canonical filesystem layout for the docking project — the single source of truth.

Every script derives its input/output locations from here instead of hard-coding
relative paths, so the tree can be reorganised in one place. Paths are computed from
this file's location (``src/paths.py`` → repo root is its parent's parent), which keeps
them correct no matter what the current working directory is.

Layout
------
    <ROOT>/
      src/          this package (all Python modules + drivers)
      data/
        structures/ uploaded / working receptor PDBs
        libraries/  SMILES libraries for screening (JSON: {name: SMILES})
      results/
        validation/ crystallographic-redocking sweeps (batch validation)
        screens/    prospective screening docking-results directories
        mmpbsa/      end-point rescoring runs (poses + rescored energies)
      scratch/      transient run scratch (git-ignored; safe to delete)
"""
from __future__ import annotations

import os

# ``src/paths.py`` -> src -> repo root
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

DATA = os.path.join(ROOT, "data")
STRUCTURES = os.path.join(DATA, "structures")
LIBRARIES = os.path.join(DATA, "libraries")

RESULTS = os.path.join(ROOT, "results")
VALIDATION = os.path.join(RESULTS, "validation")
SCREENS = os.path.join(RESULTS, "screens")
MMPBSA = os.path.join(RESULTS, "mmpbsa")

SCRATCH = os.path.join(ROOT, "scratch")


def rel(path: str) -> str:
    """Repo-root-relative form of an absolute path (stable across directory moves)."""
    try:
        return os.path.relpath(os.path.abspath(path), ROOT)
    except ValueError:
        return path


def resolve(path: str) -> str:
    """Resolve a stored path: absolute paths pass through; relative ones join ROOT."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

_ALL = [DATA, STRUCTURES, LIBRARIES, RESULTS, VALIDATION, SCREENS,
        MMPBSA, SCRATCH]


def ensure() -> None:
    """Create every standard directory if missing (idempotent)."""
    for d in _ALL:
        os.makedirs(d, exist_ok=True)


def find_structure_pdb(name: str) -> "str | None":
    """Locate a receptor source PDB by structure name across the standard locations.

    Searches data/structures first, then anywhere under results/. Excludes derived
    files (docked poses, complexes, reference-ligand extracts).
    """
    import glob
    cand = os.path.join(STRUCTURES, f"{name}.pdb")
    if os.path.isfile(cand):
        return cand
    for hit in glob.glob(os.path.join(RESULTS, "**", f"{name}*.pdb"), recursive=True):
        if not any(t in hit for t in ("_vina_", "_hydrated_", "_complex", "reference_ligand")):
            return hit
    return None


def scratch_dir(*parts: str) -> str:
    """Return (and create) a subdirectory under scratch/ for transient run files."""
    d = os.path.join(SCRATCH, *parts)
    os.makedirs(d, exist_ok=True)
    return d
