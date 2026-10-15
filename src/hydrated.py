#!/usr/bin/env python3
"""
Hydrated (AD4 water-model) docking support for the GPCR pipeline.

This is a SECOND docking engine that lives alongside the default AutoDock Vina
path in run_screen.py. It implements the Forli-Olson hydrated docking protocol
(https://autodock-vina.readthedocs.io/en/latest/docking_hydrated.html):

  1. Explicit water ("W") atoms are attached to each ligand (Meeko hydrate).
  2. Receptor affinity grids are precomputed with autogrid4, then a water
     affinity map (W.map) is derived from the OA + HD maps (mapwater.py).
  3. Docking uses the AutoDock4 (AD4) scoring function -- the ONLY scoring
     function that understands the W atom type -- via Vina(sf_name='ad4') +
     v.load_maps(). This is why hydrated docking cannot reuse the Vina path's
     on-the-fly v.compute_vina_maps().
  4. After docking, each pose is post-processed with dry.py: waters overlapping
     the ligand/receptor are removed and the rest are scored against the W map,
     keeping STRONG/WEAK conserved waters and dropping DISPLC (displaced) ones.

External tools (see README "Hydrated docking"):
  - meeko          : Python API (MoleculePreparation), in the main environment.
  - autogrid4      : AutoDock4 grid generator. Found on PATH, or set the
                     AUTOGRID4_BIN env var (e.g. if it lives in a separate env;
                     it is distributed for osx-64/linux-64 and runs under
                     Rosetta on Apple Silicon).
  - prepare_receptor4.py : AutoDockTools / ADFR. Found next to the active
                     Python, on PATH, or via the PREPARE_RECEPTOR4 env var.
  - mapwater.py /
    dry.py         : vendored under ./autodock_scripts/ (run with this same
                     Python interpreter -- they only need numpy).

NOTE: AD4 binding energies are NOT on the same scale as Vina scores; results
from the two engines must not be compared directly (see README).
"""
import os
import shutil
import sys
import subprocess

from rdkit import Chem, RDLogger

from meeko import MoleculePreparation, PDBQTWriterLegacy

# Reuse the exact PROPKA protonation the standard prep uses, so both engines
# see an identically protonated receptor.
from preptarget import propka_protonate

HERE = os.path.dirname(os.path.abspath(__file__))
# The vendored AD4 helpers live in autodock_scripts/ at the repo root (a sibling of src/).
# After the src/ reorg this module sits one level down, so resolve from the root first and
# fall back to a copy next to this file if one is ever placed there.
AUTODOCK_SCRIPTS = next(
    (p for p in (os.path.join(os.path.dirname(HERE), "autodock_scripts"),
                 os.path.join(HERE, "autodock_scripts"))
     if os.path.isdir(p)),
    os.path.join(os.path.dirname(HERE), "autodock_scripts"))
MAPWATER_PY = os.path.join(AUTODOCK_SCRIPTS, "mapwater.py")
DRY_PY = os.path.join(AUTODOCK_SCRIPTS, "dry.py")

# autogrid4 binary: the AUTOGRID4_BIN env var wins (e.g. when it lives in a separate
# osx-64 env on Apple Silicon), otherwise whatever is on PATH.
AUTOGRID4_BIN = (os.environ.get("AUTOGRID4_BIN")
                 or shutil.which("autogrid4") or "autogrid4")

# prepare_receptor4.py (AutoDockTools) usually ships next to the active Python
# interpreter. It is far more tolerant of real multi-chain GPCR / cryo-EM
# complex structures (disulfides, inter-chain bonds) than Meeko's strict
# polymer builder, which raised "excess inter-residue bond" errors on them.
_PREP_REC4 = (os.environ.get("PREPARE_RECEPTOR4")
              or shutil.which("prepare_receptor4.py")
              or os.path.join(os.path.dirname(sys.executable), "prepare_receptor4.py"))


def _tool_exists(path: str) -> bool:
    """True if `path` is an existing file or resolves to an executable on PATH."""
    return os.path.isfile(path) or shutil.which(path) is not None

# Atom-type affinity maps autogrid4 computes for ligands. Covers Meeko's output
# for drug-like molecules; the water (W) map is derived from OA + HD afterwards
# by mapwater.py (autogrid4 cannot compute W directly).
LIGAND_MAP_TYPES = ["A", "C", "HD", "N", "NA", "OA", "SA",
                    "F", "P", "S", "Cl", "Br", "I"]
GRID_SPACING = 0.375  # Angstrom (AutoDock default)


class HydratedDockingError(RuntimeError):
    """Raised for any unrecoverable failure in the hydrated/AD4 setup."""


def _run(cmd: list, cwd: "str | None" = None, what: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess command, raising HydratedDockingError (with output tail) on non-zero exit."""
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise HydratedDockingError(
            f"{what or cmd[0]} failed (exit {proc.returncode}).\n"
            f"  cmd: {' '.join(str(c) for c in cmd)}\n"
            f"  output: {tail}"
        )
    return proc


def assert_tools() -> None:
    """Fail fast (before any docking) if the hydrated toolchain is incomplete."""
    missing = []
    if not _tool_exists(AUTOGRID4_BIN):
        missing.append(f"autogrid4 (looked at {AUTOGRID4_BIN}; put it on PATH or set AUTOGRID4_BIN)")
    if not _tool_exists(_PREP_REC4):
        missing.append(f"prepare_receptor4.py (looked at {_PREP_REC4}; set PREPARE_RECEPTOR4)")
    for s in (MAPWATER_PY, DRY_PY):
        if not os.path.isfile(s):
            missing.append(s)
    if missing:
        raise HydratedDockingError(
            "Hydrated docking is unavailable -- missing tools:\n  - "
            + "\n  - ".join(missing)
        )


# ==========================================================================
# RECEPTOR: precompute AD4 affinity maps (once per screen -- box is fixed)
# ==========================================================================
def _write_gpf(rec_name: str, receptor_pdbqt: str, center: tuple, box_size: tuple, gpf_path: str) -> None:
    """Write an autogrid4 grid-parameter file. Directive order is what autogrid4
    requires: the per-type `map` lines must come (in ligand_types order) before
    `elecmap`/`dsolvmap`."""
    rec_types = sorted({ln.split()[-1] for ln in open(receptor_pdbqt)
                        if ln.startswith(("ATOM", "HETATM"))})
    # Even grid-point counts spanning the box at the default spacing.
    npts = []
    for b in box_size:
        n = int(float(b) / GRID_SPACING)
        npts.append(n if n % 2 == 0 else n + 1)
    cx, cy, cz = center

    lines = [
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"gridfld {rec_name}.maps.fld",
        f"spacing {GRID_SPACING}",
        "receptor_types " + " ".join(rec_types),
        "ligand_types " + " ".join(LIGAND_MAP_TYPES),
        f"receptor {rec_name}.pdbqt",
        f"gridcenter {cx} {cy} {cz}",
        "smooth 0.5",
    ]
    lines += [f"map {rec_name}.{t}.map" for t in LIGAND_MAP_TYPES]
    lines += [f"elecmap {rec_name}.e.map",
              f"dsolvmap {rec_name}.d.map",
              "dielectric -0.1465"]
    with open(gpf_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_ad4_maps(receptor_pdb: str, rec_name: str, center: tuple, box_size: tuple,
                   maps_dir: str, ph: str = "7.4") -> tuple[str, str]:
    """Protonate the receptor and precompute autogrid4 maps incl. the W map.

    Returns (map_prefix, receptor_pdbqt):
      - map_prefix      : prefix passed to Vina's v.load_maps() (no extension)
      - receptor_pdbqt  : AD4 receptor PDBQT (used by dry.py for clash removal)
    """
    assert_tools()
    os.makedirs(maps_dir, exist_ok=True)

    pqr = os.path.join(maps_dir, f"{rec_name}_protonated.pqr")
    if not propka_protonate(receptor_pdb, pqr, ph):
        raise HydratedDockingError("PROPKA protonation (pdb2pqr) failed for AD4 maps.")

    # PQR -> protonated PDB for AutoDockTools' (lenient) receptor prep.
    prot_pdb = os.path.join(maps_dir, f"{rec_name}_protonated.pdb")
    _run(["obabel", pqr, "-O", prot_pdb], what="obabel pqr->pdb")

    prefix = os.path.join(maps_dir, rec_name)
    receptor_pdbqt = f"{prefix}.pdbqt"

    # Receptor PDBQT via AutoDockTools (tolerant of multi-chain complexes).
    _run([_PREP_REC4, "-r", prot_pdb, "-o", receptor_pdbqt],
         what="prepare_receptor4.py")
    if not os.path.isfile(receptor_pdbqt):
        raise HydratedDockingError(f"prepare_receptor4 did not write {receptor_pdbqt}.")

    # Write the autogrid GPF ourselves (deterministic, no fragile dependency).
    _write_gpf(rec_name, receptor_pdbqt, center, box_size,
               os.path.join(maps_dir, f"{rec_name}.gpf"))

    # autogrid4 computes the per-atom-type affinity maps (incl. OA and HD).
    _run([AUTOGRID4_BIN, "-p", f"{rec_name}.gpf", "-l", f"{rec_name}.glg"],
         cwd=maps_dir, what="autogrid4")

    # mapwater derives the W (water) affinity map from OA + HD. It resolves the
    # OA/HD map filenames relative to the CWD, so run it inside maps_dir.
    _run([sys.executable, MAPWATER_PY, "-r", f"{rec_name}.pdbqt",
          "-s", f"{rec_name}.W.map"], cwd=maps_dir, what="mapwater.py")

    w_map = f"{prefix}.W.map"
    if not os.path.isfile(w_map):
        raise HydratedDockingError(f"mapwater.py did not produce {w_map}.")

    return prefix, receptor_pdbqt


# ==========================================================================
# LIGAND: attach explicit waters and write an AD4 PDBQT
# ==========================================================================
def prepare_hydrated_ligand_pdbqt(rdkit_mol, out_path: str) -> int:
    """Write an AD4 PDBQT for `rdkit_mol` (3D, explicit H) with W atoms attached.

    Returns the number of water (W) atoms added. Raises HydratedDockingError if
    Meeko cannot write the molecule.
    """
    prep = MoleculePreparation(hydrate=True)
    setups = prep.prepare(rdkit_mol)
    if not setups:
        raise HydratedDockingError("Meeko returned no molecule setup for the ligand.")
    pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise HydratedDockingError(f"Meeko PDBQT writer failed: {err}")
    with open(out_path, "w") as f:
        f.write(pdbqt_str)
    return sum(1 for ln in pdbqt_str.splitlines()
               if (ln.startswith("ATOM") or ln.startswith("HETATM"))
               and ln.rstrip().endswith(" W"))


def _finite_gasteiger(mol) -> bool:
    """True if every Gasteiger charge is finite (Meeko rejects NaN charges)."""
    try:
        from rdkit.Chem import AllChem
        m = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(m)
        return all(a.GetDoubleProp("_GasteigerCharge") == a.GetDoubleProp("_GasteigerCharge")
                   for a in m.GetAtoms())
    except Exception:
        return False


_CCD_SMILES_CACHE = {}  # 3-letter ligand code -> canonical SMILES (or None if unavailable)


def _template_mol_from_ccd(ref_pdb: str):
    """Re-derive a reference ligand's bond orders from its PDB Chemical Component
    Dictionary template (canonical SMILES, looked up online by 3-letter code) and
    apply them to the coordinate-only ligand. This is the robust fix when
    OpenBabel's geometry-based perception leaves impossible valences (e.g. a
    5-valent carbon in fused N-rich heteroaromatics). Returns an RDKit Mol, or None
    if offline / unknown code / atom-set mismatch (caller keeps the OpenBabel mol)."""
    import json
    import urllib.request
    from rdkit.Chem import AllChem
    code = None
    for ln in open(ref_pdb, errors="ignore"):
        if ln.startswith(("HETATM", "ATOM")):
            code = ln[17:20].strip()
            break
    if not code:
        return None
    # Fetch the canonical SMILES once per 3-letter code (cached across structures and
    # resumes); None caches a known-bad lookup so we don't re-hit the network.
    if code not in _CCD_SMILES_CACHE:
        smi = None
        try:
            d = json.load(urllib.request.urlopen(
                f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=20))
            descs = d.get("pdbx_chem_comp_descriptor", []) if isinstance(d, dict) else []
            cands = ([x["descriptor"] for x in descs if x.get("type") == "SMILES_CANONICAL"]
                     or [x["descriptor"] for x in descs if x.get("type") == "SMILES"])
            smi = cands[0] if cands else None
        except Exception:
            smi = None  # offline / unknown code / unexpected JSON -> graceful no-op
        _CCD_SMILES_CACHE[code] = smi
    smi = _CCD_SMILES_CACHE[code]
    if not smi:
        return None
    template = Chem.MolFromSmiles(smi)
    lig = Chem.MolFromPDBFile(ref_pdb, removeHs=False, sanitize=False)
    if template is None or lig is None:
        return None
    frags = Chem.GetMolFrags(lig, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        lig = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, lig)
        return Chem.AddHs(fixed, addCoords=True)
    except Exception:
        return None  # template did not match the extracted atoms


def reference_mol_for_hydration(ref_pdb: str, ph, workdir: str):
    """Build a sanitized, H-added RDKit mol from a bare co-crystal ligand PDB.

    Co-crystal ligand PDBs carry coordinates but no bond orders, so OpenBabel
    perceives bonds and adds protons; RDKit then parses the result. Real ligands
    vary widely (odd valences, multiple co-crystal copies, alt-locs), so this
    tries progressively more lenient parses and keeps the largest fragment.
    Returns an RDKit Mol; raises HydratedDockingError with a specific reason if
    every strategy fails (so the UI can surface why, e.g. fall back to Vina).
    """
    sdf = os.path.join(workdir, "reference_for_meeko.sdf")
    ob = subprocess.run(["obabel", ref_pdb, "-O", sdf, "-p", str(ph)],
                        capture_output=True, text=True)
    if not os.path.isfile(sdf) or os.path.getsize(sdf) == 0:
        raise HydratedDockingError(
            "OpenBabel could not convert the reference ligand to SDF.\n"
            f"  obabel stderr: {(ob.stderr or '')[-800:]}")

    # OpenBabel/PDB bond perception routinely trips RDKit's strict valence check
    # (e.g. a protonated amine read as a neutral 4-valent N). Those expected
    # errors are handled below, so silence RDKit's logger during parsing.
    RDLogger.DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromMolFile(sdf, removeHs=False)  # strict (sanitized)
        if mol is None:                                  # lenient parse + repair
            mol = Chem.MolFromMolFile(sdf, removeHs=False, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                # Repair protonated groups that perception leaves over-valent AND
                # neutral (protonated amine N with 4 bonds, oxocarbenium O with 3):
                # assign the missing +1 so RDKit sanitizes and Gasteiger charges stay
                # finite -- Meeko's PDBQT writer rejects NaN charges otherwise.
                for at in mol.GetAtoms():
                    n, v, fc = at.GetAtomicNum(), at.GetExplicitValence(), at.GetFormalCharge()
                    if fc == 0 and ((n == 7 and v == 4) or (n == 8 and v == 3)):
                        at.SetFormalCharge(1)
                mol.UpdatePropertyCache(strict=False)
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    # Last resort: skip kekulization so we still return a mol;
                    # the caller surfaces any downstream charge/writer failure.
                    try:
                        Chem.SanitizeMol(
                            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except Exception:
                        pass
    finally:
        RDLogger.EnableLog("rdApp.error")

    if mol is None or mol.GetNumAtoms() == 0:
        raise HydratedDockingError(
            "RDKit could not parse the reference ligand after OpenBabel bond "
            "perception. The co-crystal ligand may have unusual chemistry. Use the "
            "standard Vina engine for validation of this structure.")

    # A co-crystal "ligand" extracted by 3-letter code can include several copies
    # (different chains/alt-locs); hydrate only the largest connected fragment.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())

    # OpenBabel geometry-based perception can leave impossible valences (e.g. a
    # 5-valent carbon in fused N-rich heteroaromatics such as ZM241385) that no
    # charge repair can mend; these produce NaN Gasteiger charges that Meeko
    # rejects. Only when the offline parse is broken, re-derive bond orders from
    # the ligand's PDB Chemical Component template (network touched in this case).
    if not _finite_gasteiger(mol):
        templated = _template_mol_from_ccd(ref_pdb)
        if templated is not None and _finite_gasteiger(templated):
            mol = templated
        else:
            # Neither OpenBabel nor the CCD template gave finite charges; Meeko would
            # reject this ligand with an opaque "nan charge" error mid-batch. Fail
            # loudly and legibly instead so the caller skips it / falls back to Vina.
            raise HydratedDockingError(
                "The reference ligand still has non-finite partial charges after "
                "template-based bond perception (unusual chemistry — e.g. covalent or "
                "metal-coordinated). Use the standard Vina engine for this structure.")
    return mol


# ==========================================================================
# POSE POST-PROCESSING: strip displaced waters, keep conserved ones
# ==========================================================================
def is_water_pdbqt_line(line: str) -> bool:
    """True for an AD4 PDBQT atom line whose atom type is W (explicit water)."""
    return ((line.startswith("ATOM") or line.startswith("HETATM"))
            and line.rstrip().endswith(" W"))


def dry_pose(docked_pdbqt: str, rank: int, receptor_pdbqt: str, w_map: str, workdir: str) -> "str | None":
    """Run dry.py on a single pose (`rank`, 1-based) of a multi-model Vina output.

    dry.py ignores the water map on multi-pose inputs, so we use its native
    `-p RANK` pose extraction. Returns the path to the dried+scored single-pose
    PDBQT (ligand + conserved waters only), or None if dry.py fails -- callers
    must fall back to the raw pose.
    """
    base = os.path.splitext(os.path.basename(docked_pdbqt))[0]
    out = os.path.join(workdir, f"{base}_pose_{rank}_DRY_SCORED.pdbqt")
    try:
        _run([sys.executable, DRY_PY, "-q",
              "-i", os.path.abspath(docked_pdbqt),
              "-r", os.path.abspath(receptor_pdbqt),
              "-m", os.path.abspath(w_map),
              "-p", rank],
             cwd=workdir, what="dry.py")
    except HydratedDockingError:
        return None
    return out if os.path.isfile(out) else None


def split_ligand_and_waters(pdbqt_path: str) -> tuple[list[str], list[str]]:
    """Split a (possibly dried) pose PDBQT into ligand atom lines and water
    HETATM/HOH PDB records.

    Returns (ligand_pdbqt_lines, water_pdb_lines). Water W atoms are re-emitted
    as standard HOH oxygens so PLIP can detect water-bridged interactions; the
    ligand lines are returned untouched for the normal obabel -> PDB conversion.
    """
    ligand_lines, water_pdb = [], []
    widx = 0
    with open(pdbqt_path) as f:
        for line in f:
            if is_water_pdbqt_line(line):
                widx += 1
                x, y, z = line[30:38], line[38:46], line[46:54]
                water_pdb.append(
                    f"HETATM{widx:>5} O    HOH W{widx:>4}    "
                    f"{x:>8}{y:>8}{z:>8}  1.00  0.00           O\n"
                )
            elif line.startswith(("ATOM", "HETATM")):
                ligand_lines.append(line)
    return ligand_lines, water_pdb
