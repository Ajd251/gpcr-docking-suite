#!/usr/bin/env python3
"""
Headless batch redocking-validation core for the GPCR docking pipeline.

Redocks a receptor's native co-crystallised ligand and measures how faithfully
the protocol reproduces the crystallographic pose, as a function of search
exhaustiveness and random seed. Drives the "Batch Validation" tab in app.py.

Two RMSD metrics are reported per docking run (the sampling-vs-scoring
distinction):
  - rank1_rmsd : RMSD of the top-SCORED pose      -> scoring power
  - best_rmsd  : lowest RMSD across all N poses    -> sampling power
Success threshold: RMSD <= 2.0 A (the community convention for a correct redock).

Performance: a receptor's affinity maps are computed ONCE per (structure, engine)
and written to disk, then reused by a fresh seeded Vina object for every
(seed, exhaustiveness) combination. This keeps each run independent and exactly
reproducible while paying the expensive map-generation cost only once.
"""
import csv
import os
import shutil
import subprocess
import threading

import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS, Descriptors
from vina import Vina

import preptarget
import hydrated

SUCCESS_RMSD = 2.0  # Angstrom

# Resnames that are never the ligand of interest (water, common ions/buffers).
_NON_LIGANDS = {"HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "SO4",
                "PO4", "GOL", "EDO", "ACT", "CL", "BR", "IOD", "MN", "FE", "NI",
                "CU", "CO", "CD", "HG"}


# ==========================================================================
# Receptor charge sanitisation (shared with app.py Tab 1 / Tab 4)
# ==========================================================================
def fix_explosive_charges(pdbqt_filepath: str) -> None:
    """Reset divergent receptor Gasteiger charges to +0.000 and re-pad columns.

    OpenBabel's Gasteiger charges diverge on un-relaxed homology models, producing
    huge values that overflow Vina's fixed-width PDBQT format. The Vina forcefield
    ignores receptor partial charges, so they are zeroed and the columns rebuilt to
    the strict PDBQT specification.

    This is a deliberate workaround, NOT a bug — do not "fix" the zeroed charges.
    Re-exported by app.py (``fix_explosive_charges = validation.fix_explosive_charges``);
    the name and signature must stay stable. Rewrites the file in place; returns None.
    """
    if not os.path.exists(pdbqt_filepath):
        return
    with open(pdbqt_filepath, "r") as f:
        lines = f.readlines()
    with open(pdbqt_filepath, "w") as f:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # The element/atom-type token is always the last whitespace field.
                atom_type = line.strip().split()[-1]
                # Columns 1-66 hold the record up to and including the coordinates;
                # pad to exactly 66 so the appended charge/type land in the fixed
                # PDBQT columns regardless of the original line's ragged width.
                base_coords = line[:66].ljust(66)
                # Force every partial charge to +0.000 (Vina ignores receptor charges)
                # and left-justify the 2-char atom type into the trailing field.
                f.write(f"{base_coords}    +0.000 {atom_type:<2}\n")
            else:
                f.write(line)


# ==========================================================================
# Native-ligand discovery and extraction
# ==========================================================================
def candidate_ligand_codes(pdb_text: str) -> list:
    """Return [(resname, n_atoms), ...] for plausible co-crystal ligands, by size.

    Filters out water, common ions and buffers; the largest remaining HETATM
    group is almost always the ligand of interest (good default for the UI).

    Parameters
    ----------
    pdb_text : full PDB text of the structure.

    Returns a list of (resname, atom_count) tuples sorted by descending count.
    """
    counts: dict = {}
    for line in pdb_text.split("\n"):
        if line.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            if resname and resname not in _NON_LIGANDS:
                counts[resname] = counts.get(resname, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def extract_native_ligand(pdb_text: str, ligand_code: str) -> "tuple[str | None, list | None]":
    """Isolate a co-crystal ligand by 3-letter code; return (pdb_block, centroid).

    The centroid becomes the docking box centre for the redock. Returns
    ``(None, None)`` if no HETATM records match ``ligand_code``.
    """
    lig, x, y, z = [], [], [], []
    code = ligand_code.upper()
    for line in pdb_text.split("\n"):
        # Match only HETATM records whose 3-letter residue name (cols 18-20) equals code.
        if line.startswith("HETATM") and line[17:20].strip().upper() == code:
            lig.append(line + "\n")
            try:
                # Fixed-column PDB coordinate fields: x=31-38, y=39-46, z=47-54.
                x.append(float(line[30:38])); y.append(float(line[38:46])); z.append(float(line[46:54]))
            except ValueError:
                pass
    if not lig:
        return None, None
    centroid = [sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)]
    # Terminate the extracted block with END so downstream parsers accept it.
    return "".join(lig) + "END\n", centroid


def ligand_descriptors(ref_pdb: str) -> "tuple[int, float]":
    """(heavy_atoms, rotatable_bonds) for the reference ligand; NaN if unparsable.

    Heavy-atom count and rotatable-bond count are the two difficulty/flexibility
    axes used to stratify the validation results. Heavy atoms are counted directly
    from the PDB (robust); rotatable bonds require an RDKit parse and fall back to
    NaN when the ligand cannot be sanitised.
    """
    # Heavy atom = any ATOM/HETATM whose element column (77-78) is not "H".
    heavy = sum(1 for ln in open(ref_pdb)
                if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() != "H")
    RDLogger.DisableLog("rdApp.error")  # silence expected valence warnings on raw ligand PDBs
    mol = Chem.MolFromPDBFile(ref_pdb, removeHs=True)
    RDLogger.EnableLog("rdApp.error")
    rot = Descriptors.NumRotatableBonds(mol) if mol else float("nan")
    return heavy, rot


# ==========================================================================
# RMSD (MCS / automorphism-corrected)
# ==========================================================================
def _robust_mol_from_pdb(path: str) -> "Chem.Mol | None":
    """Parse a ligand(-pose) PDB into an RDKit mol for RMSD, tolerant of the odd
    valences that geometry-based PDB bond perception produces (e.g. a docked
    trimethylammonium N⁺ read as an over-valent neutral N, which fails strict
    sanitization and would otherwise drop the whole pose's RMSD to NaN). Strict
    sanitize first; on failure, a lenient parse + protonated-group repair + ring
    perception, so MCS still works. Returns an RDKit Mol or None.

    Parameters
    ----------
    path : path to a single-molecule ligand or pose PDB.
    """
    # Fast path: strict parse succeeds for well-behaved ligands.
    m = Chem.MolFromPDBFile(path, removeHs=True)
    if m is not None:
        return m
    # Fallback path: re-parse WITHOUT sanitization so an over-valent atom does not
    # abort the whole parse (returns None otherwise).
    m = Chem.MolFromPDBFile(path, removeHs=True, sanitize=False)
    if m is None:
        return None
    try:
        m.UpdatePropertyCache(strict=False)
        for at in m.GetAtoms():
            n, v, fc = at.GetAtomicNum(), at.GetExplicitValence(), at.GetFormalCharge()
            # Repair the classic geometry-perception artefact: a protonated amine
            # N(4) or a protonated/over-bonded O(3) read as neutral. Assigning the
            # missing +1 restores a legal valence so sanitization can proceed.
            if fc == 0 and ((n == 7 and v == 4) or (n == 8 and v == 3)):
                at.SetFormalCharge(1)
        # Sanitize everything EXCEPT kekulization and property re-perception, which
        # can still choke on borderline aromatic perception from raw coordinates.
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                         ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass
    try:
        Chem.FastFindRings(m)  # MCS needs ring info even on a lenient parse
    except Exception:
        pass
    return m


def _mcs_rmsd(ref_mol: "Chem.Mol | None", dock_mol: "Chem.Mol | None") -> "float | None":
    """Automorphism-corrected RMSD over the maximum common substructure. The MCS is
    matched onto BOTH molecules with `uniquify=False`, and the *minimum* RMSD over
    all symmetry-equivalent atom mappings is returned — so a symmetric group (e.g. a
    2,6-disubstituted phenyl, or a flipped ring) is not penalised when the pose is in
    fact superimposed. (The previous version took only the first match, which
    inflated the RMSD of otherwise-correct symmetric poses.)

    Returns the RMSD in Angstrom, or None when either molecule is missing, lacks 3D
    coordinates, or has no usable (>=3-atom) common substructure.
    """
    if ref_mol is None or dock_mol is None:
        return None
    if ref_mol.GetNumConformers() == 0 or dock_mol.GetNumConformers() == 0:
        return None  # a lenient parse can yield a mol with no 3D coordinates
    try:
        res = rdFMCS.FindMCS([ref_mol, dock_mol], completeRingsOnly=True, timeout=20)
        if not res.smartsString:
            return None
        patt = Chem.MolFromSmarts(res.smartsString)
        a = ref_mol.GetSubstructMatch(patt)  # reference labelling (first match)
        # A meaningful 3D superposition needs >= 3 matched atoms (three non-collinear
        # points fix a rigid body). This is a geometric requirement, not a per-ligand
        # threshold, so it generalises to any structure; it rejects only a truly
        # degenerate 1-2-atom match (which could otherwise report a spurious low RMSD).
        if not a or len(a) < 3:
            return None
        # All symmetry-equivalent mappings of the MCS onto the docked pose; the count is
        # bounded so a pathologically symmetric ligand cannot stall a long batch.
        dock_matches = dock_mol.GetSubstructMatches(patt, uniquify=False, maxMatches=5000)
        if not dock_matches:
            return None
        rc, dc = ref_mol.GetConformer(), dock_mol.GetConformer()
        rp = [rc.GetAtomPosition(i) for i in a]
        best = None
        for b in dock_matches:
            s = sum((rp[k].x - dc.GetAtomPosition(j).x) ** 2 +
                    (rp[k].y - dc.GetAtomPosition(j).y) ** 2 +
                    (rp[k].z - dc.GetAtomPosition(j).z) ** 2
                    for k, j in enumerate(b))
            r = (s / len(b)) ** 0.5
            best = r if best is None else min(best, r)
        return best
    except Exception:
        return None


def split_pose_models(docked_pdbqt: str, mode: str, workdir: str) -> list:
    """Return the docked poses as a list of per-pose PDB blocks (one obabel call).

    Hydrated poses carry explicit W (water) atoms; these are stripped first so the
    ligand-only geometry is compared / fingerprinted. Shared by the RMSD and the
    interaction-fingerprint steps.

    Parameters
    ----------
    docked_pdbqt : multi-MODEL PDBQT written by Vina.
    mode : "hydrated" (strip W atoms) or anything else (Vina; no stripping).
    workdir : scratch dir for the intermediate dewatered/converted files.

    Returns a list of PDB text blocks, one per pose.
    """
    src = docked_pdbqt
    if mode == "hydrated":
        src = os.path.join(workdir, "_dewater.pdbqt")
        with open(docked_pdbqt) as fi, open(src, "w") as fo:
            for ln in fi:
                if not hydrated.is_water_pdbqt_line(ln):
                    fo.write(ln)
    poses_pdb = os.path.join(workdir, "_poses.pdb")
    subprocess.run(["obabel", src, "-O", poses_pdb], capture_output=True)

    # Split the multi-MODEL PDB into per-pose atom blocks; drop MASTER/END trailers.
    models, cur = [], []
    for line in open(poses_pdb):
        if line.startswith("MODEL"):
            cur = []                      # new pose begins
        elif line.startswith("ENDMDL"):
            models.append("".join(cur))   # pose complete
        elif not line.startswith(("MASTER", "END")):
            cur.append(line)
    # Single-pose PDBs may lack MODEL/ENDMDL wrappers; capture the trailing block.
    if not models and cur:
        models.append("".join(cur))
    return models


def pose_rmsds(ref_pdb: str, models: list, workdir: str) -> list:
    """MCS-corrected RMSD of every docked pose (PDB blocks) vs the crystal ligand.

    Parameters
    ----------
    ref_pdb : path to the crystal reference-ligand PDB.
    models : per-pose PDB text blocks (from ``split_pose_models``).
    workdir : scratch dir for the transient per-pose files.

    Returns a list of RMSD floats (or None per pose that fails to parse/match),
    index-aligned with ``models``.
    """
    RDLogger.DisableLog("rdApp.error")  # suppress per-pose valence warnings in the loop
    ref_mol = _robust_mol_from_pdb(ref_pdb)
    rmsds = []
    for i, m in enumerate(models):
        # RDKit's PDB reader needs a file; write each pose out, parse, then clean up.
        f = os.path.join(workdir, f"_pose{i}.pdb")
        with open(f, "w") as out:
            out.write(m)
        rmsds.append(_mcs_rmsd(ref_mol, _robust_mol_from_pdb(f)))
        os.remove(f)
    RDLogger.EnableLog("rdApp.error")
    return rmsds


# ==========================================================================
# Interaction fingerprint (residue-contact / SIFt-style) pose selection
# ==========================================================================
def _heavy_xyz(pdb_block: str) -> np.ndarray:
    """Heavy-atom (non-H) coordinates from a PDB block as an (N,3) array.

    Hydrogens are excluded so the contact fingerprint depends only on the heavy-atom
    scaffold (H placement is protonation/parse-dependent and would add noise).
    Returns an empty (0,3) array when nothing parses.
    """
    xyz = []
    for ln in pdb_block.splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            elem = ln[76:78].strip()   # element column (preferred)
            name = ln[12:16].strip()   # atom name (fallback when element blank)
            # Skip hydrogens: element == "H", or a blank-element atom whose name starts with H.
            if elem == "H" or (not elem and name[:1] == "H"):
                continue
            try:
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                pass
    return np.array(xyz) if xyz else np.zeros((0, 3))


def receptor_pocket_atoms(receptor_pdbqt: str, center, radius: float = 15.0) -> "tuple[list, np.ndarray]":
    """(residue-id list, xyz array) for receptor atoms within `radius` of `center`.

    Restricting to the pocket keeps the contact search fast even for active-state
    receptor-G-protein complexes (thousands of distal atoms).

    Parameters
    ----------
    receptor_pdbqt : path to the prepared receptor PDBQT.
    center : (x, y, z) box centre (the crystal-ligand centroid).
    radius : keep only atoms within this many Angstrom of ``center``.

    Returns (rids, xyz) index-aligned: rids are "chain:resnum" tags, xyz an (N,3) array.
    """
    rids, xyz = [], []
    for ln in open(receptor_pdbqt):
        if ln.startswith(("ATOM", "HETATM")):
            try:
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
            # Per-atom residue id "chain:resnum" (blank chain -> "_") lets the
            # fingerprint collapse to per-residue contacts regardless of atom count.
            chain = ln[21].strip() or "_"
            rids.append(f"{chain}:{ln[22:26].strip()}")
    xyz = np.array(xyz) if xyz else np.zeros((0, 3))
    if len(xyz):
        # Vectorised radius filter around the pocket centre.
        keep = np.linalg.norm(xyz - np.asarray(center, float), axis=1) <= radius
        rids = [r for r, k in zip(rids, keep) if k]
        xyz = xyz[keep]
    return rids, xyz


def contact_fingerprint(rec_rids: list, rec_xyz: np.ndarray, lig_xyz: np.ndarray,
                        cutoff: float = 4.0) -> frozenset:
    """Frozenset of receptor residue ids with any atom within `cutoff` of a ligand atom.

    This is a SIFt-style (Structural Interaction Fingerprint) set: a residue is "in
    contact" if its nearest atom-atom distance to the ligand is <= cutoff. The
    resulting set is compared across poses by Tanimoto to pick the pose whose contact
    pattern best matches the crystal ligand.
    """
    if len(rec_xyz) == 0 or len(lig_xyz) == 0:
        return frozenset()
    # cdist -> full receptor×ligand distance matrix; min over ligand atoms gives each
    # receptor atom's closest approach, then threshold at the contact cutoff.
    near = cdist(rec_xyz, lig_xyz).min(axis=1) <= cutoff
    return frozenset(r for r, n in zip(rec_rids, near) if n)


def _tanimoto(a: "frozenset | set", b: "frozenset | set") -> float:
    """Tanimoto (Jaccard) similarity of two contact-fingerprint sets: |A∩B| / |A∪B|.

    Returns 0.0 when either set is empty. Used to rank docked poses by how well their
    receptor-contact set reproduces the crystal ligand's.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    # |A ∪ B| == |A| + |B| - |A ∩ B|
    return inter / (len(a) + len(b) - inter)


# ==========================================================================
# Receptor & ligand preparation (once per structure+engine)
# ==========================================================================
def prepare_receptor_maps(receptor_pdb: str, rec_name: str, center, box, workdir: str,
                          mode: str, ph: str = "7.4") -> "tuple[str, str]":
    """Prepare the receptor + reusable affinity maps; return (map_prefix, receptor_pdbqt).

    Built ONCE per (structure, engine); every (seed, exhaustiveness) run reloads these
    maps so runs are independent and exactly reproducible. The returned receptor PDBQT
    lets ``redock_once`` build the interaction fingerprint.

    Parameters
    ----------
    receptor_pdb : cleaned receptor PDB.
    rec_name : short receptor name used for output filenames.
    center / box : docking box centre and (x, y, z) edge lengths in Angstrom.
    workdir : per-(structure, engine) output dir.
    mode : "hydrated" (AD4 maps) or Vina (default).
    ph : protonation pH for PROPKA.
    """
    os.makedirs(workdir, exist_ok=True)
    if mode == "hydrated":
        prefix, receptor_pdbqt = hydrated.build_ad4_maps(
            receptor_pdb, rec_name, center, box,
            os.path.join(workdir, "ad4_maps"), ph)
        return prefix, receptor_pdbqt

    # Standard Vina: PROPKA -> rigid PDBQT (Gasteiger) -> sanitise -> precompute maps.
    pqr = os.path.join(workdir, f"{rec_name}_protonated.pqr")
    if not preptarget.propka_protonate(receptor_pdb, pqr, ph):
        raise RuntimeError(f"PROPKA protonation failed for {rec_name}.")
    pdbqt = os.path.join(workdir, f"{rec_name}.pdbqt")
    subprocess.run(["obabel", pqr, "-O", pdbqt, "-xr", "--partialcharge", "gasteiger"],
                   capture_output=True)
    fix_explosive_charges(pdbqt)

    prefix = os.path.join(workdir, f"{rec_name}_vina")
    v = Vina(sf_name="vina")
    v.set_receptor(pdbqt)
    # write_maps requires an even voxel count per axis; force it so any box edge
    # works (e.g. 22 A / 0.375 spacing rounds to an odd 59 otherwise).
    v.compute_vina_maps(center=list(center), box_size=list(box), force_even_voxels=True)
    v.write_maps(map_prefix_filename=prefix, overwrite=True)
    return prefix, pdbqt


def prepare_reference_ligand(ref_pdb: str, out_pdbqt: str, mode: str, ph, workdir: str) -> str:
    """Prepare the native ligand as a PDBQT (hydrated for AD4, Gasteiger for Vina).

    Hydrated mode attaches explicit W (water) atoms via Meeko; Vina mode uses an
    OpenBabel Gasteiger conversion. Returns ``out_pdbqt`` (the written path).
    """
    if mode == "hydrated":
        mol = hydrated.reference_mol_for_hydration(ref_pdb, ph, workdir)
        hydrated.prepare_hydrated_ligand_pdbqt(mol, out_pdbqt)
    else:
        subprocess.run(["obabel", ref_pdb, "-O", out_pdbqt, "-p", str(ph),
                        "--partialcharge", "gasteiger"], capture_output=True)
    return out_pdbqt


# ==========================================================================
# A single redock measurement
# ==========================================================================
def _pose_as_hetatm(block: str) -> list:
    """A pose PDB block as HETATM lines (the HETATM rule) for complex assembly."""
    return ["HETATM" + l[6:] if l.startswith("ATOM") else l
            for l in block.splitlines(True) if l.startswith(("ATOM", "HETATM"))]


def rescore_poses(models: list, rmsds: list, ctx: dict, workdir: str) -> dict:
    """End-point rescoring of a pose set -> MMPBSA_* / MMGBSA_* selection columns.

    Each pose is rebuilt as receptor+ligand, minimised and scored; the best-scoring
    pose is selected and its RMSD reported, exactly mirroring how the
    interaction-fingerprint axis selects a pose. Scoring is parallel across poses
    (each Amber child is single-threaded, so this is the real speed-up).

    `ctx` keys: receptor_pdb, smiles, cache_dir, indi, maxcyc, n_jobs, also_gb, ph.
    Never raises -- unscoreable poses come back NaN and simply cannot be selected.
    """
    import mmpbsa
    from concurrent.futures import ThreadPoolExecutor

    out = {"MMPBSA_RMSD": float("nan"), "MMPBSA_Rank": 0, "MMPBSA_dG": float("nan"),
           "MMGBSA_RMSD": float("nan"), "MMGBSA_Rank": 0, "MMGBSA_dG": float("nan")}
    if not models:
        return out
    rec = [l for l in open(ctx["receptor_pdb"], errors="ignore") if l.startswith("ATOM")]
    smiles, cache = ctx["smiles"], ctx.get("cache_dir")
    indi, maxcyc = float(ctx.get("indi", 2.0)), int(ctx.get("maxcyc", 250))
    ph = ctx.get("ph", 7.4)
    also_gb = bool(ctx.get("also_gb", True))
    methods = ("pb", "gb") if also_gb else ("pb",)
    base = os.path.join(workdir, "mmpbsa")
    os.makedirs(base, exist_ok=True)

    # Warm the AM1-BCC cache once at the correct protonation state. Cold-cache parallel
    # workers would each derive the same charges independently (~20 s each) first.
    try:
        mol0 = mmpbsa.ligand_mol_from_pose(_pose_as_hetatm(models[0]), smiles, ph=ph)
        mmpbsa.prepare_ligand_params(mol0, os.path.join(base, "_warm"), cache_dir=cache)
    except Exception:
        pass

    def _one(idx_model):
        i, block = idx_model
        # rescore_pose minimises once, scores GB+PB on the same geometry, protonates the
        # ligand to `ph`, and DELETES its scratch when done (no per-pose Amber-file bloat).
        r = mmpbsa.rescore_pose(rec + _pose_as_hetatm(block), smiles,
                                os.path.join(base, f"pose{i + 1}"),
                                ph=ph, indi=indi, maxcyc=maxcyc, cache_dir=cache,
                                methods=methods)
        return i, {"gb": r.get("gb_dG", float("nan")), "pb": r.get("pb_dG", float("nan"))}

    n_jobs = max(1, int(ctx.get("n_jobs", 8)))
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            scored = dict(ex.map(_one, list(enumerate(models))))
    except Exception:
        return out

    for key, prefix in (("pb", "MMPBSA"), ("gb", "MMGBSA")):
        if key == "gb" and not also_gb:
            continue
        cand = [(i, v[key]) for i, v in scored.items() if v[key] == v[key]]
        if not cand:
            continue
        bi, bdg = min(cand, key=lambda t: t[1])
        out[f"{prefix}_dG"] = round(float(bdg), 2)
        out[f"{prefix}_Rank"] = bi + 1
        if bi < len(rmsds) and rmsds[bi] is not None:
            out[f"{prefix}_RMSD"] = round(float(rmsds[bi]), 2)
    return out


def redock_once(map_prefix: str, ref_ligand_pdbqt: str, ref_pdb: str, mode: str, seed: int,
                exhaustiveness: int, n_poses: int, workdir: str, cpu: int = 8,
                receptor_pdbqt: "str | None" = None,
                keep_poses_as: "str | None" = None,
                mmpbsa_ctx: "dict | None" = None) -> dict:
    """Dock the reference ligand at (seed, exhaustiveness); return a result dict.

    Maps are loaded from disk (prepared once) so every call is independent and
    reproducible. Returns three pose-quality axes -- rank-1 (scoring), best-pose
    (sampling) and interaction-selected (the pose whose receptor-contact fingerprint
    best matches the crystal ligand's) -- plus energies.

    Parameters
    ----------
    map_prefix : affinity-map prefix from ``prepare_receptor_maps``.
    ref_ligand_pdbqt : prepared native-ligand PDBQT to dock.
    ref_pdb : crystal reference-ligand PDB (RMSD/fingerprint target).
    mode : "hydrated" -> AD4 scoring, else Vina.
    seed / exhaustiveness / n_poses : Vina search controls (seed makes runs reproducible).
    workdir : scratch dir. cpu : threads. receptor_pdbqt : enables fingerprint selection.

    Returns a dict with the RMSD/rank/energy fields (see BATCH_COLUMNS).
    """
    sf = "ad4" if mode == "hydrated" else "vina"
    v = Vina(sf_name=sf, cpu=cpu, seed=seed)
    v.load_maps(map_prefix)
    v.set_ligand_from_file(ref_ligand_pdbqt)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    docked = os.path.join(workdir, "_docked.pdbqt")
    v.write_poses(docked, n_poses=n_poses, overwrite=True)
    energies = v.energies(n_poses=n_poses)

    models = split_pose_models(docked, mode, workdir)
    rmsds = pose_rmsds(ref_pdb, models, workdir)
    valid = [(i, r) for i, r in enumerate(rmsds) if r is not None]
    # Rank-1 = top-SCORED pose (Vina orders poses best-score-first) -> scoring power.
    rank1 = rmsds[0] if rmsds and rmsds[0] is not None else float("nan")
    if valid:
        # Best-pose = LOWEST RMSD across all poses -> sampling power (did the search
        # generate a near-native pose at all, regardless of how it was scored?).
        best_idx, best = min(valid, key=lambda t: t[1])
        best_rank = best_idx + 1
        mean_rmsd = float(np.mean([r for _, r in valid]))
    else:
        best, best_rank, mean_rmsd = float("nan"), 0, float("nan")
    all_e = [float(e[0]) for e in energies]
    top_energy = round(all_e[0], 2) if all_e else float("nan")
    mean_energy = round(float(np.mean(all_e)), 2) if all_e else float("nan")

    # Interaction-fingerprint pose selection (defaults to rank-1 if unavailable).
    sel_rmsd, sel_rank, sel_sim = rank1, 1, float("nan")
    if receptor_pdbqt and os.path.isfile(receptor_pdbqt) and models:
        ref_lig = _heavy_xyz(open(ref_pdb).read())
        if len(ref_lig):
            rids, rxyz = receptor_pocket_atoms(receptor_pdbqt, ref_lig.mean(axis=0))
            ref_fp = contact_fingerprint(rids, rxyz, ref_lig)
            if ref_fp:
                sims = [_tanimoto(ref_fp, contact_fingerprint(rids, rxyz, _heavy_xyz(m)))
                        for m in models]
                # highest fingerprint match; ties broken toward the better score rank
                bi = max(range(len(sims)), key=lambda i: (sims[i], -i))
                sel_sim = sims[bi]
                if rmsds[bi] is not None:
                    sel_rmsd, sel_rank = rmsds[bi], bi + 1

    # Persist this run's poses under a name unique to (structure, engine, seed, exh).
    # The sweep otherwise reuses one `_docked.pdbqt` per (structure, engine) and
    # clobbers it every run, which would lose the per-seed poses and force a full
    # re-dock to add a new scoring axis.
    # Keeping them makes any future rescoring free of docking.
    poses_file = ""
    if keep_poses_as:
        try:
            os.makedirs(os.path.dirname(keep_poses_as), exist_ok=True)
            shutil.copyfile(docked, keep_poses_as)
            side = os.path.splitext(keep_poses_as)[0] + "_rmsd.csv"
            with open(side, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Pose_Rank", "RMSD_A", "Vina_Energy"])
                for i, r in enumerate(rmsds):
                    w.writerow([i + 1,
                                "" if r is None else round(float(r), 3),
                                round(all_e[i], 3) if i < len(all_e) else ""])
            poses_file = keep_poses_as
        except Exception:
            poses_file = ""

    extra = {"MMPBSA_RMSD": float("nan"), "MMPBSA_Rank": 0, "MMPBSA_dG": float("nan"),
             "MMGBSA_RMSD": float("nan"), "MMGBSA_Rank": 0, "MMGBSA_dG": float("nan")}
    if mmpbsa_ctx:
        try:
            extra.update(rescore_poses(models, rmsds, mmpbsa_ctx, workdir))
        except Exception:
            pass          # rescoring must never break the docking benchmark

    def _r(x):
        return round(x, 2) if x == x else float("nan")  # NaN-safe round

    return {
        "Rank1_RMSD": _r(rank1),
        "BestPose_RMSD": _r(best),
        "Mean_RMSD": _r(mean_rmsd),
        "Selected_RMSD": _r(sel_rmsd),
        "Selected_Rank": sel_rank,
        "Selected_FPsim": round(sel_sim, 3) if sel_sim == sel_sim else float("nan"),
        "Best_Rank": best_rank,
        "Top_Energy": top_energy,
        "Mean_Energy": mean_energy,
        "Poses_File": poses_file,
        **extra,
    }


def publication_settings() -> str:
    """Standard PyMOL render settings for publication-quality figures (ray-traced,
    white background, cel-shaded outlines, orthoscopic). Shared by the validation
    overlay and the screening scripts so all exported figures look consistent.

    Returns a PyMOL command block (string); the exact text is load-bearing (consumed
    verbatim by generated .pml scripts) and must not be altered.
    """
    return (
        "# --- publication render settings ---\n"
        "bg_color white\n"
        "set ray_opaque_background, 1\n"
        "set orthoscopic, 1\n"
        "set ray_trace_mode, 1\n"
        "set ray_trace_color, black\n"
        "set ray_shadows, 0\n"
        "set antialias, 2\n"
        "set depth_cue, 0\n"
        "set stick_radius, 0.15\n"
        "set cartoon_transparency, 0.6\n"
        "set label_size, 18\n"
        "set label_color, black\n"
        "set label_outline_color, white\n"
    )


def export_overlay(docked_pdbqt: str, ref_pdb: str, receptor_pdb: str, best_rank: int,
                   mode: str, out_dir: str, name: str,
                   best_rmsd=None, rank1_rmsd=None) -> "dict | None":
    """Write reference + best-pose + rank-1-pose ligand PDBs, copy the receptor, and a
    publication-grade PyMOL overlay script. Returns a dict of file paths, or None if
    poses could not be extracted.

    The scene shows the crystal reference (green), the best-sampled pose (orange) and
    the top-scored rank-1 pose (magenta) over a faint receptor cartoon, with the
    binding-pocket side chains and the best pose's polar contacts (H-bond dashes), and
    the RMSDs annotated — so the figure communicates the sampling-vs-scoring distinction
    and interaction recovery at a glance. Running the script in PyMOL renders a 300-DPI
    PNG and saves an editable .pse session.

    Parameters
    ----------
    docked_pdbqt : multi-MODEL docked poses PDBQT.
    ref_pdb / receptor_pdb : crystal ligand and receptor source PDBs (copied into out_dir).
    best_rank : 1-based rank of the best-sampled pose to highlight.
    mode : "hydrated" (strip W atoms first) or Vina.
    out_dir / name : output dir and filename stem for the exported package.
    best_rmsd / rank1_rmsd : optional numbers for the figure caption.

    Every emitted PyMOL line is load-bearing render text — annotate only, never alter.
    """
    src = docked_pdbqt
    if mode == "hydrated":
        src = os.path.join(out_dir, f"{name}_dewater.pdbqt")
        with open(docked_pdbqt) as fi, open(src, "w") as fo:
            for ln in fi:
                if not hydrated.is_water_pdbqt_line(ln):
                    fo.write(ln)
    multimodel = os.path.join(out_dir, f"{name}_poses.pdb")
    subprocess.run(["obabel", src, "-O", multimodel], capture_output=True)

    models, cur = [], []
    for line in open(multimodel):
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith("ENDMDL"):
            models.append("".join(cur))
        elif not line.startswith(("MASTER", "END")):
            cur.append(line)
    if not models and cur:
        models.append("".join(cur))
    if not models:
        return None

    ref_out = os.path.join(out_dir, f"{name}_reference.pdb")
    rank1_out = os.path.join(out_dir, f"{name}_rank1.pdb")
    best_out = os.path.join(out_dir, f"{name}_best.pdb")
    rec_out = os.path.join(out_dir, f"{name}_receptor.pdb")
    pml_out = os.path.join(out_dir, f"{name}_overlay.pml")

    shutil.copyfile(ref_pdb, ref_out)
    shutil.copyfile(receptor_pdb, rec_out)
    with open(rank1_out, "w") as f:
        f.write(models[0])
    bi = (best_rank - 1) if 1 <= best_rank <= len(models) else 0
    with open(best_out, "w") as f:
        f.write(models[bi])

    if best_rmsd is not None and rank1_rmsd is not None:
        caption = f"best {best_rmsd} A   |   rank-1 {rank1_rmsd} A"
    else:
        caption = "docked vs crystal reference"

    with open(pml_out, "w") as f:
        f.write(f"# Publication overlay: {name}\n")
        f.write("# green = crystal reference | orange = best-sampled pose | magenta = top-scored (rank-1) pose\n")
        f.write("hide everything\n\n")
        f.write(f"load {os.path.basename(rec_out)}, receptor\n")
        f.write(f"load {os.path.basename(ref_out)}, reference\n")
        f.write(f"load {os.path.basename(best_out)}, docked_best\n")
        f.write(f"load {os.path.basename(rank1_out)}, docked_rank1\n\n")

        f.write(publication_settings() + "\n")

        # Receptor: faint cartoon for context only.
        f.write("show cartoon, receptor\ncolor gray80, receptor\n\n")

        # Ligands: contrasting carbon colours, CPK heteroatoms.
        f.write("show sticks, reference\ncolor green, reference\nutil.cnc reference\n")
        f.write("show sticks, docked_best\ncolor orange, docked_best\nutil.cnc docked_best\n")
        f.write("show sticks, docked_rank1\ncolor magenta, docked_rank1\nutil.cnc docked_rank1\n\n")

        # Binding-pocket side chains around the crystal reference.
        f.write("select pocket, byres (receptor within 5 of reference) and polymer\n")
        f.write("show sticks, pocket and sidechain\ncolor wheat, pocket\nutil.cnc pocket\n")
        f.write("set stick_radius, 0.11, pocket\n\n")

        # Polar contacts (H-bond dashes) recovered by the best-sampled pose.
        f.write("distance hbonds, (docked_best and (elem N+O)), (pocket and (elem N+O)), 3.5, mode=2\n")
        f.write("set dash_color, marine, hbonds\nset dash_width, 3, hbonds\nhide labels, hbonds\n\n")

        # RMSD annotation pinned at the reference ligand.
        f.write(f'pseudoatom rmsd_tag, selection=reference, label="{caption}"\n')
        f.write("hide nonbonded, rmsd_tag\n\n")

        # Camera + render to a publication PNG (comment out to adjust the view first).
        f.write("orient reference\nzoom reference, 5\n\n")
        f.write("# --- render (comment out the next three lines to tweak the view first) ---\n")
        f.write("ray 2400, 1800\n")
        f.write(f"png {name}_figure.png, dpi=300\n")
        f.write(f"save {name}_session.pse\n")

    return {"reference": ref_out, "best": best_out, "rank1": rank1_out,
            "receptor": rec_out, "pml": pml_out}


# --------------------------------------------------------------------------- #
#  Resumable batch sweep (shared by the Streamlit app and the headless CLI).   #
# --------------------------------------------------------------------------- #
BATCH_COLUMNS = ["Structure", "Ligand", "Engine", "Seed", "Exhaustiveness",
                 "Rank1_RMSD", "BestPose_RMSD", "Selected_RMSD", "Mean_RMSD",
                 "Best_Rank", "Selected_Rank", "Selected_FPsim",
                 "Top_Energy", "Mean_Energy", "Heavy_Atoms", "Rotatable_Bonds",
                 # End-point rescoring axes (populated only when a sweep runs with an
                 # mmpbsa_ctx). APPEND-ONLY: older result CSVs must stay readable, and
                 # DictWriter fills these as blank for runs that predate them.
                 "MMPBSA_RMSD", "MMPBSA_Rank", "MMPBSA_dG",
                 "MMGBSA_RMSD", "MMGBSA_Rank", "MMGBSA_dG", "Poses_File"]


def _done_keys(csv_path: str) -> set:
    """Set of (Structure, Engine, Seed, Exhaustiveness) runs already in csv_path.

    Used to resume a crashed/interrupted sweep: any run already written is skipped,
    so re-pointing at the same output folder continues where it left off. Missing/
    malformed rows are tolerated (skipped) rather than raising.
    """
    import csv as _csv
    done = set()
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="") as f:
                for row in _csv.DictReader(f):
                    try:
                        # Seed/Exhaustiveness via float() first so "42.0"-style values
                        # from re-saved CSVs still parse to the same int key.
                        done.add((row["Structure"], row["Engine"],
                                  int(float(row["Seed"])), int(float(row["Exhaustiveness"]))))
                    except (KeyError, ValueError, TypeError):
                        continue
        except Exception:
            pass
    return done


_MANIFEST_LOCK = threading.Lock()


def _append_pose_manifest(out_dir: str, structure: str, engine: str, seed: int,
                          exh: int, poses_file: str) -> None:
    """Record (run -> pose file) so stored poses can be found without re-docking."""
    path = os.path.join(out_dir, "poses_manifest.csv")
    with _MANIFEST_LOCK:
        try:
            new = not os.path.isfile(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["Structure", "Engine", "Seed", "Exhaustiveness",
                                "Poses_File", "Per_Pose_CSV"])
                w.writerow([structure, engine, seed, exh, poses_file,
                            os.path.splitext(poses_file)[0] + "_rmsd.csv"])
        except OSError:
            pass


def run_batch_sweep(jobs: list, out_dir: str, seeds: list, exh_list: list, engines: list,
                    box_edge: float, n_poses: int, cpu: int = 4, ph: str = "7.4",
                    cooldown: float = 0, csv_path: "str | None" = None,
                    progress_cb=None, log_cb=None, make_overlays: bool = True,
                    keep_poses: bool = True, mmpbsa_cfg: "dict | None" = None) -> "tuple[str, list]":
    """Run the full redocking-validation sweep with crash-safe, resumable output.

    Each completed run is appended to the results CSV *immediately* (and flushed),
    so an interruption loses at most the run in progress; re-running with the same
    `out_dir` skips runs already present and continues. This is the single code path
    used by both the app's Batch Validation tab and the headless CLI.

    Parameters
    ----------
    jobs : list of dict, each {"name", "pdb_path", "ligand"} (ligand = 3-letter code)
    engines : list of (mode, label) e.g. [("vina","Vina"),("hydrated","AD4 (hydrated)")]
    box_edge : grid box edge in Angstrom (cubic)
    seeds / exh_list : random seeds and exhaustiveness values to sweep (full matrix)
    n_poses / cpu / ph / cooldown : per-run docking controls (cooldown = sleep between runs)
    csv_path : override results-CSV path (defaults to out_dir/batch_validation_results.csv)
    progress_cb(done, total, message) / log_cb(message) : optional UI callbacks
    keep_poses : persist every run's poses to `<struct>/poses/<name>__<mode>_s<seed>_e<exh>.pdbqt`
        plus a per-pose RMSD/energy CSV, and index them in `poses_manifest.csv`. Costs ~10 kB
        per run and means a later rescoring experiment never has to re-dock.
    mmpbsa_cfg : enables end-point rescoring. Keys as for `rescore_poses`, plus
        `smiles_by_ligand` mapping the 3-letter ligand code to its SMILES. Runs are
        MUCH slower with this on (~100 s per pose), so leave it None for plain sweeps.
    make_overlays : also write a representative PyMOL overlay per (structure, engine)

    Returns (csv_path, overlays).
    """
    import csv as _csv
    import time as _time

    os.makedirs(out_dir, exist_ok=True)
    csv_path = csv_path or os.path.join(out_dir, "batch_validation_results.csv")
    done = _done_keys(csv_path)
    total = len(jobs) * len(engines) * len(seeds) * len(exh_list)
    progress = len(done)            # count already-completed runs toward progress
    overlays = []
    box = [box_edge, box_edge, box_edge]
    rep_exh = max(exh_list) if exh_list else None

    new_file = (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0
    fcsv = open(csv_path, "a", newline="")
    writer = _csv.DictWriter(fcsv, fieldnames=BATCH_COLUMNS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        fcsv.flush()

    def _emit(msg):
        if progress_cb:
            progress_cb(progress, total, msg)

    try:
        for job in jobs:
            name, pdb_path, lig_code = job["name"], job["pdb_path"], job["ligand"].strip()
            struct_dir = os.path.join(out_dir, name)
            os.makedirs(struct_dir, exist_ok=True)
            try:
                text = open(pdb_path, "r", errors="ignore").read()
            except Exception as ex:
                if log_cb:
                    log_cb(f"Skipping {name}: cannot read {pdb_path} - {ex}")
                progress += len(engines) * len(seeds) * len(exh_list)
                _emit(f"skip {name}")
                continue

            block, center = extract_native_ligand(text, lig_code)
            if not block:
                if log_cb:
                    log_cb(f"Skipping {name}: ligand '{lig_code}' not found.")
                progress += len(engines) * len(seeds) * len(exh_list)
                _emit(f"skip {name}")
                continue
            ref_pdb = os.path.join(struct_dir, "reference_ligand.pdb")
            with open(ref_pdb, "w") as f:
                f.write(block)
            heavy, rot = ligand_descriptors(ref_pdb)

            for mode, engine_label in engines:
                todo = [(s, e) for s in seeds for e in exh_list
                        if (name, engine_label, int(s), int(e)) not in done]
                if not todo:
                    if log_cb:
                        log_cb(f"{name} ({engine_label}): already complete - skipping.")
                    continue
                eng_dir = os.path.join(struct_dir, mode)
                os.makedirs(eng_dir, exist_ok=True)
                try:
                    _emit(f"{name} ({engine_label}) - preparing receptor maps")
                    map_prefix, rec_pdbqt = prepare_receptor_maps(
                        pdb_path, name, center, box, eng_dir, mode, str(ph))
                    lig_pdbqt = prepare_reference_ligand(
                        ref_pdb, os.path.join(eng_dir, "ref.pdbqt"), mode, str(ph), eng_dir)
                except Exception as ex:
                    if log_cb:
                        log_cb(f"Skipping {name} ({engine_label}): prep failed - {ex}")
                    progress += len(todo)
                    _emit(f"prep failed {name}")
                    continue

                for seed, exh in todo:
                    _emit(f"{name} ({engine_label}) - seed {seed}, exhaustiveness {exh}")
                    try:
                        # One pose file per run -- never overwritten (see redock_once).
                        keep_as = (os.path.join(struct_dir, "poses",
                                                f"{name}__{mode}_s{int(seed)}_e{int(exh)}.pdbqt")
                                   if keep_poses else None)
                        ctx = None
                        if mmpbsa_cfg:
                            ctx = dict(mmpbsa_cfg)
                            ctx.setdefault("receptor_pdb", pdb_path)
                            ctx["smiles"] = ctx.get("smiles_by_ligand", {}).get(lig_code) \
                                or ctx.get("smiles")
                            if not ctx.get("smiles"):
                                ctx = None          # no SMILES -> cannot parameterise; skip
                        res = redock_once(map_prefix, lig_pdbqt, ref_pdb, mode,
                                          int(seed), int(exh), n_poses, eng_dir,
                                          cpu=int(cpu), receptor_pdbqt=rec_pdbqt,
                                          keep_poses_as=keep_as, mmpbsa_ctx=ctx)
                        res.update({"Structure": name, "Ligand": lig_code, "Engine": engine_label,
                                    "Seed": int(seed), "Exhaustiveness": int(exh),
                                    "Heavy_Atoms": heavy, "Rotatable_Bonds": rot})
                        writer.writerow(res)
                        fcsv.flush()
                        if res.get("Poses_File"):
                            _append_pose_manifest(out_dir, name, engine_label, int(seed),
                                                  int(exh), res["Poses_File"])
                        if make_overlays and int(seed) == int(seeds[0]) and int(exh) == int(rep_exh):
                            try:
                                ov = export_overlay(
                                    os.path.join(eng_dir, "_docked.pdbqt"), ref_pdb, pdb_path,
                                    res["Best_Rank"], mode, struct_dir, f"{name}__{mode}",
                                    best_rmsd=res["BestPose_RMSD"], rank1_rmsd=res["Rank1_RMSD"])
                                if ov:
                                    overlays.append({"label": f"{name} - {engine_label}",
                                                     "best_rmsd": res["BestPose_RMSD"],
                                                     "rank1_rmsd": res["Rank1_RMSD"], **ov})
                            except Exception:
                                pass
                    except Exception as ex:
                        if log_cb:
                            log_cb(f"Run failed ({name}, {engine_label}, seed {seed}, exh {exh}): {ex}")
                    progress += 1
                    _emit(f"{name} ({engine_label}) - seed {seed}, exhaustiveness {exh}")
                    if cooldown:
                        _time.sleep(cooldown)
    finally:
        fcsv.close()
    return csv_path, overlays
