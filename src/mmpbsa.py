#!/usr/bin/env python3
"""
End-point free-energy rescoring of docked poses (AmberTools backend)
====================================================================

Computes **MM/GBSA** (Generalized Born) and **MM/PBSA** (Poisson-Boltzmann) on the
same minimised geometry. The validated protocol is a hybrid -- GB minimisation, PB
scoring at solute dielectric indi=2 -- because MM/GBSA's ranking collapses onto van
der Waals burial: specific electrostatics are cancelled almost exactly by the GB
desolvation penalty. Raising the solute dielectric damps that cancellation.

Rescores poses that the Vina/AD4 pipeline has already generated, so that pose
*selection* and compound *ranking* can be tested against a physics-based energy
instead of the empirical docking score.

Protocol:

  pose  ->  split complex into receptor + ligand
        ->  ligand bond orders recovered from the known SMILES (never perceived
            from coordinates -- that is the documented failure mode in
            ``hydrated.reference_mol_for_hydration``)
        ->  antechamber (AM1-BCC / GAFF2) + parmchk2      [cached per SMILES]
        ->  tleap: ff14SB protein + GAFF2 ligand, single-trajectory topologies
        ->  sander: restrained implicit-solvent minimisation
        ->  MMPBSA.py: single-snapshot MM/GBSA or MM/PBSA

Deliberate design choices
-------------------------
* **Restrained minimisation, not MD.** Unrestrained MD of a GPCR in implicit
  solvent *without a lipid bilayer* solvates the TM bundle as though it were in
  water -- physically wrong, and it would degrade results while appearing more
  rigorous. Explicit-membrane MD is the correct alternative but is
  disproportionate for thousands of poses. The receptor is therefore restrained
  near its experimental geometry and only the ligand/pocket relaxes.
* **ff14SB, not ff19SB.** ff19SB is parameterised for OPC explicit water and is
  the wrong pairing for implicit GB.
* **Only chains near the ligand are kept.** In the single-trajectory approach the
  rest of the assembly (e.g. the G protein, >30 A away) cancels *exactly* in
  dG = E_complex - E_receptor - E_ligand, so carrying it is pure cost.
* **Entropy is not computed.** Standard for relative ranking within a congeneric
  series; absolute affinities are not claimed.

AmberTools may live in the docking environment or in a separate conda env, so
every Amber call is a subprocess with ``AMBERHOME`` set -- the same pattern
``hydrated.py`` uses for ``autogrid4``. ``AMBERHOME`` is taken from the env var,
else from the ``tleap`` on PATH, else the active conda env.

Failure contract: any pose that cannot be parameterised or scored returns NaN and
is logged. A single bad pose never aborts a sweep.
"""
from __future__ import annotations

import atexit
import glob
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _find_amberhome() -> str:
    """AMBERHOME env var, else the install prefix of the tleap on PATH, else the active env."""
    if os.environ.get("AMBERHOME"):
        return os.environ["AMBERHOME"]
    tleap = shutil.which("tleap")
    if tleap:
        return os.path.dirname(os.path.dirname(os.path.realpath(tleap)))
    return os.environ.get("CONDA_PREFIX") or sys.prefix


AMBERHOME = _find_amberhome()

# Chains with no atom within this distance of the ligand are dropped (they cancel
# exactly in the single-trajectory free energy).
CHAIN_CUTOFF_A = 12.0
# Disulfides are detected geometrically and declared to tleap as CYX.
SS_BOND_MAX_A = 2.5

STD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
             "THR TRP TYR VAL HID HIE HIP CYX".split())


def _amber_env() -> dict:
    """Environment for AmberTools subprocesses (AMBERHOME + its python path)."""
    env = os.environ.copy()
    env["AMBERHOME"] = AMBERHOME
    env["PATH"] = os.path.join(AMBERHOME, "bin") + os.pathsep + env.get("PATH", "")
    # MMPBSA.py imports MMPBSA_mods from the env's site-packages (python version varies).
    sp = next((d for d in glob.glob(os.path.join(AMBERHOME, "lib", "python3*", "site-packages"))
               if os.path.isdir(os.path.join(d, "MMPBSA_mods"))), None)
    if sp:
        env["PYTHONPATH"] = sp + os.pathsep + env.get("PYTHONPATH", "")
    return env


# Process groups of AmberTools children currently in flight. A sweep runs many poses
# in parallel, so an interrupt must take the whole tree down: killing only the Python
# driver leaves `sander` processes running (one per worker) that keep every core busy
# indefinitely. Each child therefore gets its own session, and the handlers below kill
# the group rather than the single direct child.
_ACTIVE_PGIDS: set = set()

# SIGKILL cannot be caught, so a `kill -9`-ed driver leaves its children running. Every
# in-flight group id is therefore also journalled to this file, giving a precise recovery
# path afterwards (`cleanup_stale()`) that cannot hit unrelated sander jobs the user may
# be running. Set via the MMPBSA_PGID_FILE env var; disabled when empty.
PGID_FILE = os.environ.get("MMPBSA_PGID_FILE", "")


# Poses are scored on parallel worker threads, so the journal's read-modify-write must
# be serialised -- without this, concurrent updates lose entries and the corresponding
# process group survives cleanup (observed: 4 children, only 3 journalled).
_JOURNAL_LOCK = threading.Lock()


def _journal_pgid(pgid: int, add: bool) -> None:
    if not PGID_FILE:
        return
    with _JOURNAL_LOCK:
        try:
            cur = set()
            if os.path.exists(PGID_FILE):
                cur = {int(x) for x in open(PGID_FILE).read().split() if x.strip().isdigit()}
            cur.add(pgid) if add else cur.discard(pgid)
            with open(PGID_FILE, "w") as fh:
                fh.write(" ".join(str(p) for p in sorted(cur)))
        except OSError:
            pass


def cleanup_stale(path: str | None = None) -> int:
    """Kill process groups left behind by a SIGKILL-ed run; returns how many were killed."""
    path = path or PGID_FILE
    if not path or not os.path.exists(path):
        return 0
    killed = 0
    for tok in open(path).read().split():
        if tok.strip().isdigit():
            try:
                os.killpg(int(tok), 0)          # probe: does the group still exist?
            except OSError:
                continue
            _kill_group(int(tok))
            killed += 1
    try:
        os.remove(path)
    except OSError:
        pass
    return killed


def _kill_group(pgid: int) -> None:
    """SIGTERM then SIGKILL an entire process group, ignoring races."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.2)


def _kill_all_active() -> None:
    """Tear down every in-flight AmberTools process group (atexit / signal handler)."""
    for pgid in list(_ACTIVE_PGIDS):
        _kill_group(pgid)
    _ACTIVE_PGIDS.clear()


atexit.register(_kill_all_active)
for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        _prev = signal.getsignal(_sig)

        def _handler(signum, frame, _prev=_prev):
            _kill_all_active()
            if callable(_prev):
                _prev(signum, frame)
            else:
                raise KeyboardInterrupt if signum == signal.SIGINT else SystemExit(1)

        signal.signal(_sig, _handler)
    except (ValueError, OSError):
        pass          # not the main thread / signal unavailable -- atexit still applies


def _run(cmd: list, cwd: str, timeout: int = 1800) -> "subprocess.CompletedProcess":
    """Run an AmberTools command, capturing output (never raises on non-zero).

    The child is placed in its own process group so that a timeout -- or an interrupt
    of the parent -- kills the whole tree instead of orphaning `sander`/`mmpbsa_py_energy`.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env=_amber_env(), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is not None:
        _ACTIVE_PGIDS.add(pgid)
        _journal_pgid(pgid, True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            _kill_group(pgid)
        else:
            proc.kill()
        out, err = proc.communicate()
        out = (out or "") + f"\n[timed out after {timeout}s]"
    except BaseException:                    # KeyboardInterrupt / SystemExit included
        if pgid is not None:
            _kill_group(pgid)
        raise
    finally:
        if pgid is not None:
            _ACTIVE_PGIDS.discard(pgid)
            _journal_pgid(pgid, False)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


class RescoreError(RuntimeError):
    """Raised internally; callers get NaN via the public scoring helpers."""


MMGBSAError = RescoreError          # backwards-compatible alias


# --------------------------------------------------------------------------- #
# Pose / complex handling
# --------------------------------------------------------------------------- #
def _xyz(line: str):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _is_hydrogen(line: str) -> bool:
    elem = line[76:78].strip()
    name = line[12:16].strip()
    return elem == "H" or (not elem and name[:1] == "H")


# Two-letter elements whose symbol gets truncated to one letter in the element column
# by PDBQT->PDB conversion (e.g. chlorine written as name "CL19" but element " C").
# RDKit then reads a carbon, the ligand graph no longer matches the SMILES template,
# and bond-order transfer fails. Same repair as make_complex.ligand_lines().
TWO_LETTER_ELEMENTS = {"CL": "Cl", "BR": "Br", "SE": "Se", "SI": "Si", "FE": "Fe",
                       "ZN": "Zn", "MG": "Mg", "MN": "Mn", "NA": "Na", "CA": "Ca",
                       "CU": "Cu", "NI": "Ni", "CO": "Co", "PT": "Pt"}


def _fix_element_columns(lines: list) -> "list[str]":
    """Return atom + CONECT lines with the element column repaired from the atom name.

    CONECT records are preserved: they carry the true connectivity, so RDKit does not
    have to re-perceive bonds from interatomic distances.
    """
    out = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            ln = ln.rstrip("\n").ljust(80)
            aname = ln[12:16].strip().upper()
            for sym, elem in TWO_LETTER_ELEMENTS.items():
                if aname.startswith(sym):
                    ln = ln[:76] + f"{elem:>2}" + ln[78:]
                    break
            out.append(ln.rstrip() + "\n")
        elif ln.startswith("CONECT"):
            out.append(ln if ln.endswith("\n") else ln + "\n")
    return out


def iter_models(ensemble_pdb: str):
    """Yield (model_number, [lines]) for each MODEL in an ensemble PDB.

    The screening pipeline writes the FULL receptor+ligand complex per MODEL
    (receptor as ATOM, docked ligand relabelled HETATM), so an ensemble file is
    already a stack of MM/GBSA-ready complexes.

    Pose rank is a running 1-based counter rather than the declared MODEL number:
    some exported pose files carry a nested `MODEL 1` from the per-pose conversion,
    so the declared numbers are not unique. Poses are written best-score-first, so
    the counter *is* the rank.
    """
    cur, idx = [], 0
    for ln in open(ensemble_pdb):
        if ln.startswith("MODEL"):
            if cur:                      # nested/duplicate MODEL tag: keep accumulating
                continue
            cur = []
        elif ln.startswith("ENDMDL"):
            if cur:
                idx += 1
                yield idx, cur
            cur = []
        else:
            cur.append(ln)
    if any(l.startswith(("ATOM", "HETATM")) for l in cur):   # trailing MASTER/END is not a pose
        yield idx + 1, cur


def split_complex(lines: list, cutoff: float = CHAIN_CUTOFF_A) -> "tuple[list, list]":
    """Split complex lines into (receptor ATOM lines near the ligand, ligand HETATM lines).

    Only chains with an atom within `cutoff` of the ligand are retained -- distant
    subunits cancel exactly in the single-trajectory free energy.
    """
    lig = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() != "HOH"]
    rec_all = [l for l in lines if l.startswith("ATOM")]
    if not lig or not rec_all:
        raise RescoreError("complex lacks a ligand or a receptor")
    lig_xyz = np.array([_xyz(l) for l in lig])

    by_chain: dict = {}
    for l in rec_all:
        by_chain.setdefault(l[21], []).append(l)
    keep = []
    for ch, ls in by_chain.items():
        xyz = np.array([_xyz(l) for l in ls])
        if np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2).min() <= cutoff:
            keep.extend(ls)
    if not keep:
        raise RescoreError("no receptor chain within cutoff of the ligand")
    return keep, lig


def write_receptor_pdb(rec_lines: list, out_pdb: str) -> "list[tuple]":
    """Write a tleap-ready receptor PDB (heavy atoms only) and return disulfide pairs.

    Hydrogens are stripped so tleap adds a complete, correctly-named Amber set --
    the pipeline's receptors come from PDBQT prep and carry merged/partial H.
    Cysteines in an SS bond are renamed CYX and reported so the caller can emit
    the matching ``bond`` commands (tleap does not infer them).
    """
    heavy = [l for l in rec_lines if not _is_hydrogen(l)]
    # geometric disulfide detection on SG atoms
    sg = [(i, l) for i, l in enumerate(heavy) if l[12:16].strip() == "SG"]
    ss_res, pairs = set(), []
    for a in range(len(sg)):
        for b in range(a + 1, len(sg)):
            ia, la = sg[a]
            ib, lb = sg[b]
            if np.linalg.norm(np.array(_xyz(la)) - np.array(_xyz(lb))) <= SS_BOND_MAX_A:
                ka = (la[21], int(la[22:26]))
                kb = (lb[21], int(lb[22:26]))
                ss_res.add(ka)
                ss_res.add(kb)
                pairs.append((ka, kb))

    out, seen_res, order = [], {}, []
    for l in heavy:
        key = (l[21], int(l[22:26]))
        if l[17:20].strip() == "CYS" and key in ss_res:
            l = l[:17] + "CYX" + l[20:]
        if key not in seen_res:
            seen_res[key] = len(order) + 1        # 1-based tleap residue index
            order.append(key)
        out.append(l)
    # TER must also be emitted at INTERNAL chain breaks, not just when the chain letter
    # changes. Cryo-EM GPCR models routinely omit flexible loops; the residue before such
    # a gap carries a terminal OXT, and without a TER tleap reads that OXT as mid-chain
    # and dies with "Atom .R<GLY 232>.A<OXT 8> does not have a type." Splitting at the gap
    # lets tleap build proper termini for each fragment.
    #
    # The artificial charged termini this creates are an approximation, but the receptor is
    # identical in the complex and receptor terms of the single-trajectory free energy, so
    # their contribution very largely cancels in dG.
    with open(out_pdb, "w") as fh:
        prev_chain, prev_resnum = None, None
        for l in out:
            chain, resnum = l[21], int(l[22:26])
            if prev_chain is not None and (chain != prev_chain or
                                           (resnum != prev_resnum and resnum != prev_resnum + 1)):
                fh.write("TER\n")
            fh.write(l if l.endswith("\n") else l + "\n")
            prev_chain, prev_resnum = chain, resnum
        fh.write("TER\nEND\n")
    # map disulfide pairs onto tleap residue indices
    return [(seen_res[a], seen_res[b]) for a, b in pairs]


def protonate_smiles(smiles: str, ph: float = 7.4) -> str:
    """Return the SMILES protonated to `ph`, using the SAME `obabel -p` call the docking
    pipeline uses for ligand prep (run_screen.py) — so the rescored species matches what
    was actually docked.

    This is the single most important correctness fix for MM/PBSA on this pipeline: the
    docking protonates every ligand at pH 7.4 (aminergic amines -> +1, forming the
    conserved Asp3.32 salt bridge; carboxylates -> anion). Parameterising the neutral CCD
    form instead deletes those charges from the electrostatics. obabel is a heuristic, not
    a rigorous pKa predictor, but matching the docking is exactly the point. Falls back to
    the input SMILES if obabel yields nothing.
    """
    try:
        r = subprocess.run(["obabel", f"-:{smiles}", "-osmi", "-p", str(ph)],
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout or "").split()
        return out[0] if out else smiles
    except Exception:
        return smiles


def ligand_mol_from_pose(lig_lines: list, smiles: str, ph: "float | None" = 7.4) -> "Chem.Mol":
    """Rebuild a chemically-correct ligand from pose coordinates + the known SMILES.

    Bond orders and formal charges come from the SMILES template, never from
    coordinate perception (which mis-assigns exactly the over-valent nitrogens and
    fused heteroaromatics documented in ``hydrated.py``). The template is first
    protonated to `ph` (default 7.4, matching the docking) so the formal charges that
    transfer onto the pose are the physiological ones; pass ``ph=None`` to keep the
    ligand exactly as the SMILES specifies. Hydrogens are added with coordinates so
    antechamber sees a complete molecule.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdmolops  # noqa: F401  (import kept explicit for clarity)

    block = "".join(_fix_element_columns(lig_lines)) + "END\n"
    pose = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    if pose is None:
        raise RescoreError("could not read ligand pose PDB block")
    # `removeHs=True` is a no-op when sanitize=False, so strip explicitly. Leftover
    # explicit hydrogens change the heavy-atom degrees, the template then maps onto the
    # wrong atoms, and sanitisation reports an over-valent nitrogen.
    pose = Chem.RemoveHs(pose, sanitize=False)
    template_smiles = protonate_smiles(smiles, ph) if ph is not None else smiles
    template = Chem.MolFromSmiles(template_smiles) or Chem.MolFromSmiles(smiles)
    if template is None:
        raise RescoreError(f"unparseable SMILES: {smiles}")
    template = Chem.RemoveHs(template)
    # Kekulize the template first: transferring *aromatic* bond orders onto a mol that
    # carries no aromatic perception leaves RDKit unable to kekulize the result (fused
    # purines fail this way). Explicit single/double bonds transfer cleanly instead.
    try:
        Chem.Kekulize(template, clearAromaticFlags=True)
    except Exception:
        pass                                  # non-kekulizable template: try as-is
    # AssignBondOrdersFromTemplate emits "More than one matching pattern found - picking
    # one" whenever the ligand has a symmetric substructure: the several substructure
    # matches are symmetry-equivalent, so the pick is immaterial and the message is pure
    # noise. Silence just that benign warning during the transfer, then restore logging.
    RDLogger.DisableLog("rdApp.warning")
    try:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, pose)
        except Exception as exc:
            pose = _repair_collapsed_halogens(pose, template)
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, pose)
            except Exception:
                # Last resort for poses whose bond graph can't be perceived (a fused-ring bond
                # stretched by embedding/PDBQT round-trip -- RDKit and OpenBabel both miss it, so
                # the template won't map). Ignore the pose bonds entirely: match atoms by
                # element-MCS and transplant the pose coordinates onto the template's known graph.
                # The subsequent restrained minimisation relaxes the strained ring.
                mol = _mcs_coord_transplant(template, pose)
                if mol is None:
                    raise RescoreError(f"bond-order transfer failed: {exc}")
    finally:
        RDLogger.EnableLog("rdApp.warning")
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(mol)
    return mol


def _mcs_coord_transplant(template: "Chem.Mol", pose: "Chem.Mol") -> "Chem.Mol | None":
    """Rebuild the ligand when pose bond-perception is unreliable: find the atom
    correspondence by maximum-common-substructure (by element, bond order ignored) and
    put the pose's coordinates onto the *template's* correct connectivity. Returns a mol
    with template chemistry + pose geometry, or None if the MCS doesn't cover the ligand.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from rdkit.Geometry import Point3D
    try:
        mcs = rdFMCS.FindMCS([template, pose],
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             ringMatchesRingOnly=False, completeRingsOnly=False, timeout=30)
        if not mcs.smartsString or mcs.numAtoms < template.GetNumHeavyAtoms():
            return None                        # partial cover -> can't place every atom
        q = Chem.MolFromSmarts(mcs.smartsString)
        tm, pm = template.GetSubstructMatch(q), pose.GetSubstructMatch(q)
        if not tm or not pm or len(tm) != template.GetNumHeavyAtoms():
            return None
        conf = pose.GetConformer()
        tconf = Chem.Conformer(template.GetNumAtoms())
        for ti, pi in zip(tm, pm):
            p = conf.GetAtomPosition(pi)
            tconf.SetAtomPosition(ti, Point3D(p.x, p.y, p.z))
        out = Chem.Mol(template)
        out.RemoveAllConformers()
        out.AddConformer(tconf, assignId=True)
        return out
    except Exception:
        return None


def _repair_collapsed_halogens(pose: "Chem.Mol", template: "Chem.Mol") -> "Chem.Mol":
    """Restore halogen identity that PDBQT->PDB conversion collapsed onto carbon.

    Some ligands emerge from `prepare_reference_ligand` with aromatic-ring chlorines
    typed as aromatic carbon (AutoDock type `A`, atom name `C`), so OpenBabel writes a
    PDB where the halogen reads as carbon. The pose then has the right atom COUNT but
    the wrong composition, and the template match fails. Because ring halogens are
    terminal (a single heavy-atom neighbour) and carbons in a drug-like scaffold rarely
    are, the terminal heavy atoms mis-read as carbon are re-typed to the template's
    halogens, matched by neighbour environment. Non-halogen mismatches are left alone.
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    HAL = {"F", "Cl", "Br", "I"}
    want = [a.GetSymbol() for a in template.GetAtoms() if a.GetSymbol() in HAL]
    if not want:
        return pose
    have = sum(1 for a in pose.GetAtoms() if a.GetSymbol() in HAL)
    if have >= len(want):
        return pose

    rw = Chem.RWMol(pose)
    try:
        rdmolops.FastFindRings(rw)            # need ring membership to spot terminal atoms
    except Exception:
        pass
    # terminal heavy carbons in the pose are the collapsed-halogen candidates
    cand = [a.GetIdx() for a in rw.GetAtoms()
            if a.GetSymbol() == "C" and a.GetDegree() == 1 and not a.GetIsAromatic()]
    # need one candidate per missing halogen; assign by descending atomic number so
    # heavier halogens (which are less ambiguous) are placed first
    need = sorted((s for s in want), key=lambda s: -Chem.GetPeriodicTable().GetAtomicNumber(s))
    missing = need[: len(want) - have] if have else need
    for sym, idx in zip(missing, cand):
        rw.GetAtomWithIdx(idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(sym))
    return rw.GetMol()


# --------------------------------------------------------------------------- #
# Ligand parameterisation (cached -- AM1-BCC dominates runtime)
# --------------------------------------------------------------------------- #
def _cache_key(smiles: str) -> str:
    from rdkit import Chem
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else smiles
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def prepare_ligand_params(mol, workdir: str, cache_dir: str | None = None,
                          smiles: str | None = None) -> "tuple[str, str, int]":
    """Parameterise the ligand: -> (gaff2 mol2 path, frcmod path, net charge).

    AM1-BCC charge derivation is by far the slowest step, and charges depend only
    on the molecule (not the pose), so results are cached per canonical SMILES and
    the pose-specific coordinates are re-applied afterwards.
    """
    from rdkit import Chem

    # Absolute: every Amber tool is invoked with cwd=workdir, so a relative workdir
    # would make the -i/-o arguments resolve against the wrong directory.
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    net_q = int(round(sum(a.GetFormalCharge() for a in mol.GetAtoms())))
    # Key on the MOL's own canonical SMILES, which includes formal charges, so a ligand
    # cached at one protonation state is never reused for another. Keying on the passed
    # (neutral) SMILES would collide across pH states and serve the wrong AM1-BCC charges.
    key = _cache_key(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    cdir = os.path.join(cache_dir, key) if cache_dir else None
    elements = "".join(a.GetSymbol() for a in mol.GetAtoms())

    # Hand antechamber an MDL molfile, NOT a PDB: PDB carries no bond orders, so
    # antechamber would re-perceive them from geometry -- the very failure mode this
    # module avoids upstream (fused N-heteroaromatics get an impossible bond graph,
    # and tleap then dies building the 1-4 list). The molfile preserves the bond
    # orders recovered from the SMILES template.
    in_mol = os.path.join(workdir, "lig_in.mol")
    Chem.MolToMolFile(mol, in_mol)
    out_mol2 = os.path.join(workdir, "lig.mol2")
    frcmod = os.path.join(workdir, "lig.frcmod")

    # ---- fast path: reuse cached AM1-BCC charges for THIS pose's coordinates ----
    # Coordinates are never edited by hand (column-slicing a mol2 silently corrupts
    # the fixed-width fields). Instead antechamber re-runs in "read charges" mode,
    # which skips the expensive sqm/AM1-BCC step but still writes a well-formed
    # mol2 with the pose geometry. The cached element sequence guards against
    # applying charges to a molecule whose atom order differs.
    if cdir and os.path.exists(os.path.join(cdir, "charges.dat")) and \
            os.path.exists(os.path.join(cdir, "lig.frcmod")) and \
            os.path.exists(os.path.join(cdir, "elements.txt")) and \
            open(os.path.join(cdir, "elements.txt")).read().strip() == elements:
        r = _run(["antechamber", "-i", in_mol, "-fi", "mdl", "-o", out_mol2, "-fo", "mol2",
                  "-c", "rc", "-cf", os.path.join(cdir, "charges.dat"), "-nc", str(net_q),
                  "-at", "gaff2", "-rn", "LIG", "-pf", "y"], cwd=workdir, timeout=600)
        if os.path.exists(out_mol2):
            return out_mol2, os.path.join(cdir, "lig.frcmod"), net_q
        # otherwise fall through and recompute from scratch

    # ---- slow path: full AM1-BCC derivation (dominates runtime; cached below) ----
    r = _run(["antechamber", "-i", in_mol, "-fi", "mdl", "-o", out_mol2, "-fo", "mol2",
              "-c", "bcc", "-s", "2", "-nc", str(net_q), "-at", "gaff2", "-rn", "LIG",
              "-pf", "y"], cwd=workdir, timeout=3600)
    if not os.path.exists(out_mol2):
        raise RescoreError(f"antechamber failed: {(r.stderr or r.stdout)[-400:]}")

    r = _run(["parmchk2", "-i", out_mol2, "-f", "mol2", "-o", frcmod, "-s", "gaff2"],
             cwd=workdir)
    if not os.path.exists(frcmod):
        raise RescoreError(f"parmchk2 failed: {(r.stderr or r.stdout)[-400:]}")

    if cdir:
        os.makedirs(cdir, exist_ok=True)
        shutil.copy(out_mol2, os.path.join(cdir, "lig.mol2"))
        shutil.copy(frcmod, os.path.join(cdir, "lig.frcmod"))
        with open(os.path.join(cdir, "charges.dat"), "w") as fh:
            qs = read_mol2_charges(out_mol2)
            for i in range(0, len(qs), 8):                     # antechamber -cf layout
                fh.write("".join(f"{q:10.6f}" for q in qs[i:i + 8]) + "\n")
        with open(os.path.join(cdir, "elements.txt"), "w") as fh:
            fh.write(elements)
    return out_mol2, frcmod, net_q


def read_mol2_charges(mol2_path: str) -> "list[float]":
    """Per-atom partial charges from the @<TRIPOS>ATOM block (last field)."""
    qs, in_atoms = [], False
    for ln in open(mol2_path):
        if ln.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if ln.startswith("@<TRIPOS>"):
            if in_atoms:
                break
            continue
        if in_atoms and ln.strip():
            qs.append(float(ln.split()[-1]))
    return qs


# --------------------------------------------------------------------------- #
# Topology, minimisation, scoring
# --------------------------------------------------------------------------- #
# Each GB model requires its matching intrinsic-radii set; sander refuses to run
# igb=7/8 on mbondi2 radii (H radius 0.8 A falls outside their allowed range).
GB_RADII = {1: "mbondi", 2: "mbondi2", 5: "mbondi2", 7: "bondi", 8: "mbondi3"}

LEAP_TEMPLATE = """source leaprc.protein.ff14SB
source leaprc.gaff2
loadamberparams {frcmod}
LIG = loadmol2 {mol2}
REC = loadpdb {rec_pdb}
{ss_bonds}
COMPLEX = combine {{ REC LIG }}
set default PBRadii {radii}
saveamberparm COMPLEX complex.prmtop complex.inpcrd
saveamberparm REC receptor.prmtop receptor.inpcrd
saveamberparm LIG ligand.prmtop ligand.inpcrd
quit
"""

# Restrained implicit-solvent minimisation: the receptor is held near its
# experimental geometry (see module docstring) while the ligand/pocket relaxes.
MIN_TEMPLATE = """restrained minimisation
 &cntrl
  imin=1, maxcyc={maxcyc}, ncyc={ncyc},
  ntb=0, igb={igb}, cut=999.0, saltcon={saltcon},
  ntr=1, restraint_wt={wt}, restraintmask='{mask}',
  ntpr=200,
 /
"""

MMPBSA_TEMPLATE = """single-snapshot MM/GBSA
&general
   startframe=1, endframe=1, interval=1, verbose=2, keep_files=0,
/
&gb
   igb={igb}, saltcon={saltcon},
/
"""

# Poisson-Boltzmann variant. MMPBSA.py's &gb namelist has no interior-dielectric
# option (intdiel is not a GB variable in this build), so testing whether a higher
# solute dielectric rescues specificity -- by damping the near-exact cancellation
# between specific electrostatics and GB desolvation -- requires PB, where `indi`
# is available. Slower than GB but the more rigorous solvation model.
MMPBSA_PB_TEMPLATE = """single-snapshot MM/PBSA
&general
   startframe=1, endframe=1, interval=1, verbose=2, keep_files=0,
/
&pb
   istrng={saltcon}, indi={indi}, exdi=80.0, inp=1,
/
"""


def build_topology(rec_pdb: str, lig_mol2: str, frcmod: str, ss_pairs: list,
                   workdir: str, igb: int = 8) -> dict:
    """tleap -> complex/receptor/ligand prmtop+inpcrd (single-trajectory protocol).

    Disulfides must be declared explicitly; tleap does not infer them from geometry.
    The radii set is chosen to match `igb` (see GB_RADII) -- topologies are therefore
    GB-model specific and must be rebuilt when sweeping igb.
    """
    # tleap tokenises its input on whitespace and offers no quoting, so ANY path
    # containing a space breaks it ("loadAmberParams: Improper number of arguments!").
    # A project directory like ".../My Projects/docking" is enough to do it. Every
    # input is therefore copied into the working directory and referenced by bare
    # basename, with tleap run from that directory -- the same reason MMPBSA.py's
    # arguments are passed relative (its legacy binary has fixed-size path buffers).
    def _localise(src: str, default_name: str) -> str:
        dst = os.path.join(workdir, default_name)
        try:
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)
        except OSError as exc:
            raise RescoreError(f"could not stage {src} for tleap: {exc}")
        return default_name

    frcmod_l = _localise(frcmod, "lig.frcmod")
    mol2_l = _localise(lig_mol2, "lig.mol2")
    rec_l = _localise(rec_pdb, "rec.pdb")

    def _run_tleap(rec_ref: str, ss_str: str):
        script = LEAP_TEMPLATE.format(frcmod=frcmod_l, mol2=mol2_l, rec_pdb=rec_ref,
                                      ss_bonds=ss_str, radii=GB_RADII.get(igb, "mbondi3"))
        with open(os.path.join(workdir, "leap.in"), "w") as fh:
            fh.write(script)
        rr = _run(["tleap", "-f", "leap.in"], cwd=workdir)
        o = {k: os.path.join(workdir, f"{k}.prmtop") for k in ("complex", "receptor", "ligand")}
        o["inpcrd"] = os.path.join(workdir, "complex.inpcrd")
        return o, [k for k, p in o.items() if not os.path.exists(p)], rr

    ss = "\n".join(f"bond REC.{a}.SG REC.{b}.SG" for a, b in ss_pairs)
    out, missing, r = _run_tleap(rec_l, ss)

    if missing:
        # Fallback for ragged deposited receptors. Some structures (non-monotonic residue
        # numbering, missing-loop termini in cryo-EM depositions) make tleap mis-index the
        # hand-built receptor, so the explicit disulfide `bond` commands reference the wrong
        # residues and the build aborts. pdb4amber normalises the residue order/numbering and
        # emits CONECT records for the disulfides, which tleap then forms itself -- so NO
        # explicit `bond` lines are passed here (adding them would duplicate the CONECT and
        # fatally error). The primary attempt above is byte-for-byte unchanged, so any
        # structure that already builds never reaches this path.
        amber = os.path.join(workdir, "rec_amber.pdb")
        _run(["pdb4amber", "-i", "rec.pdb", "-o", "rec_amber.pdb"], cwd=workdir)
        if os.path.exists(amber):
            out, missing, r = _run_tleap("rec_amber.pdb", "")
            if missing:
                # Last resort: reduce the disulfides. Pathological cysteine geometries
                # (clustered SG atoms, cross-linked S-C-S angles from poorly-resolved
                # cryo-EM regions) make even pdb4amber's CONECT disulfides unbuildable.
                # Stripping the CONECT and renaming CYX->CYS leaves free (reduced)
                # cysteines that always parameterise. The receptor is IDENTICAL in the
                # complex and receptor terms of the single-trajectory free energy, so the
                # omitted SS bonds very largely cancel in dG -- the same rationale as the
                # ragged-termini approximation above.
                lines = [l for l in open(amber) if not l.startswith("CONECT")]
                lines = [(l[:17] + "CYS" + l[20:]
                          if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == "CYX" else l)
                         for l in lines]
                red = os.path.join(workdir, "rec_reduced.pdb")
                with open(red, "w") as fh:
                    fh.writelines(lines)
                out, missing, r = _run_tleap("rec_reduced.pdb", "")

    if missing:
        log = (r.stdout or "") + (r.stderr or "")
        errs = "; ".join(l.strip() for l in log.splitlines()
                         if re.search(r"error|fatal", l, re.I))[:400]
        raise RescoreError(f"tleap failed ({missing}): {errs or log[-300:]}")
    return out


def _n_receptor_residues(rec_pdb: str) -> int:
    return len({(l[21], int(l[22:26])) for l in open(rec_pdb)
                if l.startswith(("ATOM", "HETATM"))})


def minimise(tops: dict, rec_pdb: str, workdir: str, igb: int = 8,
             saltcon: float = 0.15, maxcyc: int = 250, ncyc: int = 100,
             restraint_wt: float = 5.0) -> str:
    """Restrained implicit-solvent minimisation with sander; returns the .rst7 path.

    Only a short minimisation is needed (and it dominates runtime: ~36 s/100 cycles
    for a 5 kAtom complex). With the receptor restrained, the RMS gradient falls from
    ~13 to <1 kcal/mol/A within 100 cycles; the residual drift is mostly relaxation of
    the hydrogens tleap added, which occurs in the complex AND receptor terms and so
    largely cancels in dG. The purpose here is to relieve docking clashes, not to
    converge the protein.
    """
    nres = _n_receptor_residues(rec_pdb)
    mask = f":1-{nres} & !@H="          # receptor heavy atoms restrained; ligand free
    with open(os.path.join(workdir, "min.in"), "w") as fh:
        fh.write(MIN_TEMPLATE.format(maxcyc=maxcyc, ncyc=ncyc, igb=igb,
                                     saltcon=saltcon, wt=restraint_wt, mask=mask))
    rst = os.path.join(workdir, "min.rst7")
    r = _run(["sander", "-O", "-i", "min.in", "-p", tops["complex"], "-c", tops["inpcrd"],
              "-ref", tops["inpcrd"], "-o", "min.out", "-r", "min.rst7", "-inf", "min.info"],
             cwd=workdir, timeout=3600)
    if not os.path.exists(rst):
        log = open(os.path.join(workdir, "min.out")).read() if \
            os.path.exists(os.path.join(workdir, "min.out")) else (r.stderr or r.stdout)
        raise RescoreError(f"sander minimisation failed: {log[-400:]}")
    return rst


def _parse_mmpbsa(dat_path: str) -> dict:
    """Parse the DELTA (complex - receptor - ligand) block of FINAL_RESULTS."""
    txt = open(dat_path).read()
    block = txt.split("Differences (Complex - Receptor - Ligand):")[-1]
    want = {"VDWAALS": "dE_vdW", "EEL": "dE_elec",
            "EGB": "dG_polar", "EPB": "dG_polar",              # GB / PB polar term
            "ESURF": "dG_nonpolar", "ENPOLAR": "dG_nonpolar",  # GB / PB nonpolar term
            "DELTA TOTAL": "dG_total"}
    out = {}
    for line in block.splitlines():
        for key, name in want.items():
            if line.strip().startswith(key):
                nums = re.findall(r"-?\d+\.\d+", line)
                if nums:
                    out[name] = float(nums[0])
    if "dG_total" not in out:
        raise RescoreError("could not parse DELTA TOTAL from MMPBSA output")
    return out


def mmgbsa_energy(tops: dict, traj: str, workdir: str, igb: int = 8,
                  saltcon: float = 0.15, intdiel: float = 1.0,
                  method: str = "gb") -> dict:
    """Single-snapshot MM/GBSA on `traj`; returns dG_total + decomposition (kcal/mol).

    `intdiel` is the solute interior dielectric. The default of 1.0 leaves specific
    electrostatics almost exactly cancelled by the GB desolvation penalty, so the
    ranking collapses onto van der Waals/burial; raising it to 2-4 damps that
    cancellation and is a standard variant for polar/charged binding sites.
    """
    if method == "pb":
        body = MMPBSA_PB_TEMPLATE.format(saltcon=saltcon, indi=intdiel)
        out_name = f"FINAL_RESULTS_pb_indi{intdiel:g}.dat"
    else:
        body = MMPBSA_TEMPLATE.format(igb=igb, saltcon=saltcon)
        out_name = f"FINAL_RESULTS_igb{igb}.dat"
    with open(os.path.join(workdir, "mmpbsa.in"), "w") as fh:
        fh.write(body)
    out_dat = os.path.join(workdir, out_name)
    # Every file argument is passed RELATIVE to cwd. The legacy `mmpbsa_py_energy`
    # binary has fixed-size filename buffers and fails ("failed with prmtop ...") on
    # long absolute paths -- a ~150-character scratch path is already enough to break
    # it, while the identical run with basenames succeeds.
    def _rel(p):
        return os.path.relpath(p, workdir)

    r = _run([os.path.join(AMBERHOME, "bin", "python3"),
              os.path.join(AMBERHOME, "bin", "MMPBSA.py"),
              "-O", "-i", "mmpbsa.in", "-o", out_name,
              "-cp", _rel(tops["complex"]), "-rp", _rel(tops["receptor"]),
              "-lp", _rel(tops["ligand"]), "-y", _rel(traj)],
             cwd=workdir, timeout=3600)
    if not os.path.exists(out_dat):
        raise RescoreError(f"MMPBSA.py failed: {((r.stderr or '') + (r.stdout or ''))[-500:]}")
    res = _parse_mmpbsa(out_dat)
    res["igb"] = igb
    return res


# --------------------------------------------------------------------------- #
# Orchestration (public API)
# --------------------------------------------------------------------------- #
NAN_RESULT = {"dG_total": float("nan"), "dE_vdW": float("nan"), "dE_elec": float("nan"),
              "dG_polar": float("nan"), "dG_nonpolar": float("nan")}


def score_pose(complex_lines: list, smiles: str, workdir: str,
               cache_dir: str | None = None, igb: int = 8,
               saltcon: float = 0.15, maxcyc: int = 250, intdiel: float = 1.0,
               ph: "float | None" = 7.4) -> dict:
    """Full single-method MM/GBSA (or PB via `intdiel`) for one complex; NaN on failure.

    `complex_lines` is one MODEL from an ensemble PDB (receptor ATOM + ligand HETATM).
    `smiles` supplies the bond orders; the ligand is protonated to `ph` (default 7.4,
    matching the docking) unless ph is None. Never raises -- a single unscoreable pose
    must not abort a sweep. For the standard GB-minimise + GB&PB-score protocol with
    scratch cleanup, prefer `rescore_pose`.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    try:
        rec, lig = split_complex(complex_lines)
        ss = write_receptor_pdb(rec, os.path.join(workdir, "rec.pdb"))
        mol = ligand_mol_from_pose(lig, smiles, ph=ph)
        mol2, frcmod, _q = prepare_ligand_params(mol, workdir, cache_dir=cache_dir)
        tops = build_topology(os.path.join(workdir, "rec.pdb"), mol2, frcmod, ss,
                              workdir, igb=igb)
        rst = minimise(tops, os.path.join(workdir, "rec.pdb"), workdir, igb=igb,
                       saltcon=saltcon, maxcyc=maxcyc)
        res = mmgbsa_energy(tops, rst, workdir, igb=igb, saltcon=saltcon, intdiel=intdiel)
        return res
    except Exception as exc:                      # noqa: BLE001 - failure contract
        out = dict(NAN_RESULT)
        out["error"] = str(exc)[:200]
        out["igb"] = igb
        return out


def rescore_pose(complex_lines: list, smiles: str, workdir: str, *,
                 ph: "float | None" = 7.4, igb: int = 8, indi: float = 2.0,
                 saltcon: float = 0.15, maxcyc: int = 250,
                 cache_dir: str | None = None, methods=("pb", "gb"),
                 keep_scratch: bool = False) -> dict:
    """The production protocol for one pose: GB-minimise once, score MM/PBSA and MM/GBSA
    on the SAME minimised geometry, then delete the per-pose scratch.

    This is the single entry point for pose rescoring (validation + screening + the
    stored-pose re-runs), replacing the duplicated `score_pose`-then-`mmgbsa_energy`
    logic. Returns a flat dict::

        {"pb_dG", "gb_dG", "pb_polar", "gb_polar", ..., "net_charge", "error"}

    Never raises (failure contract). `keep_scratch=True` retains the workdir for
    debugging; otherwise it is removed regardless of success -- this is what stops the
    per-pose Amber files (prmtops, trajectories) accumulating into gigabytes.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    out = {"pb_dG": float("nan"), "gb_dG": float("nan"), "net_charge": None, "error": None}
    try:
        rec, lig = split_complex(complex_lines)
        ss = write_receptor_pdb(rec, os.path.join(workdir, "rec.pdb"))
        mol = ligand_mol_from_pose(lig, smiles, ph=ph)
        mol2, frcmod, net_q = prepare_ligand_params(mol, workdir, cache_dir=cache_dir)
        out["net_charge"] = net_q
        tops = build_topology(os.path.join(workdir, "rec.pdb"), mol2, frcmod, ss,
                              workdir, igb=igb)
        rst = minimise(tops, os.path.join(workdir, "rec.pdb"), workdir, igb=igb,
                       saltcon=saltcon, maxcyc=maxcyc)
        if "gb" in methods:
            g = mmgbsa_energy(tops, rst, workdir, igb=igb, saltcon=saltcon, method="gb")
            out["gb_dG"] = g.get("dG_total", float("nan"))
            for k in ("dE_vdW", "dE_elec", "dG_polar", "dG_nonpolar"):
                out[f"gb_{k}"] = g.get(k)
        if "pb" in methods:
            p = mmgbsa_energy(tops, rst, workdir, intdiel=indi, saltcon=saltcon, method="pb")
            out["pb_dG"] = p.get("dG_total", float("nan"))
            for k in ("dE_vdW", "dE_elec", "dG_polar", "dG_nonpolar"):
                out[f"pb_{k}"] = p.get(k)
    except Exception as exc:                      # noqa: BLE001 - failure contract
        out["error"] = str(exc)[:220]
    finally:
        if not keep_scratch:
            shutil.rmtree(workdir, ignore_errors=True)
    return out


def score_ensemble(ensemble_pdb: str, smiles: str, workdir: str,
                   cache_dir: str | None = None, igb: int = 8,
                   saltcon: float = 0.15, maxcyc: int = 250,
                   max_poses: int | None = None, n_jobs: int = 1) -> "list[dict]":
    """MM/GBSA for every MODEL in an ensemble -> list of dicts with `pose_rank`.

    Poses are independent, so `n_jobs` > 1 parallelises across them (the per-pose
    minimisation is single-threaded, so this is the effective speed-up).
    """
    models = list(iter_models(ensemble_pdb))
    if max_poses:
        models = models[:max_poses]
    jobs = [(num, lines, os.path.join(workdir, f"pose{num}")) for num, lines in models]

    def _one(job):
        num, lines, wd = job
        res = score_pose(lines, smiles, wd, cache_dir=cache_dir, igb=igb,
                         saltcon=saltcon, maxcyc=maxcyc)
        res["pose_rank"] = num
        return res

    if n_jobs and n_jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor   # work is in subprocesses
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            out = list(ex.map(_one, jobs))
    else:
        out = [_one(j) for j in jobs]
    return sorted(out, key=lambda d: d["pose_rank"])


def rescore_ensemble(ensemble_pdb: str, smiles: str, workdir: str, *,
                     ph: "float | None" = 7.4, igb: int = 8, indi: float = 2.0,
                     maxcyc: int = 250, cache_dir: str | None = None,
                     max_poses: "int | None" = None, n_jobs: int = 8) -> "list[dict]":
    """Rescore every MODEL of a screening ensemble (full receptor+ligand complex per
    MODEL) with the production protocol; return per-pose dicts sorted by pose rank.

    Used by the prospective screen (run_screen.py --rescore): the ensemble already IS
    the complex, so no separate receptor file is needed, and the SMILES comes straight
    from the screening library. Scratch is auto-cleaned per pose. `max_poses` scores only
    the top-K poses (poses are ordered best-Vina-score first) for a faster ranking.
    """
    from concurrent.futures import ThreadPoolExecutor
    models = list(iter_models(ensemble_pdb))
    if max_poses:
        models = models[:max_poses]
    if not models:
        return []
    # warm the AM1-BCC cache once at the correct protonation state
    try:
        lig0 = [l for l in models[0][1] if l.startswith("HETATM") and l[17:20].strip() != "HOH"]
        if lig0:
            prepare_ligand_params(ligand_mol_from_pose(lig0, smiles, ph=ph),
                                  os.path.join(workdir, "_warm"), cache_dir=cache_dir)
    except Exception:
        pass

    def _one(num_lines):
        num, lines = num_lines
        res = rescore_pose([l for l in lines if l.startswith(("ATOM", "HETATM"))],
                           smiles, os.path.join(workdir, f"pose{num}"),
                           ph=ph, igb=igb, indi=indi, maxcyc=maxcyc, cache_dir=cache_dir)
        res["pose_rank"] = num
        return res

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        out = list(ex.map(_one, models))
    return sorted(out, key=lambda d: d["pose_rank"])


def best_pose_by_mmgbsa(results: "list[dict]") -> "tuple[int, float]":
    """(pose_rank, dG) of the best-scoring pose; (-1, nan) if none scored."""
    ok = [r for r in results if r.get("dG_total") == r.get("dG_total")]   # drop NaN
    if not ok:
        return -1, float("nan")
    best = min(ok, key=lambda r: r["dG_total"])
    return best["pose_rank"], best["dG_total"]


_CCD_SMILES_CACHE: "dict[str, str | None]" = {}


def ccd_smiles(code: str) -> "str | None":
    """Canonical SMILES for a 3-letter ligand code from the RCSB Chemical Component
    Dictionary (cached, offline-safe). Used to parameterise a *validation native*
    ligand when only its PDB code is known; novel screening supplies SMILES directly.
    Returns None on any failure (offline, unknown code) so callers degrade gracefully.
    """
    import json
    import urllib.request

    code = (code or "").strip().upper()
    if not code:
        return None
    if code in _CCD_SMILES_CACHE:
        return _CCD_SMILES_CACHE[code]
    smi = None
    try:
        d = json.load(urllib.request.urlopen(
            f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=30))
        ds = d.get("pdbx_chem_comp_descriptor", [])
        cands = ([x["descriptor"] for x in ds if x.get("type") == "SMILES_CANONICAL"]
                 or [x["descriptor"] for x in ds if x.get("type") == "SMILES"])
        smi = cands[0] if cands else None
    except Exception:
        smi = None
    _CCD_SMILES_CACHE[code] = smi
    return smi
