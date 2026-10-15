#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from vina import Vina
from plip.structure.preparation import PDBComplex

import hydrated
import validation  # shared publication-grade PyMOL render settings


def embed_and_minimize(mol: "Chem.Mol", seed: int, n_confs: int = 10) -> "Chem.Mol | None":
    """Embed a 3D structure for an isomer with ETKDGv3, then keep the best conformer.

    ETKDGv3 applies knowledge-based ring puckering and experimental-torsion
    preferences, giving more realistic geometries than the bare distance-geometry
    embedder (whose `useExpTorsionAnglePrefs`/`useBasicKnowledge` terms are off by
    default). A handful of conformers are generated and force-field minimised, and
    the single lowest-energy one is retained, so the screen does not commit to one
    poor embedding. Deterministic for a fixed `seed`. Returns a single-conformer mol
    with explicit Hs, or None if the molecule cannot be embedded at all.
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not cids:  # last resort: random starting coordinates (cage-collapse cases)
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not cids:
        return None

    # Minimise every conformer; prefer MMFF94, fall back to UFF when MMFF lacks
    # parameters for an atom type (otherwise the energies come back non-finite).
    energies = None
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000)
        cand = [e for (_flag, e) in res]
        if cand and all(e == e for e in cand):  # reject NaNs
            energies = cand
    except Exception:
        energies = None
    if energies is None:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)
            energies = [e for (_flag, e) in res]
        except Exception:
            energies = [0.0] * len(cids)  # un-minimisable: keep the first embedding

    best_cid = cids[min(range(len(cids)), key=lambda i: energies[i])]
    for cid in list(cids):
        if cid != best_cid:
            mol.RemoveConformer(cid)
    mol.GetConformer(best_cid).SetId(0)
    return mol


def write_vina_ligand_pdbqt(mol: "Chem.Mol", out_path: str) -> bool:
    """Write a Vina-ready ligand PDBQT straight from the 3D RDKit `mol` with Meeko.

    Preserves the bond orders, formal charges and stereochemistry that the
    enumeration step established — the legacy obabel PDB round-trip discarded them
    and re-perceived from coordinates (the same lossy perception that NaNs exotic
    nitrogens elsewhere) — and lets the AutoDock reference tool build the torsion
    tree and atom types. Standard Vina scoring is charge-independent, so the ligand
    partial charges are immaterial here. Returns True on success, False if Meeko
    declines the molecule (the caller then uses the obabel fallback).
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        setups = MoleculePreparation().prepare(mol)
        if not setups:
            return False
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok or not pdbqt_str.strip():
            return False
        with open(out_path, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as _e:
        print(f"   ↳ ⚠️ Meeko ligand prep failed ({_e}); using OpenBabel fallback.")
        return False

def split_and_merge(receptor_pdb: str, docked_pdbqt: str, name: str, output_dir: str,
                    hydrated_ctx: "dict | None" = None) -> list:
    """Merge each docked ligand pose with the receptor into a PLIP-ready complex PDB.

    Splits the multi-MODEL docked PDBQT into individual poses, converts each to PDB
    (obabel), and writes one `{name}_pose{rank}_complex.pdb` per pose containing the
    receptor ATOMs followed by the ligand relabelled as HETATM (see below). Returns a
    list of `(pose_rank, complex_pdb_path)` tuples for downstream PLIP profiling.

    hydrated_ctx (hydrated/AD4 mode only): {"receptor_pdbqt":..., "w_map":...}.
    When provided, each pose is run through dry.py to drop displaced waters and
    keep conserved (STRONG/WEAK) ones, which are merged into the complex as HOH
    so PLIP can profile water bridges. When None, behaves exactly as the Vina path.
    """
    complex_files = []
    receptor_lines = []
    with open(receptor_pdb, 'r') as r_file:
        for line in r_file:
            # THE FIX: Only import the amino acid ATOMs.
            # This completely drops the old native ligand, waters, and buffers.
            if line.startswith("ATOM  "):
                receptor_lines.append(line)

    with open(docked_pdbqt, 'r') as d_file:
        lines = d_file.readlines()

    models = []
    current_model = []
    for line in lines:
        if line.startswith("MODEL"):
            current_model = []
        elif line.startswith("ENDMDL"):
            models.append(current_model)
        else:
            current_model.append(line)

    for i, model in enumerate(models):
        pose_rank = i + 1
        pose_pdbqt = os.path.join(output_dir, f"{name}_temp_pose{pose_rank}.pdbqt")
        pose_pdb = os.path.join(output_dir, f"{name}_temp_pose{pose_rank}.pdb")
        complex_pdb = os.path.join(output_dir, f"{name}_pose{pose_rank}_complex.pdb")

        water_pdb_lines = []
        if hydrated_ctx:
            # Try to strip displaced waters for this pose; on any dry.py failure,
            # fall back to the raw pose with its W (water) atoms removed.
            dried = hydrated.dry_pose(docked_pdbqt, pose_rank,
                                      hydrated_ctx["receptor_pdbqt"],
                                      hydrated_ctx["w_map"], output_dir)
            if dried:
                ligand_lines, water_pdb_lines = hydrated.split_ligand_and_waters(dried)
                os.remove(dried)
            else:
                ligand_lines = [ln for ln in model
                                if not hydrated.is_water_pdbqt_line(ln)]
            with open(pose_pdbqt, 'w') as out:
                out.writelines(ligand_lines)
        else:
            with open(pose_pdbqt, 'w') as out:
                out.writelines(model)

        subprocess.run(["obabel", pose_pdbqt, "-O", pose_pdb], capture_output=True)

        with open(complex_pdb, 'w') as out:
            out.writelines(receptor_lines)
            with open(pose_pdb, 'r') as p_file:
                for line in p_file:
                    if not line.strip().startswith(("END", "MASTER", "MODEL", "ENDMDL")):
                        # THE HETATM RULE: obabel writes the docked ligand as ATOM
                        # records, but the receptor above is also ATOM records. Relabel
                        # the *ligand's* ATOM lines to HETATM so PLIP treats it as the
                        # small-molecule ligand (not part of the protein chain) and
                        # profiles protein-ligand contacts against it. Only the record
                        # name (cols 1-6) changes; the rest of the line is preserved.
                        if line.startswith("ATOM  "):
                            line = "HETATM" + line[6:]
                        out.write(line)
            # Conserved waters (hydrated mode) as HOH HETATM records for PLIP.
            out.writelines(water_pdb_lines)

        os.remove(pose_pdbqt)
        os.remove(pose_pdb)
        complex_files.append((pose_rank, complex_pdb))

    return complex_files

def generate_pymol_script(pml_path: str, pdb_filename: str) -> None:
    """Write a per-variation PyMOL `.pml` that renders a publication figure for a pose.

    Loads the ligand ensemble, styles receptor cartoon + ligand/pocket sticks + polar
    contacts using the shared `validation.publication_settings()` block, and ends with
    ray-trace → 300-DPI PNG → editable `.pse` (render lines commented for view tweaks).
    """
    # Publication figure name derived from the script name ({var}_view.pml).
    base = os.path.basename(pml_path).replace("_view.pml", "")
    with open(pml_path, "w") as f:
        f.write(f"load {pdb_filename}, target_complex\n\n")
        f.write("hide everything\n")
        f.write(validation.publication_settings() + "\n")

        # Receptor (soft grey cartoon for context).
        f.write("show cartoon, target_complex\ncolor gray80, target_complex\n\n")

        # Ligand (orange sticks, CPK heteroatoms).
        f.write("select ligand, organic\nshow sticks, ligand\ncolor orange, ligand\nutil.cnc ligand\n\n")

        # Interacting pocket (cyan side chains).
        f.write("select pocket, byres (target_complex within 4.0 of ligand) and not ligand\n")
        f.write("show sticks, pocket and sidechain\ncolor cyan, pocket\nutil.cnc pocket\n")
        f.write("set stick_radius, 0.11, pocket\n\n")

        # Heavy-atom polar contacts (Vina may hide hydrogens, so use a 3.5 Å cutoff).
        f.write("distance all_contacts, ligand and not hydro, pocket and not hydro, 3.5, mode=0\n")
        f.write("set dash_color, marine, all_contacts\nset dash_gap, 0.3, all_contacts\n")
        f.write("set dash_width, 2.5, all_contacts\nset dash_labels, on, all_contacts\n\n")

        # Camera + render to a publication PNG (comment out to adjust the view first).
        f.write("orient ligand\nzoom ligand, 4\n\n")
        f.write("# --- render (comment out the next three lines to tweak the view first) ---\n")
        f.write("ray 2400, 1800\n")
        f.write(f"png {base}_figure.png, dpi=300\n")
        f.write(f"save {base}_session.pse\n")

def prepare_and_dock(mode: str, mol: "Chem.Mol", receptor_pdbqt: str, map_prefix: "str | None",
                     center: list, box_size: list, seed: int, exhaustiveness: int, n_poses: int,
                     temp_pdb: str, temp_pdbqt: str, docked_pdbqt: str) -> list:
    """Prepare a 3D RDKit mol as a ligand PDBQT and dock it; return Vina energies.

    vina mode     -> Gasteiger PDBQT (obabel) + Vina scoring + compute_vina_maps.
    hydrated mode -> Meeko hydrated PDBQT (explicit W atoms) + AD4 scoring +
                     load_maps (the precomputed autogrid4 grids incl. the W map).
    """
    if mode == "hydrated":
        hydrated.prepare_hydrated_ligand_pdbqt(mol, temp_pdbqt)
        v = Vina(sf_name='ad4', cpu=args.cpu, seed=seed)
        v.load_maps(map_prefix)
        v.set_ligand_from_file(temp_pdbqt)
    else:
        if not write_vina_ligand_pdbqt(mol, temp_pdbqt):
            # Fallback: legacy OpenBabel route (PDB round-trip + Gasteiger).
            Chem.MolToPDBFile(mol, temp_pdb)
            subprocess.run(["obabel", temp_pdb, "-O", temp_pdbqt, "--partialcharge", "gasteiger"], capture_output=True)
        v = Vina(sf_name='vina', cpu=args.cpu, seed=seed)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(temp_pdbqt)
        v.compute_vina_maps(center=center, box_size=box_size)

    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    v.write_poses(docked_pdbqt, n_poses=n_poses, overwrite=True)
    return v.energies(n_poses=n_poses)

def dock_and_profile(name: str, smiles: str, receptor_pdbqt: str, receptor_pdb: str,
                     center: list, box_size: list, output_dir: str, seed: int,
                     exhaustiveness: int, n_poses: int, mode: str = "vina",
                     map_prefix: "str | None" = None) -> "list | None":
    """Protonate, enumerate, dock and PLIP-profile one library compound end to end.

    Runs the full per-compound pipeline: obabel pH-7.4 protonation → tautomer/stereo
    enumeration with thermodynamic filtering → 3D embed → dock (Vina or hydrated AD4)
    → merge each pose with the receptor → PLIP interaction profiling. Returns a list
    of per-pose result dicts (Rank/Energy/Num_Contacts/Contacts/Winning_SMILES/
    Variation), or None if the compound fails outright. On combinatorial cage collapse
    (all isomers fail 3D embedding) it falls back to a single native-geometry embed.
    """
    print(f"\n🚀 [{name}] Starting processing...")
    results = []

    # Hydrated mode merges conserved waters into each complex for PLIP profiling.
    hydrated_ctx = None
    if mode == "hydrated":
        hydrated_ctx = {"receptor_pdbqt": f"{map_prefix}.pdbqt",
                        "w_map": f"{map_prefix}.W.map"}

    try:
        # --- 1D PROTONATION BRIDGE ---
        ob_protonate = subprocess.run(
            ["obabel", f"-:{smiles}", "-osmi", "-p", "7.4"], 
            capture_output=True, text=True
        )
        protonated_smiles = smiles
        if ob_protonate.returncode == 0 and ob_protonate.stdout.strip():
            protonated_smiles = ob_protonate.stdout.split()[0].strip()

        base_mol = Chem.MolFromSmiles(protonated_smiles)
        if base_mol is None:
            base_mol = Chem.MolFromSmiles(smiles)

        # --- THERMODYNAMIC STEREO/TAUTOMER FILTERING ---
        # WHY: a SMILES underspecifies tautomeric and stereochemical state, and each
        # combination is a distinct 3D molecule that could dock differently. Rather
        # than dock every combination (a combinatorial explosion), enumerate broadly
        # then keep only the physically most plausible variants, ranked by stability.
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(50) # Expand the net to avoid missing anything
        tautomers = enumerator.Enumerate(base_mol)

        # Score and sort purely by thermodynamic stability (higher score = more stable)
        scored_tautomers = [(taut, enumerator.ScoreTautomer(taut)) for taut in tautomers]
        scored_tautomers.sort(key=lambda x: x[1], reverse=True)

        # Apply the Thermodynamic Filter: Keep only the top 2 most stable tautomers
        # (the dominant forms at equilibrium) so we don't waste docking on rare ones.
        top_tautomers = [t[0] for t in scored_tautomers[:2]]

        # For each surviving tautomer, expand undefined stereocentres into concrete
        # isomers (capped per-tautomer at 4 unique) — Vina needs a defined 3D geometry.
        opts = StereoEnumerationOptions(unique=True, maxIsomers=4)
        candidate_pool = []
        for taut in top_tautomers:
            isos = list(EnumerateStereoisomers(taut, options=opts))
            candidate_pool.extend(isos)

        # Final safety cap (now prioritized by physical stability, not random chance):
        # the pool is already stability-ordered, so slicing keeps the best 8 variants.
        if len(candidate_pool) > 8:
            candidate_pool = candidate_pool[:8]

        if len(candidate_pool) > 1:
            print(f"   ↳ ⚖️ Thermodynamic Filter Active: Evaluated tautomers. Proceeding with top {len(candidate_pool)} stable variations...")
            
        isomers = candidate_pool

        # --- DOCK AND PROFILE ALL VALID VARIATIONS ---
        # Each variation is docked independently inside its own try/except so that a
        # single bad isomer (un-embeddable geometry, Vina/PLIP hiccup, IO error) is
        # skipped with a warning and `continue`d past — it must never abort the whole
        # compound. `results` accumulates across variations; the fallback below fires
        # only if EVERY variation failed and left `results` empty.
        for idx, iso_mol in enumerate(isomers):
            try:
                var_name = f"{name}_var{idx+1}"
            
                # No more subfolders. Everything goes straight into output_dir
                ensemble_filename = f"{var_name}_Ensemble.pdb"
                compound_ensemble_pdb = os.path.join(output_dir, ensemble_filename)
            
                with open(compound_ensemble_pdb, 'w') as f:
                    pass 
            
                iso_temp_pdb = os.path.join(output_dir, f"{var_name}_temp.pdb")
                iso_temp_pdbqt = os.path.join(output_dir, f"{var_name}_temp.pdbqt")
                iso_docked_pdbqt = os.path.join(output_dir, f"{var_name}_docked.pdbqt")

                iso_mol = embed_and_minimize(iso_mol, seed)
                if iso_mol is None:
                    print(f"   ↳ ⚠️ Skipping {var_name}: Physically impossible 3D geometry.")
                    continue

                energies = prepare_and_dock(mode, iso_mol, receptor_pdbqt, map_prefix,
                                            center, box_size, seed, exhaustiveness, n_poses,
                                            iso_temp_pdb, iso_temp_pdbqt, iso_docked_pdbqt)
                winning_smiles = Chem.MolToSmiles(iso_mol, isomericSmiles=True)

                complex_files = split_and_merge(receptor_pdb, iso_docked_pdbqt, var_name, output_dir, hydrated_ctx)

                for pose_rank, complex_pdb in complex_files:
                    # Guard: Vina may return fewer energy rows than poses written
                    # (e.g. after per-pose water drying), so bounds-check before
                    # indexing `energies` to avoid an IndexError on the extra poses.
                    if pose_rank - 1 >= len(energies):
                        continue
                    energy_val = energies[pose_rank - 1][0]

                    my_complex = PDBComplex()
                    my_complex.load_pdb(complex_pdb)
                
                    contact_list = [] 
                
                    for ligand in my_complex.ligands:
                        my_complex.characterize_complex(ligand)
                        key = f"{ligand.hetid}:{ligand.chain}:{ligand.position}"
                        interactions = my_complex.interaction_sets[key]
                    
                        # PLIP INTERACTION DISCOVERY (reflective scan). Preferred path:
                        # PLIP's aggregated `all_itypes` list. But that attribute is
                        # NOT reliably populated across PLIP versions/complexes, so when
                        # it comes back empty we fall back to reflection: walk every
                        # public attribute of the interaction set and collect any that
                        # is a non-empty list whose elements look like interaction
                        # objects (they carry a `.restype`). This is intentional
                        # defensive code, not dead code — it finds the per-type lists
                        # (hbonds, hydrophobic_contacts, ...) without hard-coding names.
                        all_items = getattr(interactions, 'all_itypes', [])
                        if not all_items:
                            for attr_name in dir(interactions):
                                if attr_name.startswith('_') or attr_name == 'all_itypes': continue
                                attr_val = getattr(interactions, attr_name)
                                if isinstance(attr_val, list) and len(attr_val) > 0 and hasattr(attr_val[0], 'restype'):
                                    all_items.extend(attr_val)

                        for item in all_items:
                            if hasattr(item, 'restype') and hasattr(item, 'resnr'):
                                cls_name = item.__class__.__name__.lower()
                                i_type = "Contact"
                                if 'hbond' in cls_name: i_type = "H-Bond"
                                elif 'hydro' in cls_name: i_type = "Hydro"
                                elif 'stack' in cls_name: i_type = "Pi-Stack"
                                elif 'salt' in cls_name: i_type = "SaltBridge"
                                elif 'cation' in cls_name: i_type = "Pi-Cat"
                                elif 'halo' in cls_name: i_type = "Halogen"
                                elif 'water' in cls_name: i_type = "Water"
                                elif 'metal' in cls_name: i_type = "Metal"
                            
                                dist = getattr(item, 'distance_ad', getattr(item, 'distance_aw', getattr(item, 'distance', 0.0)))
                                contact_list.append(f"{i_type}:{item.restype}{item.resnr}({dist:.2f}Å)")
                
                    total_contacts = len(contact_list)
                    contact_summary = " | ".join(contact_list) if contact_list else "No major contacts"
                
                    results.append({
                        "Rank": pose_rank,
                        "Energy": round(energy_val, 2),
                        "Num_Contacts": total_contacts,
                        "Contacts": contact_summary,
                        "Winning_SMILES": winning_smiles,
                        "Variation": var_name 
                    })
                
                    with open(compound_ensemble_pdb, 'a') as m_file:
                        m_file.write(f"MODEL {pose_rank}\n")
                        with open(complex_pdb, 'r') as c_file:
                            m_file.writelines(c_file.readlines())
                        m_file.write("ENDMDL\n")
                
                    os.remove(complex_pdb) 

                for f in (iso_temp_pdb, iso_temp_pdbqt, iso_docked_pdbqt):
                    if os.path.exists(f):
                        os.remove(f)

                # Generate individual PyMOL script in the flattened directory
                pml_path = os.path.join(output_dir, f"{var_name}_view.pml")
                generate_pymol_script(pml_path, ensemble_filename)

            except Exception as _ve:
                print(f"   ↳ ⚠️  Skipping {name}_var{idx+1}: {_ve}")
                continue
        if not results:
            print(f"   ↳ ⚠️ Combinatorial cage collapse detected! All isomers failed 3D embedding.")
            print(f"   ↳ 🔄 Falling back to native distance geometry for {name}...")
            
            var_name = f"{name}_fallback"
            
            # FLATTENED DIRECTORY for fallback
            ensemble_filename = f"{var_name}_Ensemble.pdb"
            compound_ensemble_pdb = os.path.join(output_dir, ensemble_filename)
            
            with open(compound_ensemble_pdb, 'w') as f:
                pass

            fallback_mol = embed_and_minimize(base_mol, seed)
            if fallback_mol is None:
                print(f"   ↳ ❌ CRITICAL: Molecule {name} is completely un-embeddable in 3D space.")
                return None

            fallback_temp_pdb = os.path.join(output_dir, f"{var_name}_temp.pdb")
            fallback_temp_pdbqt = os.path.join(output_dir, f"{var_name}_temp.pdbqt")
            fallback_docked_pdbqt = os.path.join(output_dir, f"{var_name}_docked.pdbqt")

            energies = prepare_and_dock(mode, fallback_mol, receptor_pdbqt, map_prefix,
                                        center, box_size, seed, exhaustiveness, n_poses,
                                        fallback_temp_pdb, fallback_temp_pdbqt, fallback_docked_pdbqt)
            winning_smiles = Chem.MolToSmiles(fallback_mol, isomericSmiles=True)

            complex_files = split_and_merge(receptor_pdb, fallback_docked_pdbqt, var_name, output_dir, hydrated_ctx)
            
            for pose_rank, complex_pdb in complex_files:
                # Same bounds guard as the main loop: skip poses with no energy row.
                if pose_rank - 1 >= len(energies):
                    continue
                energy_val = energies[pose_rank - 1][0]
                my_complex = PDBComplex()
                my_complex.load_pdb(complex_pdb)

                contact_list = []
                for ligand in my_complex.ligands:
                    my_complex.characterize_complex(ligand)
                    key = f"{ligand.hetid}:{ligand.chain}:{ligand.position}"
                    interactions = my_complex.interaction_sets[key]
                    # Same reflective PLIP interaction discovery as the main loop:
                    # prefer `all_itypes`, else scan attributes for lists of objects
                    # carrying `.restype` (see the main-loop comment for the rationale).
                    all_items = getattr(interactions, 'all_itypes', [])
                    if not all_items:
                        for attr_name in dir(interactions):
                            if attr_name.startswith('_') or attr_name == 'all_itypes': continue
                            attr_val = getattr(interactions, attr_name)
                            if isinstance(attr_val, list) and len(attr_val) > 0 and hasattr(attr_val[0], 'restype'):
                                all_items.extend(attr_val)
                    for item in all_items:
                        if hasattr(item, 'restype') and hasattr(item, 'resnr'):
                            cls_name = item.__class__.__name__.lower()
                            i_type = "Contact"
                            if 'hbond' in cls_name: i_type = "H-Bond"
                            elif 'hydro' in cls_name: i_type = "Hydro"
                            elif 'stack' in cls_name: i_type = "Pi-Stack"
                            elif 'salt' in cls_name: i_type = "SaltBridge"
                            elif 'cation' in cls_name: i_type = "Pi-Cat"
                            elif 'halo' in cls_name: i_type = "Halogen"
                            
                            dist = getattr(item, 'distance_ad', getattr(item, 'distance_aw', getattr(item, 'distance', 0.0)))
                            contact_list.append(f"{i_type}:{item.restype}{item.resnr}({dist:.2f}Å)")
                
                contact_summary = " | ".join(contact_list) if contact_list else "No major contacts"
                results.append({
                    "Rank": pose_rank, "Energy": round(energy_val, 2), "Num_Contacts": len(contact_list),
                    "Contacts": contact_summary, "Winning_SMILES": winning_smiles, "Variation": var_name
                })
                
                with open(compound_ensemble_pdb, 'a') as m_file:
                    m_file.write(f"MODEL {pose_rank}\n")
                    with open(complex_pdb, 'r') as c_file: m_file.writelines(c_file.readlines())
                    m_file.write("ENDMDL\n")
                os.remove(complex_pdb)
                
            for f in (fallback_temp_pdb, fallback_temp_pdbqt, fallback_docked_pdbqt):
                if os.path.exists(f):
                    os.remove(f)

            pml_path = os.path.join(output_dir, f"{var_name}_view.pml")
            generate_pymol_script(pml_path, ensemble_filename)

        return results
    except Exception as e:
        print(f"❌ [{name}] ERROR: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target-Agnostic GPCR Docking & Profiling Pipeline")
    parser.add_argument("--receptor_pdb", required=True)
    parser.add_argument("--receptor_pdbqt", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--box_x", type=float, default=25.0)
    parser.add_argument("--box_y", type=float, default=25.0)
    parser.add_argument("--box_z", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--n_poses", type=int, default=9)
    parser.add_argument("--cpu", type=int, default=10)
    parser.add_argument("--mode", choices=["vina", "hydrated"], default="vina",
                        help="vina = standard Vina scoring; hydrated = AD4 water model")
    parser.add_argument("--ph", default="7.4",
                        help="PROPKA pH for receptor protonation (hydrated AD4 maps); also the "
                             "ligand protonation pH used by --rescore")
    parser.add_argument("--rescore", action="store_true",
                        help="after docking, rescore each ensemble with MM/PBSA + MM/GBSA "
                             "(AmberTools) into a companion {rec}_MMPBSA.csv (slow)")
    parser.add_argument("--rescore_indi", type=float, default=2.0,
                        help="PB solute dielectric for --rescore (default 2.0)")
    parser.add_argument("--rescore_jobs", type=int, default=8,
                        help="poses scored in parallel during --rescore")
    args = parser.parse_args()

    CENTER = [args.x, args.y, args.z]
    BOX = [args.box_x, args.box_y, args.box_z]
    SCORING_FUNCTION = "ad4" if args.mode == "hydrated" else "vina"

    raw_name = os.path.splitext(os.path.basename(args.receptor_pdb))[0]
    rec_name = raw_name.replace("_clean", "")
    output_dir = f"{rec_name}_docking_results"
    os.makedirs(output_dir, exist_ok=True)

    # Hydrated mode: precompute the AD4 affinity grids + W map ONCE (the box is
    # fixed for the whole screen). This is the heavy, receptor-level setup that
    # the on-the-fly Vina path never needs.
    map_prefix = None
    if args.mode == "hydrated":
        print("🌊 Hydrated docking (AD4 water model) selected.", flush=True)
        print("   ↳ Building autogrid4 affinity maps + water (W) map...", flush=True)
        try:
            map_prefix, _ = hydrated.build_ad4_maps(
                args.receptor_pdb, rec_name, CENTER, BOX,
                os.path.join(output_dir, "ad4_maps"), args.ph)
            print(f"   ↳ ✅ AD4 maps ready: {map_prefix}.*.map", flush=True)
        except hydrated.HydratedDockingError as e:
            print(f"❌ ERROR: Hydrated map generation failed.\n{e}", flush=True)
            sys.exit(1)

    csv_filename = f"{rec_name}_Screen_Results.csv"

    with open(args.library, "r") as f:
        library = json.load(f)

    # --- NEW: Track ALL variations, not just Rank 1 ---
    master_variations = []

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # CSV header is a fixed data-flow contract consumed by app.py Tab 2 — the exact
        # column names and order must not change (analytics/fingerprint/graph parse them).
        writer.writerow(["Receptor", "Compound_Name", "SMILES", "Pose_Rank", "Binding_Energy_kcal_mol", "Total_Contacts", "PLIP_Contacts_with_Distances", "Scoring_Function"])

        total_compounds = len(library)
        current_idx = 0

        for name, smiles in library.items():
            current_idx += 1
            pose_results = dock_and_profile(
                name, smiles, args.receptor_pdbqt, args.receptor_pdb,
                CENTER, BOX, output_dir,
                args.seed, args.exhaustiveness, args.n_poses,
                mode=args.mode, map_prefix=map_prefix
            )

            if pose_results:
                for pose in pose_results:
                    writer.writerow([rec_name, pose["Variation"], pose["Winning_SMILES"], pose["Rank"], pose["Energy"], pose["Num_Contacts"], pose["Contacts"], SCORING_FUNCTION])
                    
                    if pose["Variation"] not in master_variations:
                        master_variations.append(pose["Variation"])
                file.flush()
            
            # --- EMIT PROGRESS TAG ---
            # IPC contract with app.py: it parses this exact `__PROGRESS__{cur}__{tot}__`
            # token from stdout to drive the progress bar/ETA. Do not change the format.
            print(f"__PROGRESS__{current_idx}__{total_compounds}__", flush=True)

   # --- GENERATE MULTI-STATE MASTER PYMOL OVERLAY (ABSOLUTE PATHS) ---
    master_pml_path = os.path.join(output_dir, f"{rec_name}_Master_Overlay.pml")
    
    # Get the absolute path to the receptor
    abs_receptor = os.path.abspath(args.receptor_pdb)
    
    colors = ["cyan", "magenta", "yellow", "green", "orange", "purple", "pink", "teal", "salmon", "limon"]
    
    with open(master_pml_path, "w") as f:
        f.write("bg_color white\n")
        f.write("hide everything\n")
        
        # Load the Receptor using the absolute path
        f.write(f"load {abs_receptor}, Receptor\n")
        f.write("show cartoon, Receptor\n")
        f.write("color gray80, Receptor\n")
        f.write("set cartoon_transparency, 0.2\n\n")
        
        for idx, var in enumerate(master_variations):
            color = colors[idx % len(colors)]
            
            # Construct absolute paths
            ensemble_file = os.path.join(output_dir, f"{var}_Ensemble.pdb")
            abs_ensemble = os.path.abspath(ensemble_file)
            
            f.write(f"# --- Load Multi-State Object for {var} ---\n")
            f.write(f"load {abs_ensemble}, {var}_raw\n")
            
            # Extract the ligand as a single object containing all 9 states/poses
            f.write(f"extract {var}, {var}_raw and organic\n")
            f.write(f"delete {var}_raw\n")
            
            f.write(f"show sticks, {var}\n")
            f.write(f"color {color}, {var}\n")
            f.write(f"util.cnc {var}\n") 
            
            # Disable by default, except the first one
            if idx == 0:
                f.write(f"enable {var}\n")
            else:
                f.write(f"disable {var}\n")
            f.write("\n")
            
        f.write("set depth_cue, 1\n")
        f.write("set ray_trace_mode, 1\n")
        f.write("center organic\n")
        f.write("zoom organic, 5\n")
        
        # Ensure PyMOL only shows one state at a time so the animation works
        f.write("set all_states, 0\n")

    print(f"\n✨ Screen Complete! Master PyMOL Overlay created: {master_pml_path}")

    # --- OPTIONAL: end-point (MM/PBSA + MM/GBSA) rescoring of each ensemble ---
    # Written to a SEPARATE companion CSV so the Screen_Results.csv schema (a documented
    # app.py Tab-2 contract) is untouched. The library SMILES is used directly (no CCD),
    # protonated to --ph so the rescored species matches what was docked.
    if getattr(args, "rescore", False):
        import mmpbsa
        rescore_csv = f"{rec_name}_MMPBSA.csv"
        scratch = os.path.join(output_dir, "_mmpbsa_scratch")
        print(f"\n🔬 Rescoring {len(master_variations)} variations with MM/PBSA "
              f"(indi={args.rescore_indi}) + MM/GBSA at pH {args.ph} ...", flush=True)
        with open(rescore_csv, "w", newline="") as rf:
            rw = csv.writer(rf)
            rw.writerow(["Receptor", "Compound_Name", "Variation", "Ligand_Net_Charge",
                         "Best_MMPBSA_Pose", "MMPBSA_dG", "Best_MMGBSA_Pose", "MMGBSA_dG",
                         "N_Poses_Scored", "Note"])
            for var in master_variations:
                base = var.split("_var")[0].split("_fallback")[0]
                smi = library.get(base) or library.get(var)
                ens = os.path.join(output_dir, f"{var}_Ensemble.pdb")
                if not smi or not os.path.isfile(ens):
                    rw.writerow([rec_name, base, var, "", "", "", "", "", 0,
                                 "no SMILES" if not smi else "no ensemble"])
                    continue
                try:
                    poses = mmpbsa.rescore_ensemble(
                        ens, smi, os.path.join(scratch, var),
                        ph=float(args.ph), indi=args.rescore_indi,
                        n_jobs=args.rescore_jobs)
                except Exception as exc:
                    rw.writerow([rec_name, base, var, "", "", "", "", "", 0, str(exc)[:80]])
                    continue
                q = next((p.get("net_charge") for p in poses if p.get("net_charge") is not None), "")
                pb = [(p["pose_rank"], p["pb_dG"]) for p in poses if p.get("pb_dG") == p.get("pb_dG")]
                gb = [(p["pose_rank"], p["gb_dG"]) for p in poses if p.get("gb_dG") == p.get("gb_dG")]
                bpb = min(pb, key=lambda t: t[1]) if pb else ("", "")
                bgb = min(gb, key=lambda t: t[1]) if gb else ("", "")
                note = "" if pb else (next((p.get("error") for p in poses if p.get("error")), "no poses") or "no poses")
                rw.writerow([rec_name, base, var, q, bpb[0],
                             round(bpb[1], 2) if pb else "", bgb[0],
                             round(bgb[1], 2) if gb else "", len(pb), note[:80]])
                rf.flush()
                print(f"   {var}: q={q} MM/PBSA {round(bpb[1],2) if pb else '—'} "
                      f"MM/GBSA {round(bgb[1],2) if gb else '—'} kcal/mol", flush=True)
        import shutil as _sh
        _sh.rmtree(scratch, ignore_errors=True)
        print(f"🔬 Rescoring complete → {rescore_csv}")