import os
import sys
# Make sibling modules importable when launched via `streamlit run src/app.py`
# (Streamlit does not guarantee the script's directory is on sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import wilcoxon, pearsonr, spearmanr
import py3Dmol
from stmol import showmol
import zipfile
import io
import shutil
import requests
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
from rdkit.Chem import rdFMCS
import time

# New Imports for Validation
from vina import Vina
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import hydrated
import validation
import mmpbsa
import paths
from dock_utils import base_compound, series_of  # shared name-parsing helpers

paths.ensure()
# Transient library handoff to the run_screen.py subprocess (absolute so CWD is irrelevant).
_TEMP_LIBRARY = os.path.join(paths.SCRATCH, "temp_library.json")
# Receptor working files (raw/refined PDB, prepared PDBQT, and preptarget's pqr/log
# intermediates) live here instead of cluttering the repo root; scratch/ is git-ignored.
RECEPTOR_DIR = paths.scratch_dir("receptors")

# PLIP contact-string residue pattern, e.g. "H-Bond:ASN254(2.9Å)" -> "ASN254".
PLIP_RESIDUE_RE = re.compile(r':([A-Z]+\d+)\(')

st.set_page_config(page_title="GPCR Docking Pipeline", page_icon="🧬", layout="wide")

# (Native-ligand extraction lives in validation.extract_native_ligand, used by Tab 1;
#  the former local copy here was dead and has been removed.)

# fix_explosive_charges lives in validation.py (shared with the batch validator);
# re-exported here so the Tab 1 / Tab 4 call sites read naturally.
fix_explosive_charges = validation.fix_explosive_charges


def short_label(structure) -> str:
    """Readable plot/table label from a long GPCRdb structure name -> 'RECEPTOR (PDBID)'."""
    s = str(structure)
    toks = s.split("_")
    def is_pdb(t):
        return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", t)) and bool(re.search(r"[A-Za-z]", t))
    pdb = next((t for t in toks if is_pdb(t)), None)
    rec = None
    for t in toks:
        tl = t.lower()
        if tl.startswith("class") or tl in ("human", "refined") or is_pdb(t):
            continue
        rec = t.split("-")[0]
        break
    if rec and pdb:
        return f"{rec.upper()} ({pdb})"
    return rec.upper() if rec else (pdb or s[:18])


def activation_state(structure) -> str:
    """Infer receptor activation state from a GPCRdb structure name (heuristic)."""
    s = str(structure).lower()
    if "inactive" in s:
        return "Inactive"
    if "intermediate" in s:
        return "Intermediate"
    if "active" in s:
        return "Active"
    # G-protein-coupled complex: Gα subunit token (gnas/gnai1/gnaq/gna12/…). The
    # subtype after "gna" may be a letter (gnaq) OR a digit (gna12), so match either.
    if re.search(r"gna[a-z0-9]", s) or "gprotein" in s or "gproteindb" in s:
        return "Active"
    return "Unknown"
# ==========================================
# SIDEBAR: WORKSPACE MANAGEMENT
# ==========================================
with st.sidebar:
    st.header("🧹 Workspace Management")
    st.markdown("Use this to clear old docking results and start fresh.")
    if st.button("Delete All Previous Results"):
        deleted = False
        for item in os.listdir():
            # Delete output directories
            if (item.endswith("_docking_results") or item.endswith("_validation")) and os.path.isdir(item):
                shutil.rmtree(item)
                deleted = True
            # Delete loose temporary files in the root directory
            elif item.startswith("temp_") or item.startswith("reference_") or item.startswith("validation_"):
                if os.path.isfile(item):
                    os.remove(item)
                    deleted = True
        if deleted:
            st.success("Workspace wiped clean! 3D viewer reset.")
            st.rerun()
        else:
            st.info("Workspace is already clean.")

st.title("🧬 Automated GPCR Docking & Analytics Suite")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Execution Pipeline", "📊 Advanced Analytics Dashboard", "📖 Computational Methodology", "🧪 Validation Protocol", "📐 Batch Validation"])

# ==========================================
# TAB 1: ACQUISITION, CONFIGURATION & EXECUTION
# ==========================================
with tab1:
    col_main, col_settings = st.columns([2, 1])

    with col_settings:
        st.subheader("⚙️ Vina Search Parameters")
        with st.expander("Docking Settings", expanded=True):
            docking_engine = st.radio(
                "Docking Engine",
                ["Standard (Vina)", "Hydrated (AD4 water model)"],
                help=("Standard uses AutoDock Vina scoring. Hydrated attaches explicit "
                      "water molecules to each ligand and scores with the AutoDock4 water "
                      "model (precomputed autogrid4 maps + a W map), keeping conserved "
                      "bridging waters and displacing the rest. More robust for "
                      "water-mediated GPCR binding, but slower (extra map-building step) "
                      "and its binding energies are on a different scale than Vina's — "
                      "do not compare AD4 and Vina scores directly.")
            )
            dock_mode = "hydrated" if "Hydrated" in docking_engine else "vina"
            ph_level = st.number_input("PROPKA pH Level", min_value=1.0, max_value=14.0, value=7.4, step=0.1)
            seed_val = st.number_input("Random Seed", value=42, step=1)
            _ncpu = os.cpu_count() or 8
            cpu_cores = st.slider("CPU Cores (Throttle)", min_value=1, max_value=_ncpu,
                                  value=max(1, _ncpu - 2), step=1)
            exhaustiveness = st.slider("Exhaustiveness", min_value=8, max_value=128, value=32, step=8)
            n_poses = st.slider("Max Poses per Ligand", min_value=1, max_value=20, value=9)
            if dock_mode == "hydrated":
                st.caption("🌊 Hydrated (AD4) mode active — an autogrid4 map-building step runs once before screening.")
            screen_rescore = st.checkbox(
                "🧬 MM/PBSA + MM/GBSA rescoring", value=False,
                help="After docking + PLIP, rescore every compound's pose ensemble with AmberTools "
                     "end-point free energy (MM/PBSA indi=2, MM/GBSA igb=8). The library SMILES is "
                     "protonated to the PROPKA pH, so the electrostatics match the docked chemistry. "
                     "Writes a companion {receptor}_MMPBSA.csv (per-compound best ΔG + selected pose). "
                     "Slow: ~1–2 min per pose — best for a shortlist, not a whole library.")
            if screen_rescore:
                st.caption("⏱️ Adds an AmberTools minimise+score per pose after the screen. Results appear "
                           "as an extra rescoring panel in the Analytics tab.")

    with col_main:
        st.subheader("1. Target Acquisition")
        
        if 'active_pdb' not in st.session_state:
            st.session_state['active_pdb'] = None
            st.session_state['pdb_data'] = None
            st.session_state['receptor_name'] = "Target"
            st.session_state['has_reference_ligand'] = False

        acq_mode = st.radio("Select Source:", ["Fetch Refined from GPCRdb", "Upload Local PDB"], horizontal=True)

        if acq_mode == "Fetch Refined from GPCRdb":
            col_id, col_type = st.columns(2)
            with col_id:
                query_id = st.text_input("ID (e.g., adora1_human or 5n2s)", value="5n2s").lower()
            with col_type:
                struct_type = st.selectbox("Structure Type", ["RCSB Raw (Requires PDB ID)", "GPCRdb AlphaFold Model (Requires UniProt ID)"])
            
            if st.button("⬇️ Fetch Structure"):
                with st.spinner(f"Fetching {query_id.upper()}..."):
                    try:
                        if "RCSB Raw" in struct_type:
                            api_url = f"https://files.rcsb.org/download/{query_id.upper()}.pdb"
                            response = requests.get(api_url)
                            if response.status_code == 200 and "ATOM " in response.text:
                                rec_name = query_id.upper()
                                pdb_filename = os.path.join(RECEPTOR_DIR, f"{rec_name}_raw.pdb")
                                with open(pdb_filename, "wb") as f:
                                    f.write(response.content)
                                st.session_state['active_pdb'] = pdb_filename
                                st.session_state['pdb_data'] = response.text
                                st.session_state['receptor_name'] = rec_name
                                st.success(f"✅ {rec_name} (Raw) fetched successfully from RCSB!")
                            else:
                                st.error("❌ Structure not found. Ensure the 4-letter PDB code is correct.")
                        else:
                            state = st.radio("Activation State:", ["Inactive", "Intermediate", "Active"], horizontal=True)
                            api_url = f"https://gpcrdb.org/services/structure/homology_models/{query_id}_{state.lower()}_model/pdb"
                            response = requests.get(api_url)
                            if response.status_code == 200 and len(response.content) > 5000:
                                rec_name = query_id.split('_')[0].upper()
                                pdb_filename = os.path.join(RECEPTOR_DIR, f"{rec_name}_{state}_refined.pdb")
                                with open(pdb_filename, "wb") as f:
                                    f.write(response.content)
                                st.session_state['active_pdb'] = pdb_filename
                                st.session_state['pdb_data'] = response.text
                                st.session_state['receptor_name'] = rec_name
                                st.success(f"✅ {rec_name} ({state} Model) fetched successfully from GPCRdb!")
                            else:
                                st.error("❌ Model not found. Ensure you are using a UniProt ID.")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

        else:
            rec_name = st.text_input("Name this Receptor", value="A1R")
            uploaded_pdb = st.file_uploader("Upload Local PDB", type=["pdb"])
            if uploaded_pdb:
                pdb_filename = os.path.join(RECEPTOR_DIR, f"{rec_name}_local.pdb")
                pdb_bytes = uploaded_pdb.getbuffer()
                with open(pdb_filename, "wb") as f:
                    f.write(pdb_bytes)
                
                st.session_state['active_pdb'] = pdb_filename
                st.session_state['pdb_data'] = pdb_bytes.tobytes().decode("utf-8")
                st.session_state['receptor_name'] = rec_name
                st.success(f"✅ Loaded local structure: {pdb_filename}")

        if st.session_state['pdb_data']:
            st.divider()
            
            # --- LIVE STRUCTURAL VIEWER ---
            st.subheader("🔬 1.5 Structural Inspection")
            st.markdown("Use this viewer to check for large crystallographic fusion proteins (like BRIL or T4 Lysozyme) inserted into the loops. The protein chain is colored by rainbow (N- to C-terminus), and native heteroatoms (ligands/lipids) are highlighted in magenta.")
            
            col_view1, col_view2 = st.columns([3, 1])
            with col_view1:
                view = py3Dmol.view(width=800, height=400)
                view.addModel(st.session_state['pdb_data'], 'pdb')
                # Rainbow cartoon to easily spot the loops and fusions
                view.setStyle({'cartoon': {'color': 'spectrum'}})
                # Magenta sticks for the native ligand and lipids
                view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.2}})
                view.zoomTo()
                showmol(view, height=400, width=800)
            
            with col_view2:
                st.info("💡 **GPCR Crystallography Tip:**\n\nIf you see a massive barrel-shaped protein attached to the bottom (intracellular side) of your GPCR, that is likely a **BRIL** or **T4L** fusion used to stabilize the crystal.\n\nSince it sits on the intracellular side, it usually won't interfere with orthosteric docking at the extracellular face, but it is critical to visually verify this before proceeding!")
            
            st.divider()
            st.subheader("2. Define Binding Pocket Coordinates")
            
            pocket_mode = st.radio("Grid Box Center Mode:", 
                                   ["Auto-Center on Co-Crystallized Ligand", 
                                    "Auto-Center (Geometric Mean of Protein)", 
                                    "Manual Coordinate Entry"], horizontal=True)
            
            cx, cy, cz = 0.0, 0.0, 0.0
            
            if pocket_mode == "Auto-Center on Co-Crystallized Ligand":
                st.info("💡 The co-crystallised ligand is auto-detected from the structure — pick it below. Its centroid becomes the Vina grid centre and it is saved as the validation reference.")
                cands = validation.candidate_ligand_codes(st.session_state['pdb_data'])
                if cands:
                    _opts = [f"{c}  ({n} atoms)" for c, n in cands] + ["Other (type code)…"]
                    _choice = st.selectbox("Co-crystallised ligand (auto-detected, largest first)", _opts, index=0)
                    lig_code = "" if _choice.startswith("Other") else cands[_opts.index(_choice)][0]
                    if _choice.startswith("Other"):
                        lig_code = st.text_input("Ligand 3-letter code", value="", max_chars=3).strip()
                else:
                    st.warning("No co-crystal ligand auto-detected (only waters / ions / buffers). Type the 3-letter code if you know it.")
                    lig_code = st.text_input("Ligand 3-letter code", value="", max_chars=3).strip()
                
                if lig_code:
                    extracted_pdb, center = validation.extract_native_ligand(st.session_state['pdb_data'], lig_code)
                    if center:
                        cx, cy, cz = center
                        with open("reference_native_ligand.pdb", "w") as f:
                            f.write(extracted_pdb)
                        st.session_state['has_reference_ligand'] = True
                        st.session_state['reference_ligand_code'] = lig_code.upper()
                        st.success(f"✅ Successfully isolated '{lig_code.upper()}'. Perfect coordinates calculated and reference saved for Validation Tab!")
                    else:
                        st.error(f"❌ Could not find HETATM records for '{lig_code.upper()}' in the uploaded PDB.")

            elif pocket_mode == "Auto-Center (Geometric Mean of Protein)":
                x_coords, y_coords, z_coords = [], [], []
                for line in st.session_state['pdb_data'].split('\n'):
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            x_coords.append(float(line[30:38]))
                            y_coords.append(float(line[38:46]))
                            z_coords.append(float(line[46:54]))
                        except ValueError:
                            pass
                if x_coords:
                    cx, cy, cz = sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords), sum(z_coords)/len(z_coords)

            col_x, col_y, col_z = st.columns(3)
            center_x = col_x.number_input("Center X", value=float(cx), format="%.3f")
            center_y = col_y.number_input("Center Y", value=float(cy), format="%.3f")
            center_z = col_z.number_input("Center Z", value=float(cz), format="%.3f")
            
            with st.expander("Grid Box Dimensions (Å)"):
                col_bx, col_by, col_bz = st.columns(3)
                box_x = col_bx.number_input("Box Size X", value=25.0, step=1.0)
                box_y = col_by.number_input("Box Size Y", value=25.0, step=1.0)
                box_z = col_bz.number_input("Box Size Z", value=25.0, step=1.0)

            st.divider()
            st.subheader("3. Compound Library Screening")
            default_smiles = """DPCPX, CCCn1c(=O)c2[nH]c(C3CCCC3)nc2n(CCC)c1=O\nCaffeine, Cn1cnc2c1c(=O)n(C)c(=O)n2C"""
            smiles_text = st.text_area("Format: Name, SMILES", value=default_smiles, height=100)

            if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
                if not smiles_text.strip():
                    st.error("Please enter SMILES strings!")
                else:
                    library = {}
                    for line in smiles_text.strip().split('\n'):
                        if ',' in line:
                            name, smiles = line.split(',', 1)
                            library[name.strip()] = smiles.strip()
                            
                    with open(_TEMP_LIBRARY, "w") as f:
                        json.dump(library, f)

                    pdb_filename = st.session_state['active_pdb']
                    pdbqt_filename = os.path.join(RECEPTOR_DIR, f"{st.session_state['receptor_name']}.pdbqt")

                    with st.spinner(f"Assigning PROPKA charges (pH {ph_level}) to {pdb_filename}..."):
                        prep_cmd = ["python3", os.path.join(_HERE, "preptarget.py"), "-i", pdb_filename, "-o", pdbqt_filename, "-p", str(ph_level)]
                        prep_result = subprocess.run(prep_cmd, capture_output=True, text=True)

                        if prep_result.returncode != 0 or not os.path.exists(pdbqt_filename):
                            st.error("❌ Receptor preparation failed — the screen was not started. Check the PDB (chains / format) and that pdb2pqr + obabel are available in the env.")
                            if prep_result.stderr:
                                st.code(prep_result.stderr[-3000:])
                            st.stop()
                        fix_explosive_charges(pdbqt_filename)

                    st.divider()
                    st.subheader("🖥️ Live Execution Terminal")
                    st.caption("Monitoring real-time structural generation and docking physics...")
                    
                    terminal_expander = st.expander("View Console Output", expanded=True)
                    with terminal_expander:
                        # --- NEW: Fixed-height scrollable container ---
                        scroll_box = st.container(height=400)
                        with scroll_box:
                            terminal_placeholder = st.empty()

                    with st.spinner(f"Executing high-throughput screen for {len(library)} compounds..."):
                        dock_cmd = [
                            "python3", "-u", os.path.join(_HERE, "run_screen.py"),
                            "--receptor_pdb", pdb_filename,
                            "--receptor_pdbqt", pdbqt_filename,
                            "--library", _TEMP_LIBRARY,
                            "--x", str(center_x), "--y", str(center_y), "--z", str(center_z),
                            "--box_x", str(box_x), "--box_y", str(box_y), "--box_z", str(box_z),
                            "--seed", str(seed_val),
                            "--cpu", str(cpu_cores),
                            "--exhaustiveness", str(exhaustiveness),
                            "--n_poses", str(n_poses),
                            "--mode", dock_mode,
                            "--ph", str(ph_level)
                        ]
                        if screen_rescore:
                            dock_cmd += ["--rescore", "--rescore_jobs", str(cpu_cores)]

                        process = subprocess.Popen(
                            dock_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )

                        progress_bar = st.progress(0)
                        eta_placeholder = st.empty()
                        start_time = time.time()
                        
                        log_lines = []
                        for line in iter(process.stdout.readline, ''):
                            # Intercept the hidden progress tag
                            if "__PROGRESS__" in line:
                                parts = line.strip().split("__")
                                current = int(parts[2])
                                total = int(parts[3])
                                
                                # Update visual progress bar
                                progress_bar.progress(current / total)
                                
                                # Calculate moving ETA
                                elapsed = time.time() - start_time
                                avg_time_per = elapsed / current
                                remaining_seconds = avg_time_per * (total - current)
                                
                                eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(remaining_seconds))
                                eta_placeholder.info(f"⏳ **Screening Progress:** {current} / {total} compounds completed. **ETA:** {eta_str}")
                            else:
                                # Normal console streaming
                                log_lines.append(line)
                                # --- Keep 1,000 lines of history now that it's scrollable ---
                                display_text = "".join(log_lines[-1000:]) 
                                terminal_placeholder.code(display_text, language="bash")
                            
                        process.stdout.close()
                        return_code = process.wait()
                        
                        full_log_text = "".join(log_lines)
                        
                        if return_code != 0 or "ERROR:" in full_log_text:
                            st.error("❌ Critical Error detected in the pipeline! Check the terminal above.")
                        else:
                            st.success("✅ Batch Docking Complete! Master PyMOL Overlay generated. Go to the Analytics Tab to view results.")

# ==========================================
# TAB 2: ADVANCED ANALYTICS DASHBOARD
# ==========================================
with tab2:
    st.markdown("Upload multiple `_Screen_Results.csv` files to extract medicinal chemistry metrics, enforce pharmacophore rules, and compare target selectivity.")
    
    uploaded_csvs = st.file_uploader("Upload CSV Results", type=["csv"], accept_multiple_files=True)
    
    if uploaded_csvs:
        df_list = []
        for _f in uploaded_csvs:
            try:
                df_list.append(pd.read_csv(_f))
            except Exception as _e:
                st.error(f"❌ Could not read '{getattr(_f, 'name', _f)}': {_e}")
        if not df_list:
            st.stop()
        df = pd.concat(df_list, ignore_index=True)
        _required = ["Receptor", "Compound_Name", "SMILES", "Pose_Rank",
                     "Binding_Energy_kcal_mol", "Total_Contacts", "PLIP_Contacts_with_Distances"]
        _missing = [c for c in _required if c not in df.columns]
        if _missing:
            st.error("❌ Uploaded CSV(s) are missing required column(s): " + ", ".join(_missing)
                     + ". Upload a screening-results CSV produced by the pipeline.")
            st.stop()

        # Hydrated (AD4) and standard (Vina) energies are on different scales.
        # Warn if results from both engines are mixed in the same comparison.
        if 'Scoring_Function' in df.columns and df['Scoring_Function'].nunique() > 1:
            st.warning("⚠️ These results mix **Vina** and **AD4 (hydrated)** scoring functions. "
                       "Their binding energies are on different scales — binding-energy and "
                       "selectivity comparisons across the two engines are not valid. Filter to a "
                       "single scoring function before interpreting ΔG.")

        # --- LIGAND EFFICIENCY CALCULATION ---
        def get_heavy_atoms(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol.GetNumHeavyAtoms() if mol else np.nan
            except: return np.nan
            
        if 'Heavy_Atoms' not in df.columns:
            with st.spinner("Calculating Ligand Efficiency metrics via RDKit..."):
                df['Heavy_Atoms'] = df['SMILES'].apply(get_heavy_atoms)
                df['Ligand_Efficiency'] = abs(df['Binding_Energy_kcal_mol']) / df['Heavy_Atoms']
                
        # --- EXTRACT BASE COMPOUND NAME FOR CORRELATION ---
        df['Base_Compound'] = df['Compound_Name'].apply(base_compound)

        # --- OPTIONAL: MM/PBSA + MM/GBSA end-point rescoring panel ---
        # Fed by the companion {receptor}_MMPBSA.csv that `run_screen.py --rescore`
        # (or the Tab-1 rescoring checkbox) writes alongside the screen results.
        st.divider()
        with st.expander("🧬 MM/PBSA + MM/GBSA rescoring (upload companion `_MMPBSA.csv`)", expanded=False):
            st.caption("End-point free-energy re-ranking of the docked poses. The ligand is protonated to "
                       "the docking pH, so the electrostatics match the docked chemistry. **ΔG is on the "
                       "AmberTools implicit-solvent scale — not comparable to Vina's docking score.** Use it "
                       "to re-rank a shortlist, not to compare against ΔG_dock in absolute terms.")
            mm_csvs = st.file_uploader("Upload `{receptor}_MMPBSA.csv` file(s)", type=["csv"],
                                       accept_multiple_files=True, key="mmpbsa_companion_upl")
            if mm_csvs:
                mm_list = []
                for _mf in mm_csvs:
                    try:
                        mm_list.append(pd.read_csv(_mf))
                    except Exception as _e:
                        st.error(f"❌ Could not read '{getattr(_mf, 'name', _mf)}': {_e}")
                if mm_list:
                    mmdf = pd.concat(mm_list, ignore_index=True)
                    if "Compound_Name" not in mmdf.columns or "MMPBSA_dG" not in mmdf.columns:
                        st.error("❌ Not an MM/PBSA companion CSV (needs `Compound_Name` and `MMPBSA_dG`). "
                                 "This is the `{receptor}_MMPBSA.csv` written by --rescore, not the Screen_Results CSV.")
                    else:
                        for _c in ["MMPBSA_dG", "MMGBSA_dG"]:
                            if _c in mmdf.columns:
                                mmdf[_c] = pd.to_numeric(mmdf[_c], errors="coerce")
                        # Best (most negative) ΔG per compound across its stereo/tautomer variations.
                        agg = {"MMPBSA_dG": "min"}
                        if "MMGBSA_dG" in mmdf.columns:
                            agg["MMGBSA_dG"] = "min"
                        if "Ligand_Net_Charge" in mmdf.columns:
                            agg["Ligand_Net_Charge"] = "first"
                        best = (mmdf.groupby("Compound_Name").agg(agg).reset_index()
                                    .sort_values("MMPBSA_dG"))
                        best.insert(0, "MM/PBSA Rank", range(1, len(best) + 1))
                        scored = int(mmdf["MMPBSA_dG"].notna().any() and best["MMPBSA_dG"].notna().sum())
                        st.markdown(f"**End-point free-energy ranking** — {scored} compound(s) scored, "
                                    "lowest ΔG_MM/PBSA first:")
                        st.dataframe(best, hide_index=True, use_container_width=True)
                        _plot = best.dropna(subset=["MMPBSA_dG"]).head(20)
                        if not _plot.empty:
                            fig_mm = px.bar(_plot, x="Compound_Name", y="MMPBSA_dG",
                                            title="Top compounds by MM/PBSA ΔG (more negative = tighter)",
                                            labels={"MMPBSA_dG": "ΔG_MM/PBSA (kcal/mol)",
                                                    "Compound_Name": "Compound"})
                            fig_mm.update_layout(xaxis={"categoryorder": "total ascending", "tickangle": 45})
                            st.plotly_chart(fig_mm, use_container_width=True)
                        # cross-check: how does MM/PBSA rank vs the docking score?
                        try:
                            dock_best = (df.sort_values("Binding_Energy_kcal_mol")
                                           .groupby("Base_Compound")["Binding_Energy_kcal_mol"].first())
                            merged = best.set_index("Compound_Name").join(dock_best, how="inner")
                            merged = merged.dropna(subset=["MMPBSA_dG", "Binding_Energy_kcal_mol"])
                            if len(merged) >= 3:
                                rho, _p = spearmanr(merged["Binding_Energy_kcal_mol"], merged["MMPBSA_dG"])
                                st.caption(f"Spearman ρ(ΔG_dock, ΔG_MM/PBSA) = **{rho:.2f}** over "
                                           f"{len(merged)} compounds — how much the physics re-ranks the "
                                           "docking order (ρ≈1 means it agrees; lower means real re-ranking).")
                        except Exception:
                            pass
                        st.download_button("📥 Download MM/PBSA ranking (CSV)",
                                           data=best.to_csv(index=False).encode("utf-8"),
                                           file_name="mmpbsa_ranking.csv", mime="text/csv",
                                           key="dl_mmpbsa_rank")

        st.divider()
        st.subheader("⚙️ Global Analysis & Pharmacophore Constraints")
        
        col_mode, col_filter = st.columns([1, 1])
        with col_mode:
            analysis_mode = st.radio(
                "Select the depth of your analysis:",
                ["Top Pose Only (Rank 1) - Best Thermodynamic Fit", 
                 "Pose Stability (Consensus) - Evaluate all generated poses"]
            )
        
        # --- FEATURE 2: STRICT PHARMACOPHORE FILTER ---
        with col_filter:
            st.markdown("**Strict Pharmacophore Constraint:**")
            st.caption("Filter out poses that do not interact with a specific anchor residue (e.g., ASN254).")
            
            # Extract all unique interacting residues from the dataset
            all_residues = set()
            for contacts in df['PLIP_Contacts_with_Distances'].dropna():
                for contact in str(contacts).split(" | "):
                    match = PLIP_RESIDUE_RE.search(contact)
                    if match: all_residues.add(match.group(1))
                    
            required_res = st.multiselect("Require interaction with:", sorted(list(all_residues)))
            
        # Apply the Pharmacophore Filter
        if required_res:
            def contains_required_res(contact_string):
                if pd.isna(contact_string): return False
                return all(req in contact_string for req in required_res)
            
            df_filtered = df[df['PLIP_Contacts_with_Distances'].apply(contains_required_res)].copy()
            if df_filtered.empty:
                st.error("❌ No poses in the dataset form interactions with the selected residues.")
                st.stop()
            else:
                st.success(f"✅ Filtered to {len(df_filtered)} poses containing the required pharmacophore.")
        else:
            df_filtered = df.copy()

        # --- FEATURE: interaction-anchored vs raw-score hit ranking ---
        st.divider()
        st.subheader("🎯 Hit ranking — raw score vs interaction-anchored")
        st.caption("Pick anchor residue(s) in the Pharmacophore Constraint above, then compare the "
                   "raw-ΔG hit list with the interaction-anchored one: each compound's pose is "
                   "re-selected by how many of your anchor contacts it makes (interaction-based pose "
                   "selection — guards against the raw rank-1 pose being mis-ranked).")
        if not required_res:
            st.info("☝️ Select one or more anchor residues above to enable the interaction-anchored ranking.")
        else:
            _anchor = set(required_res)
            def _resset(s):
                return set(PLIP_RESIDUE_RE.findall(str(s))) if pd.notna(s) else set()
            _w = df.copy()
            _w["_res"] = _w["PLIP_Contacts_with_Distances"].apply(_resset)
            _w["_hit"] = _w["_res"].apply(lambda r: len(r & _anchor))
            _w["_cov"] = _w["_hit"] / max(len(_anchor), 1)
            _raw, _int = [], []
            for _b, _g in _w.groupby("Base_Compound"):
                _rt = _g.sort_values("Binding_Energy_kcal_mol").iloc[0]
                _sl = _g.sort_values(["_cov", "Binding_Energy_kcal_mol"], ascending=[False, True]).iloc[0]
                _diff = bool((_rt["Compound_Name"], _rt["Pose_Rank"]) != (_sl["Compound_Name"], _sl["Pose_Rank"]))
                _raw.append({"Compound": _b, "Variant": _rt["Compound_Name"], "Pose": int(_rt["Pose_Rank"]),
                             "Binding_Energy_kcal_mol": _rt["Binding_Energy_kcal_mol"],
                             "Anchors_hit": f"{int(_rt['_hit'])}/{len(_anchor)}"})
                _int.append({"Compound": _b, "Selected_Variant": _sl["Compound_Name"], "Pose": int(_sl["Pose_Rank"]),
                             "Binding_Energy_kcal_mol": _sl["Binding_Energy_kcal_mol"],
                             "Anchors_hit": f"{int(_sl['_hit'])}/{len(_anchor)}",
                             "Coverage": round(float(_sl["_cov"]), 2), "Differs_from_raw_#1": _diff})
            _rawdf = pd.DataFrame(_raw).sort_values("Binding_Energy_kcal_mol").reset_index(drop=True)
            _rawdf.insert(0, "Rank", _rawdf.index + 1)
            _intdf = pd.DataFrame(_int).sort_values(["Coverage", "Binding_Energy_kcal_mol"],
                                                    ascending=[False, True]).reset_index(drop=True)
            _intdf.insert(0, "Rank", _intdf.index + 1)
            _c1, _c2 = st.columns(2)
            with _c1:
                st.markdown("**Non-interaction-anchored** — raw Vina ΔG")
                st.dataframe(_rawdf, hide_index=True, use_container_width=True)
                st.download_button("📥 raw_ranked.csv", _rawdf.to_csv(index=False).encode(),
                                   "raw_ranked.csv", "text/csv", key="dl_raw_rank")
            with _c2:
                st.markdown(f"**Interaction-anchored** — best recovery of {', '.join(sorted(_anchor))}")
                st.dataframe(_intdf, hide_index=True, use_container_width=True)
                st.download_button("📥 interaction_ranked.csv", _intdf.to_csv(index=False).encode(),
                                   "interaction_ranked.csv", "text/csv", key="dl_int_rank")
            _ng = sum(r["Differs_from_raw_#1"] for r in _int)
            if _ng:
                st.warning(f"⚠ {_ng} compound(s): the interaction-selected pose differs from the raw "
                           "#1 pose (the scoring gap) — trust the interaction-selected pose.")

        # Apply the Depth Mode
        if "Rank 1" in analysis_mode:
            idx = df_filtered.groupby(['Receptor', 'Compound_Name'])['Pose_Rank'].idxmin()
            df_active = df_filtered.loc[idx].copy()
            agg_function = 'size' 
            heatmap_label = "Contact Count"
        else:
            df_active = df_filtered.copy()
            agg_function = 'nunique' 
            heatmap_label = f"Poses Containing Interaction"

        st.divider()
        colA, colB = st.columns(2)
        
        with colA:
            # --- UPGRADED CHART: LIGAND EFFICIENCY LANDSCAPE ---
            st.subheader("1. Ligand Efficiency Landscape")
            st.caption("Identifies highly efficient binders. Top-right quadrant (>0.3 LE, most negative ΔG) is optimal.")
            
            fig_scatter = px.scatter(
                df_active, x="Ligand_Efficiency", y="Binding_Energy_kcal_mol", 
                color="Receptor", symbol="Receptor",
                hover_name="Compound_Name", 
                hover_data=["Pose_Rank", "Total_Contacts", "Heavy_Atoms"],
                size="Total_Contacts", size_max=15,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_scatter.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Ideal LE (>0.3)")
            fig_scatter.update_yaxes(autorange="reversed")
            
            # Give the markers a dark border so they pop against each other
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with colB:
            # --- NEW CHART: INTERACTION MECHANISM PROFILING ---
            st.subheader("2. Interaction Mechanism Profile")
            st.caption("Composition of binding forces driving the affinity.")
            
            mech_data = []
            for _, row in df_active.iterrows():
                if pd.notna(row['PLIP_Contacts_with_Distances']):
                    contacts = str(row['PLIP_Contacts_with_Distances']).split(" | ")
                    for contact in contacts:
                        if ":" in contact:
                            mech_data.append({
                                "Compound": f"{row['Compound_Name']} (Pose {row['Pose_Rank']})",
                                "Interaction_Type": contact.split(":")[0],
                                "Receptor": row['Receptor']
                            })
                            
            if mech_data:
                df_mech = pd.DataFrame(mech_data)
                fig_bar = px.histogram(df_mech, x="Compound", color="Interaction_Type", 
                                       barmode="stack", facet_col="Receptor")
                fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Not enough PLIP data to generate mechanism profile.")

        # --- THE NEW SELECTIVITY FEATURE ---
        st.divider()
        st.subheader("3. Theoretical Selectivity Profiling (ΔE)")
        st.markdown("Evaluate compound selectivity by calculating the difference in binding energy between two targets. **A difference > 1.0 kcal/mol indicates strong theoretical selectivity.**")

        receptors = df_active['Receptor'].unique()

        if len(receptors) >= 2:
            col_sel_1, col_sel_2 = st.columns(2)
            with col_sel_1:
                rec_A = st.selectbox("Target Receptor (A)", receptors, index=0, key="sel_A")
            with col_sel_2:
                rec_B = st.selectbox("Off-Target / Comparator (B)", receptors, index=1, key="sel_B")

            if rec_A != rec_B:
                # 1. Pivot the data to align the same compound across both receptors
                df_pivot = df_active.pivot_table(
                    index='Base_Compound',
                    columns='Receptor',
                    values='Binding_Energy_kcal_mol',
                    aggfunc='min' # Grabs the best energy pose
                ).dropna(subset=[rec_A, rec_B]).reset_index()

                # 2. Calculate the Selectivity Delta
                df_pivot['Delta_E'] = df_pivot[rec_A] - df_pivot[rec_B]

                # 3. Assign categories based on a 0.5 kcal/mol noise threshold
                def determine_preference(val):
                    if val < -0.5: return f"Favors {rec_A}"
                    elif val > 0.5: return f"Favors {rec_B}"
                    else: return "Non-Selective (Equipotent)"

                df_pivot['Preference'] = df_pivot['Delta_E'].apply(determine_preference)

                # 4. Sort the dataframe so the bar chart flows smoothly from Left to Right
                df_pivot = df_pivot.sort_values('Delta_E')

                # 5. Generate the Plotly Bar Chart
                fig_sel = px.bar(
                    df_pivot,
                    x='Base_Compound',
                    y='Delta_E',
                    color='Preference',
                    color_discrete_map={
                        f"Favors {rec_A}": "#EF553B", # Matches Plotly's default Red
                        f"Favors {rec_B}": "#636EFA", # Matches Plotly's default Blue
                        "Non-Selective (Equipotent)": "#7F7F7F" # Gray for noise
                    },
                    labels={
                        'Delta_E': f'Δ Binding Energy (kcal/mol)<br>Negative = {rec_A} | Positive = {rec_B}',
                        'Base_Compound': 'Compound'
                    },
                    hover_data={rec_A: True, rec_B: True} # Shows the raw scores on hover
                )
                
                # Add a hard zero line for visual grounding
                fig_sel.add_hline(y=0, line_width=2, line_color="black")
                
                # Clean up the layout
                fig_sel.update_layout(
                    xaxis={'categoryorder':'total ascending', 'tickangle': 45},
                    yaxis_title="Selectivity (Δ kcal/mol)",
                    legend_title="Receptor Preference"
                )
                
                st.plotly_chart(fig_sel, use_container_width=True)
            else:
                st.warning("Please select two different receptors to compare.")
        else:
            st.info("Upload data containing at least two distinct receptors to activate selectivity profiling.")


        st.divider()
        st.subheader(f"4. Residue Fingerprint Drill-Down: {heatmap_label}")
        
        if len(receptors) > 0:
            target_fingerprint = st.selectbox("Select Receptor for Fingerprint", receptors, key="fp_target")
            df_fp = df_active[df_active['Receptor'] == target_fingerprint].copy()
            
            parsed_data = []
            for _, row in df_fp.iterrows():
                contacts = str(row['PLIP_Contacts_with_Distances']).split(" | ")
                for contact in contacts:
                    if ":" in contact and "(" in contact:
                        interaction_type = contact.split(":")[0]
                        residue = PLIP_RESIDUE_RE.search(contact)
                        if residue:
                            parsed_data.append({
                                "Compound": row['Compound_Name'],
                                "Pose": row['Pose_Rank'],
                                "Residue": residue.group(1),
                                "Type": interaction_type
                            })
                            
            if parsed_data:
                fp_df = pd.DataFrame(parsed_data)
                if "Rank 1" in analysis_mode:
                    fp_matrix = pd.crosstab(fp_df['Residue'], fp_df['Compound']).fillna(0).astype(int)
                else:
                    fp_matrix = pd.crosstab(fp_df['Residue'], fp_df['Compound'], values=fp_df['Pose'], aggfunc=agg_function).fillna(0).astype(int)
                
                formatted_matrix = fp_matrix.replace(0, '-')
                st.dataframe(formatted_matrix.style.background_gradient(cmap='Blues', axis=None), use_container_width=True, height=400)
                
                st.divider()
                st.subheader("5. Unified Interaction Network (Consensus Pharmacophore)")
                
                df_network = df_active[df_active['Receptor'] == target_fingerprint]
                
                # --- NEW: Node Frequency Filter ---
                max_compounds = len(df_network['Compound_Name'].unique())
                if max_compounds > 1:
                    min_freq = st.slider(
                        "Consensus Filter (Minimum Compounds):", 
                        min_value=1, max_value=max_compounds, value=max(1, int(max_compounds * 0.1)),
                        help="Filters out rare interactions to highlight the core pharmacophore and prevent the graph's physics engine from overloading."
                    )
                else:
                    min_freq = 1

                st.markdown(f"Pharmacophore mapping for **{target_fingerprint}**. Displaying residues interacting with at least **{min_freq}** compounds. **Click and drag the nodes** to explore.")

                edge_dict = {} 
                
                for _, row in df_network.iterrows():
                    ligand = row['Compound_Name']
                    contacts = str(row['PLIP_Contacts_with_Distances']).split(" | ")
                    for contact in contacts:
                        if ":" in contact and "(" in contact:
                            interaction_type = contact.split(":")[0]
                            residue = PLIP_RESIDUE_RE.search(contact)
                            if residue:
                                res_name = residue.group(1)
                                edge_key = (ligand, res_name)
                                if edge_key not in edge_dict:
                                    edge_dict[edge_key] = set()
                                edge_dict[edge_key].add(interaction_type)
                
                res_counts = {}
                for (lig, res) in edge_dict.keys():
                    res_counts[res] = res_counts.get(res, 0) + 1
                    
                # Ligand–residue edges surviving the consensus filter
                kept = [(lig, res, bt) for (lig, res), bt in edge_dict.items()
                        if res_counts[res] >= min_freq]
                kept_ligands = sorted({lig for lig, _r, _b in kept})
                kept_residues = sorted({res for _l, res, _b in kept})

                if not kept:
                    st.info("No interactions meet the current consensus threshold. Lower the slider to view rarer contacts.")
                else:
                    layout_choice = st.selectbox(
                        "Network layout",
                        ["Force-directed (stable)", "Concentric (residues inner)",
                         "Bipartite columns", "Live physics (legacy)"],
                        index=0,
                        help="Stable layouts are computed once with networkx and frozen, so the graph "
                             "no longer jitters as you add compounds — you can still drag nodes and "
                             "zoom. 'Live physics' is the old vis.js simulation.",
                    )
                    live_physics = layout_choice.startswith("Live physics")

                    # Precompute a deterministic layout instead of running a live physics sim
                    pos = {}
                    if not live_physics:
                        G = nx.Graph()
                        G.add_nodes_from(kept_ligands)
                        G.add_nodes_from(kept_residues)
                        G.add_edges_from((lig, res) for lig, res, _b in kept)
                        try:
                            if layout_choice.startswith("Concentric"):
                                pos = nx.shell_layout(G, nlist=[kept_residues, kept_ligands])
                            elif layout_choice.startswith("Bipartite"):
                                pos = nx.bipartite_layout(G, kept_ligands)
                            else:  # Force-directed (stable): solved to convergence, then frozen
                                try:
                                    pos = nx.kamada_kawai_layout(G)
                                except Exception:
                                    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=300)
                        except Exception:
                            pos = nx.spring_layout(G, seed=42, k=1.2, iterations=300)
                        # Normalised coords -> canvas units that grow with the graph
                        scale = 120.0 * (len(G) ** 0.5)
                        pos = {n: (float(x) * scale, float(y) * scale) for n, (x, y) in pos.items()}

                    def _xy(node_id):
                        # physics:False keeps a placed node from being re-simulated (no jitter),
                        # but it stays draggable because it isn't pinned with fixed:True.
                        return ({"x": pos[node_id][0], "y": pos[node_id][1], "physics": False}
                                if node_id in pos else {})

                    nodes = []
                    edges = []
                    for ligand in kept_ligands:
                        nodes.append(Node(id=ligand, label=ligand, size=35, color="#1f77b4",
                                          shape="diamond", **_xy(ligand)))
                    for residue in kept_residues:
                        color = "#2ca02c" if res_counts[residue] > min_freq else "#d62728"
                        nodes.append(Node(id=residue, label=residue, size=20, color=color,
                                          shape="dot", **_xy(residue)))
                    for lig, res, bond_types in kept:
                        edges.append(Edge(source=lig, target=res, color="#A0A0A0",
                                          title=f"Interactions: {', '.join(sorted(bond_types))}"))

                    # More nodes -> taller canvas so a frozen layout stays legible
                    height = int(min(1000, max(500, 22 * len(nodes))))
                    config = Config(width="100%", height=height, directed=False,
                                    physics=live_physics, hierarchical=False)
                    agraph(nodes=nodes, edges=edges, config=config)

        st.divider()
        st.subheader("6. Experimental Correlation (Model Validation)")
        st.markdown(
            "Upload per-compound assay affinities and the dashboard will line the docking order up "
            "against the measured order — by chemical **series** and by **which metric tracks the assay**. "
            "This is where a series can rank well computationally yet fail in the wet lab."
        )
        st.caption(
            "CSV needs **`Compound_Name`** (matching the base names, e.g. `CPD1`) and **`pKi`** "
            "(higher = stronger binder). Optional **`SEM`** draws assay error bars. Other affinity "
            "units: convert to pKi first (e.g. `pKi = 9 − log10(IC50_nM)`)."
        )
        exp_file = st.file_uploader("Upload Lab Data (CSV with 'Compound_Name', 'pKi', optional 'SEM')", type=["csv"])

        if exp_file:
            try:
                df_exp = pd.read_csv(exp_file)
            except Exception as _e:
                st.error(f"Could not read the assay CSV: {_e}")
                df_exp = None

            if df_exp is not None and {'Compound_Name', 'pKi'}.issubset(df_exp.columns):
                df_exp = df_exp.rename(columns={'Compound_Name': 'Base_Compound'}).copy()
                if 'SEM' not in df_exp.columns:
                    df_exp['SEM'] = 0.0
                df_exp = df_exp[['Base_Compound', 'pKi', 'SEM']].copy()
                df_exp['Base_Compound'] = df_exp['Base_Compound'].astype(str)
                df_exp['pKi'] = pd.to_numeric(df_exp['pKi'], errors='coerce')
                df_exp['SEM'] = pd.to_numeric(df_exp['SEM'], errors='coerce').fillna(0.0)
                df_exp = df_exp.dropna(subset=['pKi'])

                # --- per-compound docking metrics across ALL poses (raw score is filter-independent) ---
                comp = (df.groupby('Base_Compound')['Binding_Energy_kcal_mol']
                          .min().reset_index(name='Best_Energy'))
                have_cov = bool(required_res)
                if have_cov:
                    _anc = set(required_res)
                    _cov = df['PLIP_Contacts_with_Distances'].apply(
                        lambda s: (len(set(PLIP_RESIDUE_RE.findall(str(s))) & _anc) / len(_anc))
                        if (pd.notna(s) and _anc) else 0.0)
                    comp = comp.merge(
                        df.assign(_cov=_cov).groupby('Base_Compound')['_cov']
                          .max().reset_index(name='Best_AnchorCoverage'),
                        on='Base_Compound', how='left')

                comp['Series'] = comp['Base_Compound'].apply(series_of)
                merged = comp.merge(df_exp, on='Base_Compound', how='inner')

                if merged.empty:
                    st.warning(
                        "No compound names matched between the screen and the assay file. "
                        "The assay `Compound_Name` must equal the base compound name (e.g. `CPD1`, not `CPD1_var1`)."
                    )
                else:
                    n_series = merged['Series'].nunique()
                    st.caption(
                        f"Matched **{len(merged)}** compounds across **{n_series}** series "
                        f"({', '.join(sorted(merged['Series'].unique()))}). "
                        "Reminder: more **negative ΔG** = better-predicted binder; higher **pKi** = stronger measured binder."
                    )

                    # ================= Per-series rollup — the headline =================
                    st.markdown("##### Per-series rollup — does the docking order match the assay order?")
                    agg = {'n': ('pKi', 'size'), 'Mean_pKi': ('pKi', 'mean'),
                           'Mean_Best_dG': ('Best_Energy', 'mean')}
                    roll = merged.groupby('Series').agg(**agg).reset_index()
                    if have_cov:
                        roll = roll.merge(
                            merged.groupby('Series')['Best_AnchorCoverage'].mean()
                                  .reset_index().rename(columns={'Best_AnchorCoverage': 'Mean_Coverage'}),
                            on='Series', how='left')
                    # ranks (1 = best): assay by high pKi, score by low ΔG, coverage by high coverage
                    roll['Assay_rank'] = roll['Mean_pKi'].rank(ascending=False, method='min').astype(int)
                    roll['Score_rank'] = roll['Mean_Best_dG'].rank(ascending=True, method='min').astype(int)
                    if have_cov:
                        roll['Coverage_rank'] = roll['Mean_Coverage'].rank(ascending=False, method='min').astype(int)
                    roll = roll.sort_values('Assay_rank')
                    _round = {'Mean_pKi': 2, 'Mean_Best_dG': 2}
                    if have_cov:
                        _round['Mean_Coverage'] = 2
                    st.dataframe(roll.round(_round), hide_index=True, use_container_width=True)

                    # flag any series the docking over-rewards relative to the assay
                    worst_assay = roll.sort_values('Mean_pKi').iloc[0]
                    if (worst_assay['Score_rank'] <= max(1, n_series // 2)) and n_series >= 2:
                        cov_txt = (f" and coverage rank #{int(worst_assay['Coverage_rank'])}"
                                   if have_cov else "")
                        st.error(
                            f"⚠️ **Series `{worst_assay['Series']}` is the weakest in the assay "
                            f"(mean pKi {worst_assay['Mean_pKi']:.2f}) yet ranks #{int(worst_assay['Score_rank'])} "
                            f"by docking score{cov_txt}** — a clear false-positive series. "
                            "The docking score may be over-rewarding this series."
                        )
                    st.download_button("⬇️ Download per-series rollup (CSV)",
                                       roll.to_csv(index=False).encode(),
                                       file_name="series_rollup_vs_assay.csv", mime="text/csv")

                    # ================= Which metric predicts the assay? =================
                    st.markdown("##### Which metric predicts the assay?")

                    def _corr_line(name, x, y, want):
                        x = pd.to_numeric(x, errors='coerce'); y = pd.to_numeric(y, errors='coerce')
                        ok = x.notna() & y.notna()
                        x, y = x[ok], y[ok]
                        if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
                            return f"- **{name}:** not enough spread to correlate (need ≥3 varied points)."
                        pr = pearsonr(x, y); sr = spearmanr(x, y)
                        r, rp = float(pr[0]), float(pr[1])
                        rho, sp = float(sr[0]), float(sr[1])
                        good = (r < 0 if want == 'neg' else r > 0) and abs(r) >= 0.3
                        mark = "✅ tracks the assay" if good else "❌ does **not** track the assay"
                        return (f"- **{name}:** Pearson r = {r:+.2f} (p={rp:.2f}), "
                                f"Spearman ρ = {rho:+.2f} (p={sp:.2f}) — {mark} "
                                f"(predictive if strongly {'negative' if want == 'neg' else 'positive'}).")

                    lines = [_corr_line("Raw Vina ΔG vs pKi", merged['pKi'], merged['Best_Energy'], 'neg')]
                    if have_cov:
                        lines.append(_corr_line("Interaction-anchor coverage vs pKi",
                                                merged['pKi'], merged['Best_AnchorCoverage'], 'pos'))
                    else:
                        lines.append("- *Interaction-anchor coverage:* select anchor residues in the "
                                     "**Pharmacophore Constraint** above to add the second metric.")
                    st.markdown("\n".join(lines))

                    # ================= Scatters, coloured by series =================
                    _palette = px.colors.qualitative.Set1
                    cols = st.columns(2) if have_cov else [st.container()]
                    with cols[0]:
                        f1 = px.scatter(
                            merged, x='pKi', y='Best_Energy', color='Series',
                            error_x='SEM', hover_name='Base_Compound',
                            labels={'pKi': 'Experimental affinity (pKi)',
                                    'Best_Energy': 'Best ΔG (kcal/mol)'},
                            title='Raw score vs assay', color_discrete_sequence=_palette)
                        f1.update_traces(marker=dict(size=11, line=dict(width=1, color='DarkSlateGrey')))
                        f1.update_yaxes(autorange='reversed')  # better binder (more −ve) toward top
                        st.plotly_chart(f1, use_container_width=True)
                    if have_cov:
                        with cols[1]:
                            f2 = px.scatter(
                                merged, x='pKi', y='Best_AnchorCoverage', color='Series',
                                error_x='SEM', hover_name='Base_Compound',
                                labels={'pKi': 'Experimental affinity (pKi)',
                                        'Best_AnchorCoverage': 'Best anchor coverage (fraction)'},
                                title='Interaction fingerprint vs assay', color_discrete_sequence=_palette)
                            f2.update_traces(marker=dict(size=11, line=dict(width=1, color='DarkSlateGrey')))
                            st.plotly_chart(f2, use_container_width=True)
                    st.info(
                        "💡 In a predictive model the points trend from bottom-left to top-right "
                        "(stronger assay binders also score/fingerprint better). A series sitting "
                        "high on the docking axis but low on pKi is a computational false positive."
                    )
            elif df_exp is not None:
                st.error("Your assay CSV must contain at least `Compound_Name` and `pKi` columns (optional `SEM`).")

        st.divider()
        st.subheader("7. Interactive 3D Viewer & PyMOL Export")
        st.markdown("Inspect the 3D complex and select specific poses.")

        if len(receptors) > 0:
            col_select1, col_select2, col_select3 = st.columns(3)
            with col_select1:
                view_receptor = st.selectbox("Select Receptor", df['Receptor'].unique(), key="3d_rec")
            with col_select2:
                compounds_for_rec = df[df['Receptor'] == view_receptor]['Compound_Name'].unique()
                view_compound = st.selectbox("Select Compound", compounds_for_rec, key="3d_comp")
            with col_select3:
                poses = df[(df['Receptor'] == view_receptor) & (df['Compound_Name'] == view_compound)]['Pose_Rank'].unique()
                view_pose = st.selectbox("Select Pose Rank", sorted(poses), key="3d_pose")

            visuals_dir = f"{view_receptor}_docking_results"
            pdb_file = f"{visuals_dir}/{view_compound}_Ensemble.pdb"
            pml_file = f"{visuals_dir}/{view_compound}_view.pml"

            col_viewer, col_download, col_2d = st.columns([5, 2, 5])

            with col_2d:
                st.markdown("#### 2D Interaction Map")
                st.caption(
                    "Ligand-interaction diagram from the PLIP contacts — residues coloured by "
                    "type, contacts by interaction. Hover a residue for distances; use the camera "
                    "icon to save a PNG."
                )
                _row = df[(df['Receptor'] == view_receptor) &
                          (df['Compound_Name'] == view_compound) &
                          (df['Pose_Rank'] == view_pose)]['PLIP_Contacts_with_Distances']
                _cstr = str(_row.values[0]) if len(_row) else "nan"
                _res_int = {}
                if _cstr and _cstr != "nan":
                    for _c in _cstr.split(" | "):
                        _m = re.match(r'\s*([A-Za-z\-]+):([A-Z]+)(\d+)\(([\d.]+)', _c)
                        if _m:
                            _rt, _rn, _num, _d = _m.group(1), _m.group(2), _m.group(3), _m.group(4)
                            _res_int.setdefault(f"{_rn}{_num}",
                                                {"name": _rn, "num": int(_num), "ints": []})["ints"].append((_rt, _d))

                if not _res_int:
                    st.info("No PLIP interactions were recorded for this pose.")
                else:
                    # amino-acid physicochemical classes -> Schrödinger-style bubble colours
                    _POS = {"ARG", "LYS", "HIS"}
                    _NEG = {"ASP", "GLU"}
                    _POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
                    _HYDRO = {"ALA", "VAL", "LEU", "ILE", "PHE", "MET", "PRO", "TRP", "GLY"}
                    _WATER = {"HOH", "WAT", "H2O"}

                    def _res_class(nm):
                        nm = nm.upper()
                        if nm in _WATER: return ("Water", "#EC7063")
                        if nm in _POS:   return ("Positive (+)", "#AEB6E8")
                        if nm in _NEG:   return ("Negative (−)", "#F1948A")
                        if nm in _POLAR: return ("Polar", "#A3E4D7")
                        if nm in _HYDRO: return ("Hydrophobic", "#ABEBC6")
                        return ("Other", "#E5E7E9")

                    _ISTYLE = {
                        "H-Bond":     ("#8E44AD", "dash",   "H-bond"),
                        "SaltBridge": ("#2E86C1", "dash",   "Salt bridge"),
                        "Pi-Stack":   ("#1E8449", "solid",  "π–π stacking"),
                        "Pi-Cat":     ("#CA6F1E", "solid",  "π–cation"),
                        "Hydro":      ("#B3B6B7", "dot",    "Hydrophobic"),
                        "Halogen":    ("#17A589", "dash",   "Halogen bond"),
                        "Water":      ("#5DADE2", "dot",    "Water bridge"),
                        "Metal":      ("#7D3C98", "solid",  "Metal"),
                        "Contact":    ("#D5DBDB", "dot",    "Contact"),
                    }

                    _residues = sorted(_res_int.values(), key=lambda r: r["num"])
                    _n = len(_residues)
                    _fig2d = go.Figure()
                    _fig2d.add_shape(type="circle", x0=-1.28, y0=-1.28, x1=1.28, y1=1.28,
                                     fillcolor="#F7F9FB", line=dict(color="#EAEDED", width=1), layer="below")

                    _nx, _ny, _nfill, _nlabel, _nhover = [], [], [], [], []
                    for _i, _r in enumerate(_residues):
                        _a = np.pi / 2 - 2 * np.pi * _i / _n  # start at top, clockwise
                        _cls, _fill = _res_class(_r["name"])
                        _nx.append(float(np.cos(_a))); _ny.append(float(np.sin(_a)))
                        _nfill.append(_fill)
                        _nlabel.append(f"{_r['name']}<br>{_r['num']}")
                        _htxt = "; ".join(f"{_ISTYLE.get(_bt, ('', '', _bt))[2]} {_dd}Å" for _bt, _dd in _r["ints"])
                        _nhover.append(f"{_r['name']}{_r['num']} — {_htxt}")
                        _k = len(_r["ints"])
                        for _j, (_bt, _dd) in enumerate(_r["ints"]):  # fan multiple contacts to one residue
                            _col, _dash, _lab = _ISTYLE.get(_bt, ("#CB4335", "solid", _bt))
                            _off = (_j - (_k - 1) / 2) * 0.07
                            _ex, _ey = float(np.cos(_a + _off)), float(np.sin(_a + _off))
                            _fig2d.add_trace(go.Scatter(
                                x=[0, _ex * 0.84], y=[0, _ey * 0.84], mode="lines",
                                line=dict(color=_col, width=3, dash=_dash),
                                showlegend=False, hoverinfo="skip"))

                    _fig2d.add_trace(go.Scatter(
                        x=_nx, y=_ny, mode="markers+text",
                        marker=dict(size=42, color=_nfill, line=dict(color="#5D6D7E", width=1.5)),
                        text=_nlabel, textposition="middle center",
                        textfont=dict(size=10, color="#1B2631"),
                        hovertext=_nhover, hoverinfo="text", showlegend=False))

                    _fig2d.add_trace(go.Scatter(
                        x=[0], y=[0], mode="markers+text",
                        marker=dict(size=66, color="#EAF2F8", symbol="circle",
                                    line=dict(color="#2874A6", width=2)),
                        text=[view_compound], textposition="middle center",
                        textfont=dict(size=10, color="#1B4F72"),
                        showlegend=False, hoverinfo="skip"))

                    # grouped legend: interaction types, then residue types
                    _int_present = []
                    for _r in _residues:
                        for _bt, _dd in _r["ints"]:
                            if _bt not in _int_present:
                                _int_present.append(_bt)
                    for _idx, _bt in enumerate(_int_present):
                        _col, _dash, _lab = _ISTYLE.get(_bt, ("#CB4335", "solid", _bt))
                        _fig2d.add_trace(go.Scatter(
                            x=[None], y=[None], mode="lines", name=_lab,
                            line=dict(color=_col, width=3, dash=_dash), legendgroup="int",
                            legendgrouptitle_text="Interactions" if _idx == 0 else None))
                    _cls_present = []
                    for _r in _residues:
                        _c = _res_class(_r["name"])
                        if _c[0] not in [_x[0] for _x in _cls_present]:
                            _cls_present.append(_c)
                    for _idx, (_cls, _fill) in enumerate(_cls_present):
                        _fig2d.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers", name=_cls,
                            marker=dict(size=13, color=_fill, line=dict(color="#7f8c8d", width=1)),
                            legendgroup="res",
                            legendgrouptitle_text="Residue type" if _idx == 0 else None))

                    _fig2d.update_layout(
                        height=560, plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(yanchor="middle", y=0.5, x=1.02, font=dict(size=11),
                                    bordercolor="#D5DBDB", borderwidth=1),
                        xaxis=dict(visible=False, range=[-1.5, 1.5]),
                        yaxis=dict(visible=False, range=[-1.42, 1.42], scaleanchor="x", scaleratio=1))
                    st.plotly_chart(_fig2d, use_container_width=True)

            if os.path.exists(pdb_file) and os.path.exists(pml_file):
                
                with col_viewer:
                    with open(pdb_file, 'r') as f:
                        pdb_data = f.read()

                    lines = pdb_data.split('\n')
                    single_pose_lines = []
                    in_model = False
                    for line in lines:
                        if line.startswith(f"MODEL {view_pose}"):
                            in_model = True
                        if in_model:
                            single_pose_lines.append(line)
                        if line.startswith("ENDMDL") and in_model:
                            break
                    
                    single_pose_pdb = '\n'.join(single_pose_lines)

                    view = py3Dmol.view(width=520, height=500)
                    view.addModel(single_pose_pdb, 'pdb') 
                    view.setStyle({'model': -1}, {'cartoon': {'color': '#e2e2e2'}}) 
                    view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15}})
                    
                    plip_data = df[(df['Compound_Name'] == view_compound) & 
                                   (df['Receptor'] == view_receptor) & 
                                   (df['Pose_Rank'] == view_pose)]['PLIP_Contacts_with_Distances'].values
                    
                    color_map = {
                        "H-Bond": "cyanCarbon", "Hydro": "greenCarbon", "Pi-Stack": "purpleCarbon",
                        "Pi-Cat": "purpleCarbon", "SaltBridge": "redCarbon", "Halogen": "yellowCarbon",
                        "Water": "blueCarbon", "Metal": "orangeCarbon"
                    }
                    
                    type_to_resids = {k: [] for k in color_map.keys()}
                    type_to_resids["Other"] = []

                    if len(plip_data) > 0 and str(plip_data[0]) != "nan":
                        contacts = str(plip_data[0]).split(" | ")
                        for contact in contacts:
                            if ":" in contact and "(" in contact:
                                bond_type = contact.split(":")[0].strip()
                                residue = re.search(r':([A-Z]+)(\d+)\(', contact)
                                if residue:
                                    res_id = int(residue.group(2))
                                    if bond_type in type_to_resids:
                                        type_to_resids[bond_type].append(res_id)
                                    else:
                                        type_to_resids["Other"].append(res_id)

                        for b_type, ids in type_to_resids.items():
                            if ids:
                                c_scheme = color_map.get(b_type, "magentaCarbon")
                                view.addStyle({'resi': ids, 'hetflag': False}, 
                                              {'stick': {'colorscheme': c_scheme, 'radius': 0.2}})

                    view.zoomTo({'hetflag': True})
                    showmol(view, height=500, width=520)

                with col_download:
                    st.info("💡 **Ready for Publication?**\n\nDownload this PyMOL package, unzip it, and double-click the `.pml` file to instantly recreate this scene.")
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        zip_file.write(pdb_file, arcname=f"{view_compound}_Ensemble.pdb")
                        zip_file.write(pml_file, arcname=f"{view_compound}_view.pml")
                    st.download_button(label="📥 Download PyMOL Package (.zip)", data=zip_buffer.getvalue(), file_name=f"{view_receptor}_{view_compound}_PyMOL_Session.zip", mime="application/zip", use_container_width=True, type="primary")

# ==========================================
# TAB 3: METHODOLOGY
# ==========================================
with tab3:
    st.header("📖 Computational Methodology & Workflow Architecture")
    st.markdown("This interactive guide outlines the computational algorithms and physical chemistry principles driving the pipeline. It is designed to bridge the gap between experimental pharmacology and *in silico* structural modeling.")
    
    st.divider()
    phase = st.select_slider(
        "**Explore the Pipeline Architecture:**",
        options=[
            "1. Receptor Preparation", "2. Ligand Parameterization", 
            "3. Conformational Sampling", "4. Geometric Profiling", 
            "5. Experimental Correlation", "6. Protocol Validation (RMSD)"
        ]
    )
    st.divider()
    
    if phase == "1. Receptor Preparation":
        st.subheader("Phase 1: Receptor Preparation & Protonation")
        st.info("**The Physical Problem:**\nExperimental structures (X-ray crystallography or cryo-EM) provide heavy-atom coordinates but cannot resolve the positions of hydrogen atoms due to low electron density. However, non-covalent ligand binding in GPCRs is overwhelmingly driven by hydrogen bonding networks. The ionization state of acidic (Asp/Glu) and basic (Lys/Arg/His) sidechains dictates whether they act as hydrogen bond donors, acceptors, or form salt bridges. Furthermore, these pKa values shift significantly based on the local dielectric constant of the buried protein pocket.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown("""
            **1. Microenvironment pKa Prediction:** 
            The `pdb2pqr` engine, utilizing the **PROPKA** algorithm, calculates the empirical pKa shift for every titratable residue. PROPKA evaluates the exact 3D distance and angle to neighboring atoms, calculating desolvation penalties (which raise the pKa of acids and lower the pKa of bases in hydrophobic environments) and specific hydrogen-bond stabilizations.
            
            **2. State Assignment & Parameterization:** 
            Based on the predicted pKa and the user-defined physiological pH (7.4), hydrogens are explicitly added. The receptor is then converted to the `PDBQT` format, where AutoDock-specific atom types ($t$) and Gasteiger partial charges ($q$) are assigned.
            
            **3. Data Sanitization Protocol (The HETATM Rule):** 
            To prevent "ligand collision," the pipeline utilizes strict PDB nomenclature rules. The engine is configured to recognize standard amino acids (`ATOM` records) while systematically stripping non-standard heteroatoms (`HETATM`). This guarantees the automatic removal of the co-crystallized native ligand, explicit water networks ($HOH$), and crystallization buffers (e.g., lipids, sulfates) prior to docking, ensuring a pristine apo-pocket.
            """)
            
    elif phase == "2. Ligand Parameterization":
        st.subheader("Phase 2: Ligand Parameterization & 3D Embedding")
        st.info("**The Physical Problem:**\nVirtual screening libraries are typically stored as 1D SMILES strings. These formats lack stereochemical depth, conformational torsion angles, and atomic partial charges—all of which are required inputs for a 3D thermodynamic scoring function. If a 2D molecule is simply forced into 3D space without energy minimization, it will suffer from severe internal steric clashes and fail the docking evaluation.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown("""
            **1. Distance Geometry Optimization:** 
            The RDKit cheminformatics suite utilizes distance geometry algorithms to fold the 2D topology into a physically viable 3D coordinate matrix, generating a preliminary spatial structure.
            
            **2. Force Field Minimization:** 
            The molecule is relaxed into its global energy minimum using the **MMFF94** (Merck Molecular Force Field). This classical mechanics approach iteratively adjusts bond lengths, bond angles, and torsional dihedrals to minimize internal steric strain and electrostatic repulsion.
            
            **3. Electronegativity Equalization:** 
            Open Babel calculates **Gasteiger partial charges**. This empirical method distributes electron density across the molecule based on the relative electronegativity and hybridization state of each bonded atom, creating the electrostatic map required for Vina's scoring function.
            """)
            
    elif phase == "3. Conformational Sampling":
        st.subheader("Phase 3: Conformational Sampling (Molecular Docking)")
        st.info("**The Physical Problem:**\nPredicting the binding pose of a ligand involves searching a massively complex multidimensional energy landscape. A flexible ligand possesses three degrees of translational freedom, three degrees of rotational freedom, and numerous internal torsional degrees of freedom. Exploring every possible conformation is computationally intractable.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown(r"""
            AutoDock Vina separates this challenge into two distinct mathematical components:
            
            **1. The Search Algorithm (Sampling):** 
            Vina utilizes an **Iterated Local Search** global optimizer. Conceptually similar to Monte Carlo simulated annealing, the algorithm generates a random initial pose, applies random mutations (translations, rotations, or torsion changes), and calculates the energy. Favorable poses are accepted, while unfavorable poses are evaluated based on the Metropolis criterion, allowing the algorithm to escape local energy minima and converge on the global minimum.
            
            **2. The Empirical Scoring Function:** 
            The calculated binding energy ($\Delta G_{calc}$) is a weighted sum of steric interactions (modeled via a modified Lennard-Jones potential), hydrophobic contacts, and directional hydrogen bonding. It is *empirical* rather than an absolute free energy calculation, meaning the weights are calibrated against known experimental binding affinities.
            """)
            
    elif phase == "4. Geometric Profiling":
        st.subheader("Phase 4: Geometric Profiling (Pharmacophore Mapping)")
        st.info("**The Physical Problem:**\nBecause docking scoring functions are heavily weighted toward steric (van der Waals) interactions, they frequently suffer from a 'size bias.' Large, highly lipophilic molecules can bury themselves in the pocket and achieve an artificially negative $\\Delta G$ score simply through non-specific bulk contacts, overshadowing smaller, highly specific ligands.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown(r"""
            To control for size bias and confirm specific binding, the pipeline incorporates the **Protein-Ligand Interaction Profiler (PLIP)**.
            
            **1. Quantum-Geometric Rules:** 
            Unlike simple distance-based cutoff algorithms, PLIP employs strict geometric rules to identify nine distinct non-covalent interaction classes. For example, a hydrogen bond is only registered if the donor-to-acceptor distance is $< 4.1$ Å *and* the donor-hydrogen-acceptor angle is $> 120^\circ$.
            
            **2. Consensus Scoring:** 
            By tracking these specific pharmacophore interactions across the entire ensemble of generated poses, the pipeline identifies compounds with high interaction stability and ligand efficiency, effectively filtering out non-specific lipophilic binders.
            """)
            
    elif phase == "5. Experimental Correlation":
        st.subheader("Phase 5: Experimental Correlation & Model Validation")
        st.info("**The Physical Problem:**\nAn *in silico* docking pipeline is only scientifically useful if its calculated thermodynamic scores predict reality. A rigid-receptor model inherently assumes that the selected protein conformation is the primary state driving biological affinity for the screened compounds. If this assumption is true, the computational scores must correlate with physical wet-lab data.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown(r"""
            The fundamental equation of binding thermodynamics relates free energy ($\Delta G$) to the dissociation constant ($K_d$):
            
            $$ \Delta G = R T \ln(K_d) $$
            
            **1. Data Transformation:** 
            The pipeline automates this correlation by converting uploaded laboratory $K_d$ values into logarithmic $pK_d$ values ($-\log_{10} K_d$). 
            
            **2. Linear Regression:** 
            It merges this experimental dataset with Vina's calculated binding energies via a shared compound identifier and plots a linear regression. A strong correlation (Structure-Activity Relationship) mathematically validates the use of the selected static receptor conformation for predictive screening.
            """)
            
    elif phase == "6. Protocol Validation (RMSD)":
        st.subheader("Phase 6: The 'Scoring vs. Sampling' Validation Protocol")
        st.info("**The Physical Problem (Scoring vs. Sampling):**\nBefore screening novel structures, a docking protocol must prove it can accurately recreate the known reality of a co-crystallized native ligand. However, due to the lack of explicit desolvation and entropy terms in empirical scoring functions, the pose Vina scores as 'best' (Rank 1) is not always the most structurally accurate. The algorithm may have successfully *sampled* the correct pose, but failed to *score* it as the global minimum.")
        with st.expander("💻 Computational Rationale & Execution", expanded=True):
            st.markdown(r"""
            The pipeline employs a rigorous, multi-pose validation protocol utilizing Automorphism-Corrected Root Mean Square Deviation (RMSD).
            
            **1. Pose Generation & Top-N Evaluation:** 
            The protocol isolates the native co-crystallized ligand, scrambles its coordinates, and blindly redocks it into the target grid to generate an ensemble of top conformers. The pipeline evaluates the entire generated ensemble (Top-N poses) to verify if the search algorithm successfully *sampled* the native geometry, regardless of its thermodynamic rank.
            
            **2. Scaffold-Corrected Alignment (RDKit MCS):** 
            Standard RMSD algorithms fail when evaluating symmetric molecules (automorphism), penalizing identical 3D topologies due to mismatched atom indices (e.g., a flipped phenol ring). The pipeline utilizes the **Maximum Common Substructure (MCS)** algorithm via the `rdFMCS` library to isolate the core heavy-atom scaffold, map the atoms topologically rather than by index, and calculate a strict, in-place RMSD.
            
            **3. Success Criteria:** 
            If any sampled pose achieves a heavy-atom RMSD $\le 2.0$ Å against the experimental crystal structure, the grid box coordinates and search parameters are mathematically validated, proving the algorithm is physically capable of finding the correct binding mode.
            """)

# ==========================================
# TAB 4: VALIDATION PROTOCOL (RMSD)
# ==========================================
with tab4:
    st.header("🧪 Protocol Validation (Crystal Redocking)")
    st.markdown("Prove the thermodynamic validity of your docking parameters. This tab generates multiple poses, calculates the scaffold-corrected RMSD for each, and identifies if the selected docking engine successfully sampled the native conformation (even if it didn't rank it #1).")
    
    if not st.session_state.get('active_pdb'):
        st.warning("⚠️ Please load a receptor in **Tab 1** first!")
    elif not st.session_state.get('has_reference_ligand'):
        st.warning("⚠️ No reference ligand found! Go back to **Tab 1**, select 'Auto-Center on Co-Crystallized Ligand', and input the 3-letter code (e.g., 8K8) to extract it.")
    else:
        st.success("✅ Reference ligand isolated and ready for validation.")
        st.caption(f"Docking engine for this validation: **{docking_engine}** (change it in Tab 1 → Vina Search Parameters).")

        val_rescore = st.checkbox(
            "🧬 Rescore poses with MM/PBSA + MM/GBSA (end-point free energy)",
            value=False,
            help="After sampling, rebuild each pose as a receptor–ligand complex and rescore it "
                 "with AmberTools implicit-solvent MM/PBSA (indi=2) and MM/GBSA (igb=8). The "
                 "ligand is protonated to the docking pH so the electrostatics match the chemistry "
                 "that was docked. Reports which pose each free-energy method selects and its RMSD — "
                 "a physics-based alternative to Vina's raw rank-1. Slow: ~1–2 min per pose.")
        if val_rescore:
            st.caption("⏱️ Adds an AmberTools minimise+score per pose (single-trajectory, no entropy). "
                       "The native ligand's SMILES is taken from the RCSB CCD by its 3-letter code.")

        if st.button(f"⚖️ Run Validation (Sampling {n_poses} Poses)", type="primary", use_container_width=True):
            
            # --- FOLDER ROUTING ---
            val_dir = f"{st.session_state['receptor_name']}_validation"
            os.makedirs(val_dir, exist_ok=True)
            
            ref_pdb = "reference_native_ligand.pdb" # Sourced from Tab 1
            ref_pdbqt = os.path.join(val_dir, "reference_native_ligand.pdbqt")
            docked_pdbqt = os.path.join(val_dir, "validation_docked.pdbqt")
            docked_pdb = os.path.join(val_dir, "validation_docked.pdb")
            
            pdb_filename = st.session_state['active_pdb']
            pdbqt_filename = os.path.join(RECEPTOR_DIR, f"{st.session_state['receptor_name']}.pdbqt")
            
            # --- THE LIVE CONSOLE UI ---
            st.divider()
            st.subheader("🖥️ Live Execution Console")
            
            # Placeholders for dynamic updates
            translation_box = st.empty()
            console_expander = st.expander("View Raw Computational Output", expanded=True)
            with console_expander:
                console_output = st.empty()
            
            # Put the string inside a list to make it mutable across scopes (VS Code fix)
            log_text = [""]

            def update_console(new_text, translation):
                log_text[0] += new_text + "\n"
                translation_box.info(f"**Current Operation:** {translation}")
                console_output.code(log_text[0], language="bash")

            with st.status(f"🧬 Initiating Redocking Protocol (Generating {n_poses} Poses)...", expanded=True) as status:
                
                # --- STEP 1: RECEPTOR PREP ---
                st.write("⚙️ Preparing receptor...")
                if not os.path.exists(pdbqt_filename):
                    update_console("> python3 preptarget.py ...", "Stripping water/lipids and assigning physiological protons (pH 7.4) to the receptor.")
                    prep_result = subprocess.run(["python3", os.path.join(_HERE, "preptarget.py"), "-i", pdb_filename, "-o", pdbqt_filename, "-p", str(ph_level)], capture_output=True, text=True)
                    fix_explosive_charges(pdbqt_filename)
                    update_console(prep_result.stdout, "Receptor successfully parameterized and formatted for AutoDock.")
                
                # --- STEP 2: LIGAND PREP (engine-dependent) ---
                st.write("⚙️ Parameterizing native ligand...")
                if dock_mode == "hydrated":
                    update_console(f"> meeko hydrate {ref_pdb}", "Attaching explicit water molecules to the native ligand for the AutoDock4 water model.")
                    try:
                        ref_mol_h = hydrated.reference_mol_for_hydration(ref_pdb, ph_level, val_dir)
                        n_waters = hydrated.prepare_hydrated_ligand_pdbqt(ref_mol_h, ref_pdbqt)
                    except Exception as e:
                        update_console(f"ERROR: {e}", "CRITICAL FAILURE in ligand preparation.")
                        st.error(f"Hydrated ligand prep failed for this structure: {e}\n\n"
                                 "Tip: switch the Docking Engine (Tab 1) to **Standard (Vina)** to validate this structure.")
                        status.update(label="❌ Ligand Prep Failed", state="error")
                        st.stop()
                    update_console(f"Added {n_waters} explicit water site(s) to the ligand.", "Hydrated ligand parameterized for the AD4 water model.")
                else:
                    update_console(f"> obabel {ref_pdb} -O {ref_pdbqt} --partialcharge gasteiger", "Calculating the electronegativity of the ligand and assigning Gasteiger partial charges to prepare for electrostatic scoring.")
                    obabel_result = subprocess.run(["obabel", ref_pdb, "-O", ref_pdbqt, "-p", str(ph_level), "--partialcharge", "gasteiger"], capture_output=True, text=True)
                    update_console(obabel_result.stderr, "Ligand successfully parameterized.")

                # --- STEP 3: DOCKING (Vina or hydrated AD4) ---
                engine_label = "Hydrated AD4" if dock_mode == "hydrated" else "Vina"
                st.write(f"🚀 Running {engine_label} Conformational Sampling...")
                try:
                    center_coords = [center_x, center_y, center_z]
                    box_dims = [box_x, box_y, box_z]

                    if dock_mode == "hydrated":
                        update_console("> Building AD4 affinity maps + water (W) map (autogrid4)...", "Precomputing the AutoDock4 grids and water map for the validation box.")
                        map_prefix, _ = hydrated.build_ad4_maps(
                            pdb_filename, st.session_state['receptor_name'],
                            center_coords, box_dims,
                            os.path.join(val_dir, "ad4_maps"), str(ph_level))
                        v = Vina(sf_name='ad4', cpu=int(cpu_cores), seed=seed_val)
                        v.load_maps(map_prefix)
                        v.set_ligand_from_file(ref_pdbqt)
                        update_console("AD4 Maps Loaded. Commencing Iterated Local Search...", "Applying random mutations to the ligand's torsion angles to find the lowest energy fit.")
                    else:
                        update_console("> Initializing AutoDock Vina Engine...", "Exploring the 3D binding pocket and generating potential ligand conformations.")
                        v = Vina(sf_name='vina', cpu=int(cpu_cores), seed=seed_val)
                        v.set_receptor(pdbqt_filename)
                        v.set_ligand_from_file(ref_pdbqt)
                        v.compute_vina_maps(center=center_coords, box_size=box_dims)
                        update_console("Vina Maps Computed. Commencing Iterated Local Search...", "Applying random mutations to the ligand's torsion angles to find the lowest energy fit.")

                    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
                    v.write_poses(docked_pdbqt, n_poses=n_poses, overwrite=True)

                    energies = v.energies(n_poses=n_poses)
                    for i, e in enumerate(energies):
                        update_console(f"Pose {i+1}: Binding Energy = {e[0]:.2f} kcal/mol", f"Evaluating thermodynamic favorability. (More negative = tighter binding).")

                    # Hydrated poses carry explicit W (water) atoms; strip them before the
                    # RMSD step so the scaffold match compares ligand-to-ligand only.
                    rmsd_src_pdbqt = docked_pdbqt
                    if dock_mode == "hydrated":
                        rmsd_src_pdbqt = os.path.join(val_dir, "validation_docked_nowater.pdbqt")
                        with open(docked_pdbqt) as fin, open(rmsd_src_pdbqt, "w") as fout:
                            for ln in fin:
                                if not hydrated.is_water_pdbqt_line(ln):
                                    fout.write(ln)
                    subprocess.run(["obabel", rmsd_src_pdbqt, "-O", docked_pdb], capture_output=True)

                except Exception as e:
                    update_console(f"ERROR: {str(e)}", f"CRITICAL FAILURE in {engine_label} Engine.")
                    status.update(label=f"❌ {engine_label} Failed", state="error")
                    st.stop()

                # --- STEP 4: RMSD CALCULATION ---
                st.write("📐 Aligning scaffolds and calculating strict in-place RMSD...")
                update_console("> Executing RDKit rdFMCS.FindMCS()", "Bypassing index errors by mapping the core heavy-atom scaffold to calculate the true structural deviation from the crystal pose.")
                try:
                    # Split the multi-model PDB output into separate strings
                    with open(docked_pdb, 'r') as f:
                        lines = f.readlines()
                    
                    models = []
                    current_model = []
                    for line in lines:
                        if line.startswith("MODEL"):
                            current_model = []
                        elif line.startswith("ENDMDL"):
                            models.append("".join(current_model))
                        elif not line.startswith("MASTER") and not line.startswith("END"):
                            current_model.append(line)
                            
                    if not models and current_model: 
                        models.append("".join(current_model))

                    ref_mol = Chem.MolFromPDBFile(ref_pdb, removeHs=True)
                    
                    best_rmsd = 999.0
                    best_pose_rank = 1
                    best_energy = 0.0
                    best_pose_pdb = ""
                    per_pose_rmsds = [float("nan")] * len(models)  # aligned to models, for rescoring

                    for idx, model_str in enumerate(models):
                        rank = idx + 1
                        temp_pose_file = os.path.join(val_dir, f"temp_val_pose_{rank}.pdb")
                        with open(temp_pose_file, "w") as f:
                            f.write(model_str)
                            
                        dock_mol = Chem.MolFromPDBFile(temp_pose_file, removeHs=True)
                        if dock_mol and ref_mol:
                            res = rdFMCS.FindMCS([ref_mol, dock_mol], completeRingsOnly=True)
                            patt = Chem.MolFromSmarts(res.smartsString)
                            
                            ref_match = ref_mol.GetSubstructMatch(patt)
                            dock_match = dock_mol.GetSubstructMatch(patt)
                            
                            if ref_match and dock_match:
                                ref_conf = ref_mol.GetConformer()
                                dock_conf = dock_mol.GetConformer()
                                
                                sq_dist = 0.0
                                for i, j in zip(ref_match, dock_match):
                                    pos1 = ref_conf.GetAtomPosition(i)
                                    pos2 = dock_conf.GetAtomPosition(j)
                                    sq_dist += (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2
                                    
                                current_rmsd = np.sqrt(sq_dist / len(ref_match))
                                per_pose_rmsds[idx] = current_rmsd
                                update_console(f"Pose {rank} RMSD: {current_rmsd:.2f} Å", "Measuring physical distance between the docked scaffold and the crystal truth.")

                                if current_rmsd < best_rmsd:
                                    best_rmsd = current_rmsd
                                    best_pose_rank = rank
                                    best_energy = energies[idx][0]
                                    best_pose_pdb = temp_pose_file

                    # --- GARBAGE COLLECTION ---
                    # Silently delete all the split poses that aren't the winner
                    for idx in range(len(models)):
                        rank = idx + 1
                        file_to_check = os.path.join(val_dir, f"temp_val_pose_{rank}.pdb")
                        if file_to_check != best_pose_pdb and os.path.exists(file_to_check):
                            os.remove(file_to_check)

                    if best_rmsd < 999.0:
                        update_console("✅ Protocol Validation Passed.", "The pipeline successfully identified the biologically relevant pose.")
                        translation_box.success("**Execution Complete:** All computational steps finished successfully.")
                        status.update(label="✅ Validation Complete!", state="complete", expanded=False)
                    else:
                        raise ValueError("Could not map the docked scaffolds back to the reference.")
                        
                except Exception as e:
                    update_console(f"ERROR: {str(e)}", "CRITICAL FAILURE in RMSD Calculation.")
                    status.update(label="❌ RMSD Math Failed", state="error")

                # --- STEP 5 (optional): MM/PBSA + MM/GBSA end-point rescoring ---
                rescore_out = None
                if val_rescore and 'best_rmsd' in locals() and best_rmsd < 999.0:
                    st.write("🧬 Rescoring poses with MM/PBSA + MM/GBSA (AmberTools)...")
                    lig_code = st.session_state.get('reference_ligand_code', '')
                    smi = mmpbsa.ccd_smiles(lig_code) if lig_code else None
                    if not smi:
                        update_console(f"WARNING: no CCD SMILES for '{lig_code}' — skipping rescoring.",
                                       "Could not resolve the native ligand's chemistry from its PDB code; rescoring skipped.")
                        st.warning(f"⚠️ Could not fetch SMILES for ligand code '{lig_code}' from the RCSB "
                                   "CCD (offline or unknown code). MM/PBSA rescoring skipped.")
                    else:
                        update_console(f"> AmberTools MM/PBSA+GBSA on {len(models)} poses (pH {ph_level})",
                                       "Parameterising the ligand at the docking pH (AM1-BCC/GAFF2), minimising each "
                                       "complex in implicit solvent, and scoring the end-point free energy.")
                        try:
                            ctx = {"receptor_pdb": pdb_filename, "smiles": smi,
                                   "cache_dir": os.path.join(val_dir, "ligand_params"),
                                   "indi": 2.0, "maxcyc": 250, "n_jobs": 8, "also_gb": True,
                                   "ph": float(ph_level)}
                            rescore_out = validation.rescore_poses(
                                models, per_pose_rmsds, ctx, os.path.join(val_dir, "_rescore"))
                            shutil.rmtree(os.path.join(val_dir, "_rescore"), ignore_errors=True)
                            pb = rescore_out.get("MMPBSA_Rank", 0)
                            gb = rescore_out.get("MMGBSA_Rank", 0)
                            update_console(f"MM/PBSA selected pose {pb}; MM/GBSA selected pose {gb}.",
                                           "End-point free-energy rescoring complete.")
                        except Exception as e:
                            update_console(f"ERROR (rescoring): {e}", "MM/PBSA rescoring failed; docking result unaffected.")
                            st.warning(f"⚠️ MM/PBSA rescoring failed: {e}")
                            rescore_out = None

            # --- DISPLAY THE BEST GEOMETRIC MATCH ---
            if 'best_rmsd' in locals() and best_rmsd < 999.0:
                st.divider()
                st.subheader("🏆 Best Geometric Match Results")
                col_metric, col_view = st.columns([1, 2])
                
                with col_metric:
                    st.metric(label="Lowest Heavy-Atom RMSD", value=f"{best_rmsd:.2f} Å")
                    st.metric(label="Thermodynamic Rank", value=f"Pose {best_pose_rank} of {n_poses}")
                    st.metric(label="Binding Energy", value=f"{best_energy:.2f} kcal/mol")

                    if best_rmsd <= 2.0:
                        st.success(f"✅ **Validation Passed!** Vina successfully sampled the experimental coordinates in Pose {best_pose_rank}.")
                        st.balloons()
                    elif best_rmsd <= 2.5:
                        st.warning("⚠️ **Borderline.** The pose is acceptable, but slightly shifted.")
                    else:
                        st.error("❌ **Validation Failed.** None of the generated poses matched the native structure.")

                    # MM/PBSA + MM/GBSA end-point rescoring outcome (if it was run)
                    if 'rescore_out' in locals() and rescore_out:
                        st.divider()
                        st.markdown("**🧬 End-point free-energy pose selection**")
                        for pre, name in (("MMPBSA", "MM/PBSA (indi=2)"), ("MMGBSA", "MM/GBSA (igb=8)")):
                            rmsd = rescore_out.get(f"{pre}_RMSD", float("nan"))
                            rank = rescore_out.get(f"{pre}_Rank", 0)
                            dG = rescore_out.get(f"{pre}_dG", float("nan"))
                            if rmsd == rmsd and rank:      # not NaN and a pose was selected
                                flag = "✅" if rmsd <= 2.0 else "❌"
                                st.metric(label=f"{flag} {name} → selected pose",
                                          value=f"{rmsd:.2f} Å",
                                          delta=f"pose {rank} · ΔG {dG:.1f} kcal/mol",
                                          delta_color="off")
                            else:
                                st.caption(f"⚠️ {name}: no pose could be scored.")
                        st.caption("Reference-free physics selection (protonated to the docking pH). "
                                   "Compare against Vina's rank-1 above — a lower RMSD means the free-energy "
                                   "term rescued a more native-like pose than the empirical score.")
                        
                with col_view:
                    view = py3Dmol.view(width=600, height=400)
                    
                    with open(ref_pdb, 'r') as f: ref_data = f.read()
                    view.addModel(ref_data, 'pdb')
                    view.setStyle({'model': 0}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.15}})
                    
                    with open(best_pose_pdb, 'r') as f: dock_data = f.read()
                    view.addModel(dock_data, 'pdb')
                    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15}})
                    
                    view.zoomTo()
                    showmol(view, height=400, width=600)
                    st.caption(f"🟢 **Reference (Crystal)** | 🟠 **Docked (Pose {best_pose_rank})**")

# ==========================================
# TAB 5: BATCH VALIDATION (REDOCK SWEEP)
# ==========================================
with tab5:
    st.header("📐 Batch Validation — Redocking Sweep")
    st.markdown(
        "Validate the docking protocol across **multiple structures, random seeds and "
        "exhaustiveness settings in one run**. Each structure's native co-crystal ligand "
        "is redocked and scored against the crystal pose by automorphism-corrected RMSD. "
        "Two metrics are reported per run: **best-pose RMSD** (can the search *sample* the "
        "native pose?) and **rank-1 RMSD** (does scoring *rank* it first?). Success = RMSD "
        f"≤ {validation.SUCCESS_RMSD:.1f} Å."
    )

    batch_files = st.file_uploader("Upload receptor PDB structures (with their co-crystal ligands)",
                                   type=["pdb"], accept_multiple_files=True, key="batch_pdbs")

    if batch_files:
        st.subheader("1. Confirm the native ligand in each structure")
        st.caption("The largest non-water/ion HETATM group is selected by default — change it if needed.")
        lig_choices = {}
        for bf in batch_files:
            text = bf.getvalue().decode("utf-8", errors="ignore")
            cands = validation.candidate_ligand_codes(text)
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"📄 **{bf.name}**")
            if cands:
                opts = [f"{code} ({n} atoms)" for code, n in cands]
                sel = c2.selectbox(f"Ligand — {bf.name}", opts, key=f"biglig_{bf.name}", label_visibility="collapsed")
                lig_choices[bf.name] = sel.split()[0]
            else:
                lig_choices[bf.name] = c2.text_input(f"3-letter ligand code — {bf.name}",
                                                     key=f"biglig_{bf.name}", label_visibility="collapsed")

        st.subheader("2. Sweep settings")
        s1, s2, s3 = st.columns(3)
        seeds_str = s1.text_input("Random seeds (comma-separated)", value="1,2,3")
        exh_list = s2.multiselect("Exhaustiveness values", [8, 16, 32, 64, 128], default=[8, 16, 32])
        engine_choice = s3.radio("Docking engine", ["Standard (Vina)", "Hydrated (AD4)", "Both"])
        s4, s5 = st.columns(2)
        batch_box = s4.number_input("Grid box edge (Å)", min_value=10.0, max_value=40.0, value=22.0, step=1.0)
        batch_nposes = s5.slider("Poses generated per run", min_value=3, max_value=20, value=9)

        with st.expander("⚙️ Performance & thermal throttling", expanded=False):
            max_cores = os.cpu_count() or 8
            st.caption("AutoDock Vina pins one CPU core per thread for the entire run, which heats "
                       "the chip and spins up fans. Fewer cores → much cooler and quieter (and slower). "
                       "A cooldown pauses between docks to let the CPU dissipate heat.")
            tc1, tc2 = st.columns(2)
            batch_cpu = tc1.slider("CPU cores for docking", min_value=1, max_value=max_cores,
                                   value=min(4, max_cores),
                                   help="Lower = cooler & quieter but slower. ~4 keeps most laptops quiet.")
            cooldown_s = tc2.number_input("Cooldown between runs (seconds)", min_value=0, max_value=120,
                                          value=0, step=5,
                                          help="Idle pause after each dock so the CPU can cool. Adds wall-clock time.")

        out_folder = st.text_input(
            "Output folder", value="batch_validation",
            help="Where docking artifacts, the PyMOL packages and the results CSV/Prism files are written. "
                 "A full path (e.g. ~/validation_run) or a name relative to the app folder. "
                 "'~' is expanded.")

        batch_rescore = st.checkbox(
            "🧬 Rescore every run with MM/PBSA + MM/GBSA (end-point free energy)",
            value=False,
            help="Adds an MM/PBSA (indi=2) and MM/GBSA (igb=8) pose-selection axis to every run: each "
                 "pose is rebuilt as a receptor–ligand complex, minimised in implicit solvent and scored, "
                 "with the ligand protonated to the docking pH. Native SMILES come from the RCSB CCD by "
                 "3-letter code. MUCH slower (~1–2 min per pose) — the sweep is still crash-safe/resumable, "
                 "so it can be interrupted and continued.")
        if batch_rescore:
            st.caption("⏱️ Every seed×exhaustiveness run gains an AmberTools minimise+score for all its poses. "
                       "The result CSV gains MMPBSA_RMSD/Rank/dG and MMGBSA_RMSD/Rank/dG columns, and the "
                       "analysis below adds them as pose-selection axes.")

        engines = {"Standard (Vina)": [("vina", "Vina")],
                   "Hydrated (AD4)": [("hydrated", "AD4 (hydrated)")],
                   "Both": [("vina", "Vina"), ("hydrated", "AD4 (hydrated)")]}[engine_choice]
        seeds = [int(s) for s in seeds_str.replace(" ", "").split(",") if s.lstrip("-").isdigit()]
        total_runs = len(batch_files) * len(engines) * max(len(seeds), 0) * max(len(exh_list), 0)
        st.info(f"**{total_runs}** docking runs "
                f"({len(batch_files)} structures × {len(engines)} engine(s) × {len(seeds)} seeds × {len(exh_list)} exhaustiveness). "
                "Receptor maps are computed once per structure and reused — but large sweeps still take time.")

        st.caption("💾 Crash-safe & resumable: each run is written to the results CSV the moment it "
                   "finishes, so an interruption loses at most the run in progress. Re-running with the "
                   "**same Output folder** skips runs already done and continues where it stopped. For long "
                   "unattended sweeps that survive sleep/closed browser, use the headless `src/run_batch_validation.py` "
                   "(see README).")
        run_disabled = (total_runs == 0)
        if st.button("⚖️ Run Batch Validation", type="primary", use_container_width=True, disabled=run_disabled):
            work_root = os.path.expanduser(out_folder.strip() or "batch_validation")
            try:
                os.makedirs(work_root, exist_ok=True)
            except Exception as e:
                st.error(f"Cannot create output folder '{work_root}': {e}")
                st.stop()

            # Persist the uploaded receptors to disk, then hand off to the shared,
            # crash-safe sweep core (used identically by the headless CLI runner).
            jobs = []
            for bf in batch_files:
                rec_name = os.path.splitext(bf.name)[0]
                struct_dir = os.path.join(work_root, rec_name)
                os.makedirs(struct_dir, exist_ok=True)
                pdb_path = os.path.join(struct_dir, bf.name)
                with open(pdb_path, "wb") as f:
                    f.write(bf.getbuffer())
                jobs.append({"name": rec_name, "pdb_path": pdb_path,
                             "ligand": lig_choices.get(bf.name, "").strip()})

            # Optional end-point rescoring: resolve each native ligand's SMILES from the CCD
            # up front so the sweep core can parameterise it at the docking pH.
            mmpbsa_cfg = None
            if batch_rescore:
                codes = sorted({j["ligand"] for j in jobs if j["ligand"]})
                smiles_by_ligand, unresolved = {}, []
                with st.spinner("Resolving native-ligand SMILES from the RCSB CCD…"):
                    for c in codes:
                        smi = mmpbsa.ccd_smiles(c)
                        if smi:
                            smiles_by_ligand[c] = smi
                        else:
                            unresolved.append(c)
                if unresolved:
                    st.warning(f"⚠️ No CCD SMILES for {', '.join(unresolved)} — those structures will "
                               "dock normally but be left un-rescored (blank MM/PBSA columns).")
                if smiles_by_ligand:
                    mmpbsa_cfg = {"smiles_by_ligand": smiles_by_ligand,
                                  "cache_dir": os.path.join(work_root, "ligand_params"),
                                  "indi": 2.0, "maxcyc": 250, "n_jobs": int(batch_cpu),
                                  "also_gb": True, "ph": float(ph_level)}
                    st.info(f"🧬 MM/PBSA + MM/GBSA rescoring enabled for {len(smiles_by_ligand)} ligand(s).")
                else:
                    st.error("No ligand SMILES could be resolved — running the sweep WITHOUT rescoring.")

            progress = st.progress(0.0)
            status = st.empty()

            def _prog(done, total, msg):
                progress.progress(min(done / total, 1.0) if total else 1.0)
                status.info(f"🚀 {msg}  ·  [{done}/{total}]")

            csv_path, overlays = validation.run_batch_sweep(
                jobs, work_root, seeds=seeds, exh_list=exh_list, engines=engines,
                box_edge=batch_box, n_poses=batch_nposes, cpu=int(batch_cpu),
                ph=str(ph_level), cooldown=int(cooldown_s), mmpbsa_cfg=mmpbsa_cfg,
                progress_cb=_prog, log_cb=lambda m: st.warning("⚠️ " + m))

            status.empty()
            try:
                bdf_done = pd.read_csv(csv_path)
            except Exception:
                bdf_done = pd.DataFrame()
            if not bdf_done.empty:
                st.session_state["batch_results"] = bdf_done
                st.session_state["batch_overlays"] = overlays
                st.session_state["batch_outdir"] = work_root
                st.success(f"✅ {len(bdf_done)} runs in `{os.path.abspath(csv_path)}`. "
                           "Re-run with the same Output folder to add more without repeating these.")
            else:
                st.error("No successful runs — check the ligand codes and inputs.")

    # ---------- ANALYSIS: combine this session's run + uploaded previous results ----------
    st.divider()
    st.subheader("📊 Analysis — combine & visualise results")
    st.caption("Upload previous `batch_validation_results.csv` file(s) to merge them with this "
               "session's run, so you can add new receptors (or drop bad ones) without re-running "
               "the earlier ones. Everything below updates over the combined, filtered set.")
    prev_csvs = st.file_uploader("Add previous results CSV(s) to combine", type=["csv"],
                                 accept_multiple_files=True, key="batch_combine_csvs")

    ANALYSIS_COLS = ["Structure", "Engine", "Seed", "Exhaustiveness", "Rank1_RMSD",
                     "BestPose_RMSD", "Mean_RMSD", "Top_Energy", "Mean_Energy"]
    frames = []
    if "batch_results" in st.session_state and not st.session_state["batch_results"].empty:
        frames.append(st.session_state["batch_results"])
    for f in (prev_csvs or []):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    bdf = None
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        missing = [c for c in ANALYSIS_COLS if c not in combined.columns]
        for c in missing:
            combined[c] = np.nan
        if missing:
            st.warning(f"Some combined rows are missing columns {missing} (filled with blanks).")
        # Optional interaction-selection + end-point-rescoring columns (absent in older
        # CSVs / un-rescored sweeps — fill, no warning).
        for c in ["Selected_RMSD", "Selected_Rank", "Selected_FPsim",
                  "MMPBSA_RMSD", "MMPBSA_Rank", "MMPBSA_dG",
                  "MMGBSA_RMSD", "MMGBSA_Rank", "MMGBSA_dG"]:
            if c not in combined.columns:
                combined[c] = np.nan
        # Drop exact duplicate runs (same structure/engine/seed/exhaustiveness).
        combined = combined.drop_duplicates(
            subset=["Structure", "Engine", "Seed", "Exhaustiveness"], keep="last")

        all_structs = sorted(combined["Structure"].dropna().unique())
        keep = st.multiselect("Structures to include in the analysis", all_structs,
                              default=all_structs, key="batch_struct_filter",
                              help="Untick mis-identified or out-of-scope structures (e.g. a wrong ligand auto-picked).")
        bdf = combined[combined["Structure"].isin(keep)].copy()
        if not bdf.empty:
            st.caption(f"Analysing **{bdf['Structure'].nunique()}** structures · **{len(bdf)}** runs · "
                       f"engines: {', '.join(sorted(bdf['Engine'].dropna().unique()))}.")
            st.download_button("📥 Download combined results (CSV)",
                               data=bdf.to_csv(index=False).encode("utf-8"),
                               file_name="batch_validation_combined.csv", mime="text/csv")

            # Readable labels + inferred activation state (editable).
            bdf["Label"] = bdf["Structure"].map(short_label)
            detected = {s: activation_state(s) for s in sorted(bdf["Structure"].unique())}
            with st.expander("⚙️ Activation state (inferred from filename — correct any if wrong)"):
                st.caption("Inactive = 'Inactive' in the name; Active = a coupled G-protein (e.g. gnas2/gnai1).")
                state_df = pd.DataFrame([{"Structure": short_label(s), "State": detected[s], "_full": s}
                                         for s in detected])
                edited = st.data_editor(
                    state_df[["Structure", "State"]],
                    column_config={"State": st.column_config.SelectboxColumn(
                        "State", options=["Inactive", "Intermediate", "Active", "Unknown"], required=True)},
                    disabled=["Structure"], hide_index=True, use_container_width=True, key="batch_state_editor")
                override = dict(zip(state_df["_full"], edited["State"]))
            bdf["State"] = bdf["Structure"].map(override)

    # ---------- RESULTS ----------
    if bdf is not None and not bdf.empty:
        thr = validation.SUCCESS_RMSD
        has_sel = "Selected_RMSD" in bdf.columns and bdf["Selected_RMSD"].notna().any()
        has_pb = "MMPBSA_RMSD" in bdf.columns and bdf["MMPBSA_RMSD"].notna().any()
        has_gb = "MMGBSA_RMSD" in bdf.columns and bdf["MMGBSA_RMSD"].notna().any()

        def _pct(s):
            s = s.dropna()
            return round((s <= thr).mean() * 100) if len(s) else float("nan")

        def succ_table(df, by):
            g = df.groupby(by)
            d = {"Runs": g.size(),
                 "Sampling_%": g["BestPose_RMSD"].apply(_pct),
                 "Scoring_%": g["Rank1_RMSD"].apply(_pct)}
            if has_sel:
                d["Selected_%"] = g["Selected_RMSD"].apply(_pct)
            if has_pb:
                d["MMPBSA_%"] = g["MMPBSA_RMSD"].apply(_pct)
            if has_gb:
                d["MMGBSA_%"] = g["MMGBSA_RMSD"].apply(_pct)
            d["Median_best"] = g["BestPose_RMSD"].median().round(2)
            return pd.DataFrame(d).reset_index()

        # ---- GraphPad Prism export helpers (tab-separated; paste straight into Prism) ----
        def _fmt_num(v):
            return "" if v is None or (isinstance(v, float) and v != v) else (
                f"{v:g}" if isinstance(v, float) else str(v))

        def _prism_columns(coldata):
            """Prism COLUMN table: one column per group, values down the rows."""
            cols = list(coldata)
            n = max((len(v) for v in coldata.values()), default=0)
            lines = ["\t".join(cols)]
            for i in range(n):
                lines.append("\t".join(_fmt_num(coldata[c][i]) if i < len(coldata[c]) else ""
                                       for c in cols))
            return "\n".join(lines)

        def _prism_xy(x_label, x_values, series):
            """Prism XY table: first column X, then one Y column per series (shared X rows)."""
            lines = ["\t".join([x_label] + list(series))]
            for i, x in enumerate(x_values):
                lines.append("\t".join([_fmt_num(x)] + [_fmt_num(series[s][i]) for s in series]))
            return "\n".join(lines)

        def _prism_grouped_xy(df, x_col, y_col, group_col, order=None):
            """Prism grouped-XY (scatter coloured by group): a shared X column plus one Y
            column per group; each point's Y sits in its group's column, others blank."""
            groups = order or sorted(df[group_col].dropna().unique(), key=str)
            lines = ["\t".join([x_col] + [str(g) for g in groups])]
            for _, r in df.iterrows():
                cells = [_fmt_num(r[x_col])] + [
                    _fmt_num(r[y_col]) if r[group_col] == g else "" for g in groups]
                lines.append("\t".join(cells))
            return "\n".join(lines)

        def _prism_button(tsv, key, label="📋 Copy for Prism"):
            with st.expander(label):
                st.code(tsv)
                st.download_button("📥 Download (.txt, tab-separated)", data=tsv.encode("utf-8"),
                                   file_name=f"prism_{key}.txt", mime="text/tab-separated-values",
                                   key=f"dl_{key}")

        def _struct_stats(df):
            """Structure-level statistics (the unit is the structure, not the run): per-engine
            mean success rate with a 95% bootstrap CI, plus the paired Wilcoxon for the
            selection-vs-scoring claim. Returns (rate_table_df, wilcoxon_p|None)."""
            rng = np.random.default_rng(0)
            axes = [("Sampling (best)", "BestPose_RMSD"), ("Scoring (rank-1)", "Rank1_RMSD")]
            if has_sel:
                axes.append(("Selected", "Selected_RMSD"))
            if has_pb:
                axes.append(("MM/PBSA", "MMPBSA_RMSD"))
            if has_gb:
                axes.append(("MM/GBSA", "MMGBSA_RMSD"))
            # per (Structure, Engine) success fraction for each axis
            rates = (df.groupby(["Structure", "Engine"])
                       .agg(**{col: (col, lambda s, c=col: (s <= thr).mean()) for _, col in axes})
                       .reset_index())
            rows = []
            for eng, ge in rates.groupby("Engine"):
                for name, col in axes:
                    x = ge[col].dropna().values
                    if not len(x):
                        continue
                    boot = [rng.choice(x, len(x), replace=True).mean() for _ in range(5000)]
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    rows.append({"Engine": eng, "Axis": name, "n_structures": len(x),
                                 "Success_pct": round(x.mean() * 100, 1),
                                 "CI95_low": round(lo * 100), "CI95_high": round(hi * 100)})
            # Pre-specified inferential test: selection vs raw scoring on the PRODUCTION
            # engine (the deployed protocol), structure-level. Vina if present.
            wp = None
            prod = "Vina" if "Vina" in rates["Engine"].values else (
                sorted(rates["Engine"].unique())[0] if len(rates) else None)
            if has_sel and prod is not None:
                paired = rates[rates["Engine"] == prod].dropna(subset=["Selected_RMSD", "Rank1_RMSD"])
                if len(paired) >= 3 and (paired["Selected_RMSD"] != paired["Rank1_RMSD"]).any():
                    try:
                        wp = wilcoxon(paired["Selected_RMSD"], paired["Rank1_RMSD"]).pvalue
                    except Exception:
                        wp = None
            return pd.DataFrame(rows), wp, prod, rates, axes

        # ---------- Overall validation metrics ----------
        st.divider()
        st.subheader("✅ Overall validation metrics")
        st.caption(f"Across all included structures, engines, seeds and exhaustiveness "
                   f"(success = RMSD ≤ {thr:.1f} Å).")
        samp = (bdf["BestPose_RMSD"] <= thr).mean() * 100
        scor = (bdf["Rank1_RMSD"] <= thr).mean() * 100
        suba = (bdf["BestPose_RMSD"] <= 1.0).mean() * 100
        cards = [("Pose found (best ≤2 Å)", f"{samp:.0f}%"),
                 ("Top-pose correct (rank-1 ≤2 Å)", f"{scor:.0f}%")]
        if has_sel:
            selp = (bdf["Selected_RMSD"].dropna() <= thr).mean() * 100
            cards.append(("Interaction-selected (≤2 Å)", f"{selp:.0f}%"))
        if has_pb:
            pbp = (bdf["MMPBSA_RMSD"].dropna() <= thr).mean() * 100
            cards.append(("MM/PBSA-selected (≤2 Å)", f"{pbp:.0f}%"))
        if has_gb:
            gbp = (bdf["MMGBSA_RMSD"].dropna() <= thr).mean() * 100
            cards.append(("MM/GBSA-selected (≤2 Å)", f"{gbp:.0f}%"))
        cards += [("Sub-Å (best ≤1 Å)", f"{suba:.0f}%"),
                  ("Structures", str(bdf["Structure"].nunique())), ("Runs", str(len(bdf)))]
        for col, (lab, val) in zip(st.columns(len(cards)), cards):
            col.metric(lab, val)
        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown("**By engine**")
            st.dataframe(succ_table(bdf, "Engine"), hide_index=True, use_container_width=True)
        with oc2:
            st.markdown("**By activation state × engine**")
            st.dataframe(succ_table(bdf, ["State", "Engine"]), hide_index=True, use_container_width=True)

        # ---------- Structure-level statistics (CIs + the one inferential test) ----------
        st.markdown("**Statistical summary** — success rate with 95% CI, computed with the "
                    "**structure** as the unit of replication")
        srt, wp, wp_eng, _rates, _axes = _struct_stats(bdf)
        if not srt.empty:
            disp = srt.copy()
            disp["Success % (95% CI)"] = disp.apply(
                lambda r: f"{r.Success_pct:.0f}  [{r.CI95_low:.0f}–{r.CI95_high:.0f}]", axis=1)
            st.dataframe(disp[["Engine", "Axis", "n_structures", "Success % (95% CI)"]],
                         hide_index=True, use_container_width=True)
            _prism_button(_prism_columns({f"{r.Engine} {r.Axis}": [r.Success_pct, r.CI95_low, r.CI95_high]
                                          for r in srt.itertuples()}), "stats_ci",
                          "📋 Copy stats summary (rate, CI low, CI high) for Prism")
            # Per-structure success rates (%), aligned by structure across columns — this is
            # the table you run the paired test / CIs on yourself in a Prism Column table.
            _structs = sorted(_rates["Structure"].unique())
            _percol = {}
            for _eng in sorted(_rates["Engine"].unique()):
                _gi = _rates[_rates.Engine == _eng].set_index("Structure")
                for _an, _col in _axes:
                    _percol[f"{_eng} | {_an}"] = [round(_gi[_col].get(s, np.nan) * 100, 1) for s in _structs]
            _prism_button(_prism_columns(_percol), "per_structure_rates",
                          "📋 Copy per-structure success rates (to RUN the test / CIs in Prism)")
            n_struct = bdf["Structure"].nunique()
            msg = (f"Unit = structure (n = {n_struct}), **not** the {len(bdf)} runs (which are "
                   "repeated measures — pooling them would be pseudoreplication). CIs are 95% "
                   "bootstrap over structures.")
            if wp is not None:
                verdict = "significant" if wp < 0.05 else "not significant"
                msg += (f" Paired test — selection vs raw scoring on the "
                        f"production **{wp_eng}** engine (Wilcoxon signed-rank, structure-level): "
                        f"**p = {wp:.3f}** ({verdict}).")
            st.caption(msg)

        # ---------- Secondary: per-structure MEAN RMSD (continuous axis) ----------
        # The success rate above answers "how often ≤2 Å?"; this answers "how far off on
        # average?", rewarding a method that is consistently closer even when both clear or
        # both miss the bar. Reported as a SECONDARY (exploratory) axis; the ≤2 Å rate stays
        # the headline. Structure is still the unit; tests are on the production engine.
        if (has_pb or has_gb) and wp_eng is not None:
            st.markdown("**Secondary — per-structure mean RMSD** (continuous, not thresholded at 2 Å)")
            mean_axes = [("Sampling (best)", "BestPose_RMSD"), ("Scoring (rank-1)", "Rank1_RMSD")]
            if has_sel:
                mean_axes.append(("Selected", "Selected_RMSD"))
            if has_pb:
                mean_axes.append(("MM/PBSA", "MMPBSA_RMSD"))
            if has_gb:
                mean_axes.append(("MM/GBSA", "MMGBSA_RMSD"))
            pe = bdf[bdf["Engine"] == wp_eng]
            pmeans = {col: pe.groupby("Structure")[col].mean() for _, col in mean_axes}
            summ = []
            for lab, col in mean_axes:
                m = pmeans[col].dropna()
                if not len(m):
                    continue
                q1, q3 = np.percentile(m.values, [25, 75])
                summ.append({"Axis": lab, "N_struct": len(m),
                             "Mean-of-means (Å)": round(float(m.mean()), 2),
                             "Median (Å)": round(float(m.median()), 2),
                             "IQR (Å)": f"{q1:.2f}–{q3:.2f}"})
            st.dataframe(pd.DataFrame(summ), hide_index=True, use_container_width=True)
            # Paired Wilcoxon on per-structure means: free-energy axes vs comparators, Holm-corrected.
            fam = [(t, tl) for tl, t in [("MM/PBSA", "MMPBSA_RMSD"), ("MM/GBSA", "MMGBSA_RMSD")]
                   if t in pmeans]
            comps = [(c, cl) for cl, c in [("rank-1", "Rank1_RMSD"), ("Selected", "Selected_RMSD")]
                     if c in pmeans]
            rowsm, pvm = [], []
            for tcol, tl in fam:
                for ccol, cl in comps:
                    j = pd.concat([pmeans[tcol].rename("t"), pmeans[ccol].rename("c")], axis=1).dropna()
                    if len(j) < 3 or (j.t.round(3) == j.c.round(3)).all():
                        rowsm.append({"Comparison": f"{tl} vs {cl}", "n": len(j), "n_disc": 0,
                                      "median ΔRMSD (Å)": np.nan, "p_raw": np.nan})
                        pvm.append(1.0)
                        continue
                    p = float(wilcoxon(j.t, j.c).pvalue)
                    rowsm.append({"Comparison": f"{tl} vs {cl}", "n": len(j),
                                  "n_disc": int((j.t.round(3) != j.c.round(3)).sum()),
                                  "median ΔRMSD (Å)": round(float((j.t - j.c).median()), 2),
                                  "p_raw": round(p, 4)})
                    pvm.append(p)
            if rowsm:
                order = sorted(range(len(pvm)), key=lambda i: pvm[i])
                adjv, prev = [0.0] * len(pvm), 0.0
                for rk, i in enumerate(order):
                    prev = max(prev, min(1.0, (len(pvm) - rk) * pvm[i]))
                    adjv[i] = round(prev, 4)
                for r, av in zip(rowsm, adjv):
                    r["p_holm"] = av
                st.dataframe(pd.DataFrame(rowsm), hide_index=True, use_container_width=True)
                st.caption("Paired Wilcoxon on per-structure **mean** RMSD on the production "
                           f"**{wp_eng}** engine (lower = better), Holm-corrected within this secondary "
                           "family. **median ΔRMSD < 0 ⇒ the free-energy axis is closer to native** than "
                           "the comparator. vs rank-1 = fair (both reference-free); vs Selected = "
                           "oracle-referenced upper bound (fingerprint sees the crystal contacts).")
                # Per-structure means, aligned by structure — for running the test yourself in Prism.
                _sm = sorted(pe["Structure"].unique())
                _prism_button(_prism_columns({lab: [round(pmeans[col].get(s, np.nan), 3) for s in _sm]
                                              for lab, col in mean_axes}),
                              "per_structure_mean_rmsd",
                              "📋 Copy per-structure mean RMSD (Å) for Prism")

        # ---------- Full success-percentage breakdown (every meaningful combination) ----------
        st.markdown("**Full success breakdown** — activation state × engine × exhaustiveness, "
                    f"with **All** marginal rows (success = ≤ {thr:.1f} Å)")

        def full_breakdown(df):
            state_levels = (["All"]
                            + [s for s in ["Inactive", "Active"] if s in df["State"].unique()]
                            + [s for s in sorted(df["State"].unique()) if s not in ("Inactive", "Active")])
            eng_levels = (["All"] + (["Vina"] if "Vina" in df["Engine"].unique() else [])
                          + [e for e in sorted(df["Engine"].unique()) if e != "Vina"])
            exh_levels = ["All"] + sorted(df["Exhaustiveness"].unique())
            rows = []
            for stt in state_levels:
                for eng in eng_levels:
                    for ex in exh_levels:
                        sub = df
                        if stt != "All": sub = sub[sub["State"] == stt]
                        if eng != "All": sub = sub[sub["Engine"] == eng]
                        if ex != "All": sub = sub[sub["Exhaustiveness"] == ex]
                        if sub.empty:
                            continue
                        r = {"State": stt, "Engine": eng, "Exhaustiveness": str(ex),
                             "Runs": len(sub),
                             "Sampling_%": _pct(sub["BestPose_RMSD"]),
                             "Scoring_%": _pct(sub["Rank1_RMSD"])}
                        if has_sel:
                            r["Selected_%"] = _pct(sub["Selected_RMSD"])
                        rows.append(r)
            return pd.DataFrame(rows)

        fb = full_breakdown(bdf)
        pct_cols = [c for c in ["Sampling_%", "Scoring_%", "Selected_%"] if c in fb.columns]
        st.dataframe(
            fb.style.background_gradient(cmap="RdYlGn", subset=pct_cols, vmin=0, vmax=100)
              .format({c: "{:.0f}" for c in pct_cols}),
            hide_index=True, use_container_width=True, height=min(38 * len(fb) + 38, 720))
        st.download_button("📥 Download breakdown (CSV)", data=fb.to_csv(index=False).encode("utf-8"),
                           file_name="success_breakdown.csv", mime="text/csv", key="dl_breakdown")
        st.caption("Each row is one slice of the panel; **All** = pooled across that dimension, so the marginal "
                   "totals (e.g. Active / All / All) and the fully-crossed cells read together. "
                   "Sampling = best-pose ≤ threshold; Scoring = rank-1 ≤ threshold"
                   + ("; Selected = interaction-selected pose ≤ threshold." if has_sel else "."))

        if has_sel:
            st.markdown("**Pose-selection robustness** — success of each selection criterion (≤ 2 Å)")
            byeng = succ_table(bdf, "Engine")
            vlist = [v for v in ["Scoring_%", "Selected_%", "Sampling_%"] if v in byeng.columns]
            mlt = byeng.melt(id_vars=["Engine"], value_vars=vlist, var_name="Criterion", value_name="pct")
            mlt["Criterion"] = mlt["Criterion"].map({"Scoring_%": "Rank-1 (raw score)",
                                                     "Selected_%": "Interaction-selected",
                                                     "Sampling_%": "Best pose (sampling ceiling)"})
            fig = px.bar(mlt, x="Criterion", y="pct", color="Engine", barmode="group",
                         labels={"pct": "≤ 2 Å success (%)"})
            fig.update_yaxes(range=[0, 105])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Compare how often each pose-selection criterion recovers the native pose (≤ 2 Å) "
                       "against the sampling ceiling. The validation uses the full crystal contact fingerprint; "
                       "prospective screening applies the conserved anchor residue(s) via the Tab 2 "
                       "pharmacophore filter.")

        # ---------- Headline summary table (averaged over seeds) ----------
        st.divider()
        st.subheader("📋 Validation Summary")
        st.caption("One row per receptor × engine × exhaustiveness, averaged across the random seeds. "
                   "RMSD in Å (lower = better); energies in kcal/mol (more negative = stronger). "
                   "Rank-1 = top-scored pose; Best = lowest-RMSD pose; Mean = averaged over all poses.")
        state_by_struct = bdf.groupby("Structure")["State"].first()
        agg = bdf.groupby(["Structure", "Engine", "Exhaustiveness"]).agg(
            Seeds=("Seed", "nunique"),
            Avg_Best_RMSD=("BestPose_RMSD", lambda s: round(s.mean(), 2)),
            Median_Best_RMSD=("BestPose_RMSD", lambda s: round(s.median(), 2)),
            IQR_Best=("BestPose_RMSD", lambda s: round(s.quantile(.75) - s.quantile(.25), 2)),
            Avg_Rank1_RMSD=("Rank1_RMSD", lambda s: round(s.mean(), 2)),
            Avg_RMSD=("Mean_RMSD", lambda s: round(s.mean(), 2)),
            Avg_Binding_Energy=("Mean_Energy", lambda s: round(s.mean(), 2)),
            Avg_Top_Binding_Energy=("Top_Energy", lambda s: round(s.mean(), 2)),
            Success_pct=("BestPose_RMSD", lambda s: round((s <= thr).mean() * 100)),
            _maxbest=("BestPose_RMSD", "max"),
        ).reset_index()
        agg["Spike"] = np.where(agg["_maxbest"] - agg["Median_Best_RMSD"] > 1.0, "⚠", "")
        agg = agg.drop(columns=["_maxbest"])
        agg.insert(0, "Receptor", agg["Structure"].map(short_label))
        agg.insert(1, "State", agg["Structure"].map(state_by_struct))
        st.dataframe(
            agg.drop(columns=["Structure"]).style.background_gradient(
                cmap="RdYlGn_r", subset=["Avg_Best_RMSD", "Median_Best_RMSD", "Avg_Rank1_RMSD"]),
            use_container_width=True, hide_index=True)
        st.caption("⚠ in **Spike** flags a row where one seed jumped > 1 Å above the median "
                   "(a stochastic sampling miss — compare median vs mean).")
        st.download_button("📥 Download summary table (CSV)",
                           data=agg.to_csv(index=False).encode("utf-8"),
                           file_name="batch_validation_summary.csv", mime="text/csv")

        # ---------- Robust per-structure summary ----------
        st.markdown("**Robust per-structure summary** — median, IQR and Tukey outliers "
                    "over all seeds × exhaustiveness (unlike the seed-averaged rows above).")

        def _tukey_out(s):
            q1, q3 = s.quantile(.25), s.quantile(.75)
            return int((s > q3 + 1.5 * (q3 - q1)).sum())

        rob = bdf.groupby(["Structure", "Engine"]).agg(
            n=("BestPose_RMSD", "size"),
            Median_Best=("BestPose_RMSD", lambda s: round(s.median(), 2)),
            Q1=("BestPose_RMSD", lambda s: round(s.quantile(.25), 2)),
            Q3=("BestPose_RMSD", lambda s: round(s.quantile(.75), 2)),
            Sampling_pct=("BestPose_RMSD", lambda s: round((s <= thr).mean() * 100)),
            Scoring_pct=("Rank1_RMSD", lambda s: round((s <= thr).mean() * 100)),
            Outliers=("BestPose_RMSD", _tukey_out),
        ).reset_index()
        if has_sel:
            sp = (bdf.groupby(["Structure", "Engine"])["Selected_RMSD"].apply(_pct)
                  .rename("Selected_pct").reset_index())
            rob = rob.merge(sp, on=["Structure", "Engine"], how="left")
        rob.insert(0, "Receptor", rob["Structure"].map(short_label))
        rob.insert(1, "State", rob["Structure"].map(state_by_struct))
        st.dataframe(rob.drop(columns=["Structure"]), hide_index=True, use_container_width=True)
        st.caption("Outliers = runs beyond the upper Tukey fence (Q3 + 1.5·IQR) — the stochastic spikes the box plots show.")

        # ---------- Plots & distributions ----------
        st.divider()
        st.subheader("📈 Plots & distributions")
        n_engines = bdf["Engine"].nunique()
        box_color_by = st.radio(
            "Box plots: colour by", ["Engine", "State × engine"], horizontal=True, key="box_color_by",
            help="State × engine keeps every box on one shared axis and colours by activation state and "
                 "engine: Inactive = blue (Vina) / light blue (AD4); Active = red (Vina) / orange (AD4).")
        state_mode = box_color_by.startswith("State")
        # receptors grouped by state (active vs inactive cluster together on the x-axis)
        struct_order = (bdf.groupby("Label")["State"].first().reset_index()
                        .sort_values(["State", "Label"])["Label"].tolist())
        # combined State × Engine colour key — single shared axis, no faceting
        bdf["StateEngine"] = bdf["State"].astype(str) + " · " + bdf["Engine"].astype(str)

        def _se_color(state, engine):
            vina = "vina" in str(engine).lower()
            if state == "Inactive":
                return "#1f77b4" if vina else "#a6cee3"   # blue / light blue
            if state == "Active":
                return "#d62728" if vina else "#ff7f0e"   # red / orange
            return "#7f7f7f"                              # Unknown fallback

        se_cmap = {se: _se_color(s, e) for s, e, se
                   in bdf[["State", "Engine", "StateEngine"]].drop_duplicates().itertuples(index=False)}
        _eng_ord = ["Vina"] + [e for e in sorted(bdf["Engine"].unique()) if e != "Vina"]
        se_order = [f"{s} · {e}" for s in ["Inactive", "Active", "Unknown"] for e in _eng_ord
                    if f"{s} · {e}" in se_cmap]

        def _box_kw(extra_orders=None):
            """Colour by Engine, or by the combined State × Engine palette on one shared axis."""
            orders = dict(extra_orders or {})
            if state_mode:
                orders["StateEngine"] = se_order
                return dict(color="StateEngine", color_discrete_map=se_cmap, category_orders=orders)
            return dict(color="Engine", **({"category_orders": orders} if orders else {}))

        # ---- in-app figure formatting & export (applies to the box/strip + success plots) ----
        with st.expander("🎛️ Plot formatting & export — size, axes, fonts, save as PNG/SVG"):
            fcol1, fcol2, fcol3 = st.columns(3)
            fig_h = fcol1.number_input("Height (px)", 300, 2000, 500, 50, key="fmt_h")
            fig_w = fcol2.number_input("Width (px, 0 = fit window)", 0, 2400, 0, 50, key="fmt_w")
            font_sz = fcol3.number_input("Font size (px)", 8, 30, 13, 1, key="fmt_font")
            ycol1, ycol2, ycol3 = st.columns(3)
            y_auto = ycol1.checkbox("Auto Y-axis", value=True, key="fmt_yauto",
                                    help="Off = set fixed RMSD limits (best-pose RMSD plots only).")
            y_min = ycol2.number_input("Y min (Å)", 0.0, 50.0, 0.0, 0.5, key="fmt_ymin", disabled=y_auto)
            y_max = ycol3.number_input("Y max (Å)", 0.5, 50.0, 6.0, 0.5, key="fmt_ymax", disabled=y_auto)
            exp_fmt = st.radio("Save format (camera button in the plot toolbar)", ["png", "svg"],
                               horizontal=True, key="fmt_exp")
            st.caption("Drag an axis end to rescale, or click a title/legend to edit it, then use the 📷 "
                       "button to download. Width applies when > 0 (otherwise the plot fits the window).")

        def _fmt_show(fig, rmsd_axis=False, min_height=None):
            """Apply the formatting controls to a figure and render it with an editable,
            high-res PNG/SVG export. rmsd_axis=True honours the fixed Y limits."""
            h = max(int(fig_h), int(min_height or 0))
            # Measure the x-axis categories so the *static* PNG/SVG export gets enough width
            # and a tick angle. On-screen the plot is responsive (fills the window), but the
            # downloaded image defaults to a narrow canvas — so receptor names that fit on
            # screen overlap in the export. Rotate long categorical labels (with
            # automargin so they're never clipped) and size the export to the label count.
            cats, maxlen, has_str = set(), 0, False
            for tr in fig.data:
                xv = getattr(tr, "x", None)
                if xv is None:
                    continue
                for v in xv:
                    cats.add(v); maxlen = max(maxlen, len(str(v)))
                    if isinstance(v, str):
                        has_str = True
            ncat = len(cats)
            crowded = has_str and (maxlen > 4 or ncat > 8)
            if crowded:
                fig.update_xaxes(tickangle=-40, automargin=True)
            fig.update_layout(height=h, font=dict(size=int(font_sz)), width=(int(fig_w) or None))
            if rmsd_axis and not y_auto:
                fig.update_yaxes(range=[float(y_min), float(y_max)])
            exp_w = (int(fig_w) if fig_w
                     else min(2400, max(900, 140 + ncat * 75)) if crowded else 1000)
            img = {"format": exp_fmt, "scale": 3, "height": h, "width": exp_w}
            cfg = {"editable": True, "displaylogo": False, "toImageButtonOptions": img}
            st.plotly_chart(fig, use_container_width=(int(fig_w) == 0), config=cfg)

        def _prism_box(df, x_col, value_col, x_order=None, decimals=None):
            """One Prism column per box the app actually draws — exactly as the figure
            looks. Each x-axis category is split by the active colour grouping
            (State × engine when that 'colour by' is selected, else Engine), in the same
            left-to-right order as the plot. So engines (and states) are never pooled:
            paste the columns into a Prism Column table and you get the same boxes."""
            grp_col = "StateEngine" if state_mode else "Engine"
            grp_order = se_order if state_mode else _eng_ord
            xs = list(x_order) if x_order is not None else sorted(df[x_col].dropna().unique())
            cols = {}
            for x in xs:
                for g in grp_order:
                    s = df[(df[x_col] == x) & (df[grp_col] == g)][value_col].dropna()
                    if len(s):
                        cols[f"{x} · {g}"] = (s.round(decimals) if decimals is not None else s).tolist()
            return _prism_columns(cols)

        def _prism_grouped_box(df, row_col, value_col, row_order=None, col_order=None):
            """Prism GROUPED table with replicate sub-columns: rows = the x-axis category
            (e.g. structures), dataset columns = Engine, replicate sub-columns = the
            individual runs (seeds). Unlike a NESTED table, a Grouped table colours each
            engine dataset independently — so you get structures clustered on the x-axis
            with Vina and AD4 boxes in their own colours. In Prism: New → Grouped → set
            'Enter N replicate values in side-by-side subcolumns' (N = the seed count
            shown in the header), then paste starting at the top-left cell."""
            cols = [c for c in (col_order or _eng_ord) if c in set(df["Engine"])]
            rows = [r for r in (row_order or sorted(df[row_col].dropna().unique()))
                    if r in set(df[row_col])]
            cell = lambda r, c: df[(df[row_col] == r) & (df["Engine"] == c)][value_col].dropna().tolist()
            width = max((len(cell(r, c)) for r in rows for c in cols), default=1) or 1
            head = [""]
            for c in cols:
                head += [str(c)] + [""] * (width - 1)
            lines = ["\t".join(head)]
            for r in rows:
                line = [str(r)]
                for c in cols:
                    vals = [_fmt_num(v) for v in cell(r, c)]
                    line += vals + [""] * (width - len(vals))
                lines.append("\t".join(line))
            return "\n".join(lines)

        pt1, pt2, pt3, pt4 = st.tabs(["Convergence & robustness", "Sampling vs scoring",
                                      "Benchmark curve", "Difficulty, state & engines"])

        with pt1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Success rate vs exhaustiveness** (RMSD ≤ {thr:.1f} Å)")
                succ = bdf.groupby(["Engine", "Exhaustiveness"]).agg(
                    Best=("BestPose_RMSD", lambda s: round((s <= thr).mean() * 100, 1)),
                    Rank1=("Rank1_RMSD", lambda s: round((s <= thr).mean() * 100, 1)),
                ).reset_index()
                long = succ.melt(id_vars=["Engine", "Exhaustiveness"], value_vars=["Best", "Rank1"],
                                 var_name="Metric", value_name="Success_pct")
                long["Metric"] = long["Metric"].map({"Best": "Best pose (sampling)", "Rank1": "Rank-1 (scoring)"})
                fig = px.line(long, x="Exhaustiveness", y="Success_pct", color="Engine", line_dash="Metric",
                              markers=True, labels={"Success_pct": "Success rate (%)"})
                fig.update_yaxes(range=[-5, 105]); fig.update_xaxes(type="category")
                st.plotly_chart(fig, use_container_width=True)
                _exhs = sorted(bdf["Exhaustiveness"].unique())
                _ser = {}
                for _e in sorted(bdf["Engine"].unique()):
                    for _m, _l in [("Best", "Best"), ("Rank1", "Rank-1")]:
                        sub = succ[succ.Engine == _e].set_index("Exhaustiveness")[_m]
                        _ser[f"{_e} {_l}"] = [sub.get(x) for x in _exhs]
                _prism_button(_prism_xy("Exhaustiveness", _exhs, _ser), "succ_vs_exh")
            with c2:
                st.markdown("**Best-pose RMSD distribution by exhaustiveness** (box-and-whisker)")
                fig = px.box(bdf, x="Exhaustiveness", y="BestPose_RMSD", points="all",
                             labels={"BestPose_RMSD": "Best-pose RMSD (Å)"}, **_box_kw())
                fig.add_hline(y=thr, line_dash="dash", line_color="green"); fig.update_xaxes(type="category")
                _fmt_show(fig, rmsd_axis=True)
                _prism_button(_prism_box(bdf, "Exhaustiveness", "BestPose_RMSD"), "rmsd_by_exh",
                              "📋 Copy for Prism (one column per box — matches the plot exactly)")

            # Success rate vs exhaustiveness, broken down by activation state AND engine
            st.markdown("**Success rate vs exhaustiveness — by state & engine**")
            metric_opts = {"Best pose (sampling)": "BestPose_RMSD", "Rank-1 (scoring)": "Rank1_RMSD"}
            if has_sel:
                metric_opts["Interaction-selected"] = "Selected_RMSD"
            sm = st.radio("Success metric", list(metric_opts), horizontal=True, key="succ_state_metric")
            scol = metric_opts[sm]
            sg = (bdf.groupby(["State", "Engine", "Exhaustiveness", "StateEngine"])[scol]
                  .apply(lambda s: round((s <= thr).mean() * 100, 1)).reset_index(name="Success_pct"))
            fig = px.line(sg, x="Exhaustiveness", y="Success_pct", color="StateEngine",
                          color_discrete_map=se_cmap, markers=True,
                          category_orders={"StateEngine": se_order,
                                           "Exhaustiveness": sorted(bdf["Exhaustiveness"].unique())},
                          labels={"Success_pct": f"{sm} success (%)", "StateEngine": "State · engine"})
            fig.update_yaxes(range=[-5, 105]); fig.update_xaxes(type="category")
            _fmt_show(fig)
            st.caption("Fraction of runs within the success threshold, split by activation state and engine "
                       "(same blue / light-blue / red / orange palette as the box plots).")
            _exhs2 = sorted(bdf["Exhaustiveness"].unique())
            _prism_button(_prism_xy("Exhaustiveness", _exhs2,
                {se: [sg[(sg.StateEngine == se) & (sg.Exhaustiveness == x)]["Success_pct"].iloc[0]
                      if len(sg[(sg.StateEngine == se) & (sg.Exhaustiveness == x)]) else None
                      for x in _exhs2] for se in se_order if se in set(sg.StateEngine)}),
                "succ_state_engine", "📋 Copy mean curves for Prism")
            # Individual points (per-structure success %) so Prism can draw error bars over
            # the structures within each State·engine group.
            _rep = {f"{se} | {lab}": [round((bdf[(bdf.StateEngine == se) & (bdf.Label == lab)
                                                  & (bdf.Exhaustiveness == x)][scol] <= thr).mean() * 100, 1)
                                      for x in _exhs2]
                    for se in se_order if se in set(bdf.StateEngine)
                    for lab in sorted(bdf[bdf.StateEngine == se]["Label"].unique())}
            _prism_button(_prism_xy("Exhaustiveness", _exhs2, _rep), "succ_state_engine_reps",
                          f"📋 Copy individual points for error bars (per structure · {sm})")

            st.markdown("**Best-pose RMSD by structure** (box-and-whisker) — one panel per exhaustiveness, receptors grouped by state")
            fig = px.box(bdf, x="Label", y="BestPose_RMSD", points="all", facet_row="Exhaustiveness",
                         labels={"BestPose_RMSD": "Best-pose RMSD (Å)", "Label": "Receptor"},
                         **_box_kw({"Label": struct_order,
                                    "Exhaustiveness": sorted(bdf["Exhaustiveness"].unique())}))
            fig.add_hline(y=thr, line_dash="dash", line_color="green")
            _fmt_show(fig, rmsd_axis=True, min_height=850)
            st.caption("One box per receptor over all seeds, one row per exhaustiveness value, "
                       "ordered so Active and Inactive cluster. **Colour by → State × engine** keeps both "
                       "engines on the same axis: Inactive = blue (Vina) / light blue (AD4); Active = red "
                       "(Vina) / orange (AD4).")
            _pex = st.selectbox("Prism table — exhaustiveness panel", ["All"] + sorted(bdf["Exhaustiveness"].unique()),
                                key="prism_bystruct_exh")
            _bs = bdf if _pex == "All" else bdf[bdf["Exhaustiveness"] == _pex]
            _prism_button(_prism_box(
                _bs, "Label", "BestPose_RMSD",
                x_order=[lab for lab in struct_order if lab in set(_bs.Label)]),
                f"rmsd_by_structure_exh{_pex}",
                "📋 Copy for Prism — Column table (one column per box, matches the selected panel)")
            _prism_button(_prism_grouped_box(
                _bs, "Label", "BestPose_RMSD",
                row_order=[lab for lab in struct_order if lab in set(_bs.Label)]),
                f"rmsd_by_structure_grouped_exh{_pex}",
                "📋 Copy for Prism — Grouped/Nested table (structures clustered, engines as "
                "colourable datasets)")

            st.markdown("**Search reproducibility** — best-pose RMSD SD across seeds (high SD = stochastic outliers)")
            sd = bdf.groupby(["Label", "Engine", "Exhaustiveness", "State", "StateEngine"])["BestPose_RMSD"].std(ddof=0).reset_index(name="SD")
            fig = px.strip(sd, x="Exhaustiveness", y="SD", hover_data=["Label"],
                           labels={"SD": "Best-pose RMSD SD across seeds (Å)"}, **_box_kw())
            fig.update_xaxes(type="category")
            _fmt_show(fig)
            _prism_button(_prism_box(sd, "Exhaustiveness", "SD", decimals=3), "reproducibility_sd",
                          "📋 Copy for Prism (one column per box — matches the plot exactly)")

        with pt2:
            st.markdown("**Sampling vs scoring** — every run; best-pose (x) vs top-scored rank-1 (y)")
            _scat_color = st.radio("Colour points by", ["State", "Exhaustiveness", "Engine"],
                                   horizontal=True, key="scatter_color_by")
            _scat_df = bdf.copy()
            _scat_df["Exhaustiveness"] = _scat_df["Exhaustiveness"].astype(str)  # discrete colours
            fig = px.scatter(_scat_df, x="BestPose_RMSD", y="Rank1_RMSD",
                             color=_scat_color, symbol="Engine",
                             category_orders={"Exhaustiveness": [str(x) for x in sorted(bdf["Exhaustiveness"].unique())],
                                              "State": ["Inactive", "Active", "Unknown"]},
                             hover_data=["Label", "Seed", "Exhaustiveness"],
                             labels={"BestPose_RMSD": "Best-pose RMSD (Å)", "Rank1_RMSD": "Rank-1 RMSD (Å)"})
            fig.add_hline(y=thr, line_dash="dash", line_color="green")
            fig.add_vline(x=thr, line_dash="dash", line_color="green")
            _fmt_show(fig)
            st.caption("Bottom-left = sampled **and** correctly ranked. Bottom-right = sampled but **mis-ranked** "
                       "(the scoring gap). Top-right = sampling failure. All runs on one plot — **colour by "
                       "Exhaustiveness** to see the points drift toward the bottom-left as search effort rises "
                       "(engine stays encoded as the marker symbol).")
            _sgrp = st.selectbox("Group scatter columns by (for Prism)",
                ["State", "Engine", "Exhaustiveness", "State × Engine",
                 "Engine × Exhaustiveness", "State × Engine × Exhaustiveness"], key="scatter_group_by")
            _sgcols = {"State": ["State"], "Engine": ["Engine"], "Exhaustiveness": ["Exhaustiveness"],
                       "State × Engine": ["State", "Engine"],
                       "Engine × Exhaustiveness": ["Engine", "Exhaustiveness"],
                       "State × Engine × Exhaustiveness": ["State", "Engine", "Exhaustiveness"]}[_sgrp]
            _stmp = bdf.copy()
            _stmp["_grp"] = _stmp[_sgcols].astype(str).agg(" · ".join, axis=1)
            _prism_button(_prism_grouped_xy(_stmp, "BestPose_RMSD", "Rank1_RMSD", "_grp"),
                          "sampling_vs_scoring_grouped",
                          f"📋 Copy grouped XY scatter for Prism (columns = {_sgrp})")
            _prism_button(
                bdf[["Label", "State", "Engine", "Exhaustiveness", "Seed", "BestPose_RMSD", "Rank1_RMSD"]]
                .to_csv(sep="\t", index=False),
                "sampling_vs_scoring_labeled",
                "📋 Copy labeled table — every point with structure/state/engine/exhaustiveness/seed")
            st.markdown("**Native-pose rank** — where the near-native pose sits in the score ranking")
            fig = px.box(bdf, x="Label", y="Best_Rank", points="all",
                         labels={"Best_Rank": "Native-pose rank (1 = best)", "Label": "Receptor"},
                         **_box_kw({"Label": struct_order}))
            _fmt_show(fig)
            st.caption("Rank 1 = scoring nails it; higher = look deeper down the pose list "
                       "(most informative where the pose was actually sampled).")
            _prism_button(_prism_box(
                bdf, "Label", "Best_Rank",
                x_order=[lab for lab in struct_order if lab in set(bdf.Label)]),
                "native_pose_rank",
                "📋 Copy for Prism (one column per box — matches the plot exactly)")

        with pt3:
            st.markdown("**Cumulative success vs RMSD threshold** — the standard redocking benchmark curve")
            ths = np.round(np.arange(0, 6.01, 0.25), 2)
            exh_order = [str(x) for x in sorted(bdf["Exhaustiveness"].unique())]
            bdown = st.radio("Break down by", ["Engine", "Exhaustiveness", "Engine × Exhaustiveness"],
                             horizontal=True, key="bench_breakdown")
            lbls = {"Threshold": "Best-pose RMSD threshold (Å)", "Success_%": "Cumulative success (%)"}
            if bdown == "Engine":
                cum = pd.DataFrame([{"Engine": e, "Threshold": t,
                                     "Success_%": round((g["BestPose_RMSD"] <= t).mean() * 100, 1)}
                                    for e, g in bdf.groupby("Engine") for t in ths])
                fig = px.line(cum, x="Threshold", y="Success_%", color="Engine", labels=lbls)
            elif bdown == "Exhaustiveness":
                cum = pd.DataFrame([{"Exhaustiveness": str(x), "Threshold": t,
                                     "Success_%": round((g["BestPose_RMSD"] <= t).mean() * 100, 1)}
                                    for x, g in bdf.groupby("Exhaustiveness") for t in ths])
                fig = px.line(cum, x="Threshold", y="Success_%", color="Exhaustiveness", labels=lbls,
                              category_orders={"Exhaustiveness": exh_order})
            else:  # Engine × Exhaustiveness — colour by exhaustiveness, one panel per engine
                cum = pd.DataFrame([{"Engine": e, "Exhaustiveness": str(x), "Threshold": t,
                                     "Success_%": round((g["BestPose_RMSD"] <= t).mean() * 100, 1)}
                                    for (e, x), g in bdf.groupby(["Engine", "Exhaustiveness"]) for t in ths])
                fig = px.line(cum, x="Threshold", y="Success_%", color="Exhaustiveness", facet_col="Engine",
                              labels=lbls, category_orders={"Exhaustiveness": exh_order})
            fig.add_vline(x=thr, line_dash="dash", line_color="green", annotation_text=f"{thr:.1f} Å")
            fig.update_yaxes(range=[-5, 105])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Fraction of runs whose best pose is within x Å of the crystal pose. A curve that rises "
                       "steeply and early = a strong protocol. Breaking down **by exhaustiveness** shows whether "
                       "more sampling shifts the convergence curve and at what threshold it saturates.")
            # Prism XY with EVERY variation as a Y column (combined, per engine, per exh, engine×exh)
            def _cum(g): return [round((g["BestPose_RMSD"] <= t).mean() * 100, 1) for t in ths]
            _bcols = {"Combined": _cum(bdf)}
            for _e in sorted(bdf["Engine"].unique()):
                _bcols[_e] = _cum(bdf[bdf.Engine == _e])
            for _x in sorted(bdf["Exhaustiveness"].unique()):
                _bcols[f"exh{_x}"] = _cum(bdf[bdf.Exhaustiveness == _x])
            for _e in sorted(bdf["Engine"].unique()):
                for _x in sorted(bdf["Exhaustiveness"].unique()):
                    _bcols[f"{_e} exh{_x}"] = _cum(bdf[(bdf.Engine == _e) & (bdf.Exhaustiveness == _x)])
            _prism_button(_prism_xy("Threshold (Å)", list(ths), _bcols), "benchmark_curve",
                          "📋 Copy mean curves for Prism (XY: combined · by engine · by exhaustiveness · engine×exh)")
            # Individual per-replicate curves so Prism can draw ERROR BARS (mean ± SD/SEM
            # over the replicate columns). Each column = one replicate's cumulative success.
            st.caption("**Error bars?** Use the export below: each column is one replicate's own "
                       "curve. In Prism, paste into an XY table and set the columns as **replicate "
                       "subcolumns** of a dataset → it draws the mean ± SD/SEM.")
            rb1, rb2 = st.columns(2)
            _runit = rb1.radio("Replicate unit", ["Structure", "Seed"], horizontal=True,
                               key="bench_rep_unit",
                               help="Structure = spread across the test set (the inference unit). "
                                    "Seed = run-to-run reproducibility at fixed settings.")
            _rsel = rb2.selectbox("Series", ["Combined"] + sorted(bdf["Engine"].unique()), key="bench_rep_series")
            _rcol = "Label" if _runit == "Structure" else "Seed"
            _rsrc = bdf if _rsel == "Combined" else bdf[bdf["Engine"] == _rsel]
            _reps = {str(r): _cum(_rsrc[_rsrc[_rcol] == r])
                     for r in sorted(_rsrc[_rcol].dropna().unique(), key=str)}
            _prism_button(_prism_xy("Threshold (Å)", list(ths), _reps), "benchmark_replicates",
                          f"📋 Copy individual points for error bars (per {_runit.lower()} · {_rsel})")

        with pt4:
            perstruct = bdf.groupby(["Label", "Engine"]).agg(
                Best=("BestPose_RMSD", "mean"), Rot=("Rotatable_Bonds", "first"),
                Heavy=("Heavy_Atoms", "first"), State=("State", "first")).reset_index()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Difficulty vs ligand flexibility**")
                fig = px.scatter(perstruct, x="Rot", y="Best", color="State", symbol="Engine",
                                 hover_name="Label", size="Heavy", size_max=18,
                                 labels={"Rot": "Rotatable bonds", "Best": "Mean best-pose RMSD (Å)"})
                fig.add_hline(y=thr, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
                _ptmp = perstruct.copy()
                _ptmp["_grp"] = _ptmp["State"] + " · " + _ptmp["Engine"]
                _prism_button(_prism_grouped_xy(_ptmp, "Rot", "Best", "_grp"),
                              "difficulty_vs_flex", "📋 Copy grouped XY (rot bonds vs mean RMSD, by state·engine)")
                _prism_button(
                    perstruct[["Label", "State", "Engine", "Rot", "Heavy", "Best"]].round(2)
                    .to_csv(sep="\t", index=False),
                    "difficulty_labeled", "📋 Copy labeled table — every structure identified")
            with c2:
                st.markdown("**Success by activation state**")
                ss = succ_table(bdf, ["State", "Engine"]).melt(
                    id_vars=["State", "Engine"], value_vars=["Sampling_%", "Scoring_%"],
                    var_name="Metric", value_name="pct")
                fig = px.bar(ss, x="State", y="pct", color="Engine", barmode="group", facet_col="Metric",
                             labels={"pct": "Success (%)"})
                fig.update_yaxes(range=[0, 105])
                st.plotly_chart(fig, use_container_width=True)
                _sw = (ss.assign(col=ss.Engine + " " + ss.Metric.str.replace("_%", "", regex=False))
                         .pivot(index="State", columns="col", values="pct").reset_index())
                _prism_button(_sw.to_csv(sep="\t", index=False), "success_by_state", "📋 Copy means for Prism")
                # Individual points = per-structure best-pose success %, one column per
                # State·engine, so Prism draws error bars over the structures in each group.
                _bystate = bdf.groupby(["State", "Engine", "Label"])["BestPose_RMSD"].apply(
                    lambda s: round((s <= thr).mean() * 100, 1)).reset_index(name="pct")
                _prism_button(_prism_columns(
                    {f"{st_} {e}": _bystate[(_bystate.State == st_) & (_bystate.Engine == e)]["pct"].tolist()
                     for st_ in sorted(bdf["State"].unique()) for e in sorted(bdf["Engine"].unique())
                     if len(_bystate[(_bystate.State == st_) & (_bystate.Engine == e)])}),
                    "success_by_state_reps", "📋 Copy individual points for error bars (per structure)")
            st.markdown("**Success heatmap** — best-pose success (%) by structure × exhaustiveness")
            heng = st.selectbox("Engine", sorted(bdf["Engine"].unique()), key="batch_heat_engine")
            hd = bdf[bdf["Engine"] == heng]
            piv = hd.pivot_table(index="Label", columns="Exhaustiveness", values="BestPose_RMSD",
                                 aggfunc=lambda s: round((s <= thr).mean() * 100))
            fig = px.imshow(piv, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                            aspect="auto", labels=dict(color="Success %", x="Exhaustiveness", y="Receptor"))
            st.plotly_chart(fig, use_container_width=True)
            _hm = piv.reset_index(); _hm.columns = ["Receptor"] + [f"exh{c}" for c in piv.columns]
            _prism_button(_hm.to_csv(sep="\t", index=False), f"heatmap_{heng}")
            if n_engines >= 2:
                engs = sorted(bdf["Engine"].unique())
                hh = bdf.groupby(["Label", "Engine"])["BestPose_RMSD"].mean().reset_index()
                hp = hh.pivot(index="Label", columns="Engine", values="BestPose_RMSD").reset_index()
                if all(e in hp.columns for e in engs[:2]):
                    st.markdown(f"**Engine head-to-head** — mean best-pose RMSD per structure")
                    fig = px.scatter(hp, x=engs[0], y=engs[1], hover_name="Label",
                                     labels={engs[0]: f"{engs[0]} (Å)", engs[1]: f"{engs[1]} (Å)"})
                    mx = float(np.nanmax(hp[engs[:2]].to_numpy())) * 1.05
                    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="grey"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Below the diagonal → {engs[1]} reproduces that target better; above → {engs[0]} does.")
                    _prism_button(hp[["Label"] + engs[:2]].round(3).to_csv(sep="\t", index=False),
                                  "engine_head_to_head")

        # ---------- PyMOL overlay viewer + export ----------
        overlays = st.session_state.get("batch_overlays", [])
        if overlays:
            st.divider()
            st.subheader("🔬 Docked Pose vs Crystal Reference")
            st.caption("Representative run (first seed, highest exhaustiveness) per receptor. "
                       "Overlays are available only for structures docked in **this session's run** "
                       "(combined CSVs carry no pose coordinates).")
            pick = st.selectbox("Receptor", [o["label"] for o in overlays], key="batch_ov_pick")
            ov = next(o for o in overlays if o["label"] == pick)
            cV, cD = st.columns([3, 1])
            with cV:
                view = py3Dmol.view(width=760, height=480)
                with open(ov["receptor"]) as f:
                    view.addModel(f.read(), "pdb")
                view.setStyle({"model": 0}, {"cartoon": {"color": "#dcdcdc"}})
                for path, scheme in [(ov["reference"], "greenCarbon"),
                                     (ov["best"], "orangeCarbon"),
                                     (ov["rank1"], "magentaCarbon")]:
                    with open(path) as f:
                        view.addModel(f.read(), "pdb")
                    view.setStyle({"model": -1}, {"stick": {"colorscheme": scheme, "radius": 0.18}})
                view.zoomTo({"model": 1})
                showmol(view, height=480, width=760)
                st.caption(f"🟢 Crystal reference  |  🟠 Best-sampled pose (RMSD {ov['best_rmsd']} Å)  |  "
                           f"🟣 Top-scored / rank-1 pose (RMSD {ov['rank1_rmsd']} Å)")
            with cD:
                st.info("Unzip and run any `.pml` in PyMOL (`@file.pml` or File ▸ Run Script) — "
                        "it renders a 300-DPI PNG and saves an editable `.pse`. Comment out the "
                        "render lines at the bottom to adjust the view first.")
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                    for o in overlays:
                        folder = o["label"].replace(" — ", "_").replace(" ", "")
                        for key in ("receptor", "reference", "best", "rank1", "pml"):
                            z.write(o[key], arcname=os.path.join(folder, os.path.basename(o[key])))
                st.download_button("📥 PyMOL overlays (.zip)", data=zbuf.getvalue(),
                                   file_name="batch_validation_pymol.zip", mime="application/zip",
                                   use_container_width=True, type="primary")

        # ---------- Copy to GraphPad Prism ----------
        st.divider()
        st.subheader("📋 Copy to GraphPad Prism")
        st.caption("Choose a metric and a grouping. Paste the tab-separated table into a Prism "
                   "**Column** data table — each column becomes a dataset (→ box-and-whisker, "
                   "scatter, column stats). The copy icon at the top-right of the box copies it all.")
        metric_map = {"Best-pose RMSD": "BestPose_RMSD", "Rank-1 RMSD": "Rank1_RMSD",
                      "Interaction-selected RMSD": "Selected_RMSD",
                      "Mean RMSD (all poses)": "Mean_RMSD", "Native-pose rank": "Best_Rank",
                      "Top binding energy": "Top_Energy", "Mean binding energy": "Mean_Energy"}
        if not has_sel:
            metric_map.pop("Interaction-selected RMSD")
        group_map = {"Exhaustiveness": "Exhaustiveness", "Receptor": "Label",
                     "Activation state": "State", "Engine": "Engine"}
        pc1, pc2 = st.columns(2)
        pm = pc1.selectbox("Metric (raw values → box-and-whisker)", list(metric_map.keys()), key="prism_metric")
        pgl = pc2.selectbox("Columns (group by)", list(group_map.keys()), key="prism_group")
        vcol, pg = metric_map[pm], group_map[pgl]

        groups = sorted(bdf[pg].dropna().unique(), key=lambda x: (isinstance(x, str), x))
        coldata = {str(g): bdf.loc[bdf[pg] == g, vcol].dropna().tolist() for g in groups}
        prism_tsv = _prism_columns(coldata)
        st.code(prism_tsv)
        st.download_button("📥 Download Prism table (.txt, tab-separated)",
                           data=prism_tsv.encode("utf-8"),
                           file_name=f"prism_{vcol}_by_{pg}.txt",
                           mime="text/tab-separated-values")

        with st.expander("Copy the robust per-structure summary (success % / median / IQR) for Prism"):
            st.caption("Paste into a Prism **Grouped** or **Column** table for summary bar/forest plots.")
            st.code(rob.drop(columns=["Structure"]).to_csv(sep="\t", index=False))

        # ---------- Raw data ----------
        st.divider()
        with st.expander("View raw per-run results"):
            st.dataframe(bdf, use_container_width=True, hide_index=True)
        st.download_button("📥 Download full per-run results (CSV)",
                           data=bdf.to_csv(index=False).encode("utf-8"),
                           file_name="batch_validation_results.csv", mime="text/csv",
                           use_container_width=True)
        st.caption("Note: best-pose RMSD probes search **sampling**; rank-1 RMSD probes **scoring**. "
                   "AD4 (hydrated) and Vina are different scoring functions — RMSD is comparable across "
                   "them, but binding energies are **not**.")