# GPCR Docking & Analytics Suite

A molecular docking and analysis pipeline for G-protein-coupled receptors (GPCRs), driven from a
Streamlit web app or from the command line.

**Workflow:** fetch or upload a receptor structure → protonate and convert it → dock a library of
small molecules → profile protein–ligand interactions with PLIP → visualise and analyse the results.
Redocking validation and optional end-point (MM/PBSA, MM/GBSA) rescoring are built in.

## Features

- **Two docking engines**
  - **Standard:** AutoDock Vina scoring.
  - **Hydrated:** the AutoDock4 water model (Forli & Olson 2012). Explicit waters are attached to
    each ligand; conserved waters are kept and displaced ones are dropped after docking.
- **Receptor preparation:** PROPKA protonation at a chosen pH (via PDB2PQR), then conversion to PDBQT.
- **Ligand preparation:** RDKit tautomer and stereoisomer enumeration; only the most stable variants
  are docked.
- **Interaction profiling:** PLIP for every pose. Detects H-bonds, hydrophobic contacts, π-stacking,
  salt bridges, π-cation, halogen bonds, water bridges and metal contacts.
- **Analytics dashboard:** ranking, interaction fingerprints, residue-network graphs, pharmacophore
  (anchor-residue) filtering, and comparison against uploaded assay data (pKi).
- **Crystal-redocking validation:** for one structure or a batch. RMSD is automorphism-corrected
  and MCS-based. Each run reports three pose-quality measures: sampling (best pose), scoring
  (rank-1 pose) and interaction-fingerprint selection. Statistics are computed per structure,
  with bootstrap confidence intervals.
- **MM/PBSA + MM/GBSA rescoring** (optional, uses AmberTools) of docked poses.
- **Publication figures:** generated PyMOL `.pml` scripts render 300-DPI PNGs and editable sessions.
  Every chart has a "Copy for Prism" export.

## Repository layout

```
src/
  app.py                   Streamlit UI (5 tabs) — the main entry point
  preptarget.py            receptor prep CLI: PDB -> protonated PDBQT
  run_screen.py            batch docking + PLIP profiling CLI
  hydrated.py              AutoDock4 water-model engine (autogrid4 maps, Meeko hydration, drying)
  validation.py            headless redocking / RMSD / batch-sweep core
  run_batch_validation.py  headless, resumable batch-validation runner
  mmpbsa.py                MM/PBSA + MM/GBSA rescoring core (AmberTools)
  rescore.py               headless rescoring driver for stored poses / screens
  prescreen.py             post-screen ranking: raw score vs interaction-anchored
  dock_utils.py, paths.py  shared helpers and the canonical directory layout
autodock_scripts/          vendored AutoDock Vina helpers (mapwater.py, dry.py)
environment.yml            conda environment
```

At runtime `src/paths.py` creates `data/`, `results/` and `scratch/` working directories under the
repository root. All three are git-ignored.

## Installation

```bash
conda env create -f environment.yml
conda activate gpcr-docking
which python3        # must point inside the gpcr-docking env
```

> Use `python3`. On some setups a bare `python` resolves to a different interpreter.

### Optional: hydrated (AD4 water-model) docking

Hydrated docking needs two extra tools:

- **`autogrid4`** from AutoDock4, e.g. bioconda `autodock`. It is built for osx-64/linux-64; on
  Apple Silicon, install it into a separate osx-64 env, where it runs under Rosetta.
- **`prepare_receptor4.py`** from AutoDockTools / MGLTools or the ADFR suite.

They are found on `PATH`, or you can point to them explicitly:

```bash
export AUTOGRID4_BIN=/path/to/autogrid4
export PREPARE_RECEPTOR4=/path/to/prepare_receptor4.py
```

### Optional: MM/PBSA + MM/GBSA rescoring

Install AmberTools (`conda install -c conda-forge ambertools`), either into the same env or into a
separate one. `AMBERHOME` is taken from the environment variable if set; otherwise it is derived
from the `tleap` on `PATH`, or the active conda env.

```bash
export AMBERHOME=/path/to/env/with/ambertools
```

## Usage

### Web app

```bash
streamlit run src/app.py
```

On macOS you can instead double-click `Launch GPCR Docking.command`. It activates the env named in
`GPCR_DOCKING_ENV` (default `gpcr-docking`) and starts the app.

| Tab | Purpose |
|---|---|
| 1 · Execution Pipeline | fetch a structure (RCSB / GPCRdb) or upload one, set the grid box (manually or centred on a co-crystal ligand), choose the engine, run the screen |
| 2 · Analytics Dashboard | analyse `*_Screen_Results.csv`: rankings, fingerprints, networks, assay comparison, MM/PBSA results |
| 3 · Methodology | reference notes on the methods |
| 4 · Validation | redock the co-crystal ligand of one structure and report the RMSD (pass ≤ 2.0 Å) |
| 5 · Batch Validation | sweep many structures × seeds × exhaustiveness × engines, then analyse and plot |

Run everything from the repository root.

### Command line

```bash
# Receptor prep: cleaned PDB -> Vina-ready PDBQT
python3 src/preptarget.py -i RECEPTOR_clean.pdb -o RECEPTOR.pdbqt -p 7.4

# Dock a JSON library {"name": "SMILES", ...} against a prepared receptor
python3 src/run_screen.py \
  --receptor_pdb RECEPTOR_clean.pdb --receptor_pdbqt RECEPTOR.pdbqt \
  --library library.json \
  --x 12.3 --y 4.5 --z -6.7 --box_x 25 --box_y 25 --box_z 25 \
  --exhaustiveness 32 --n_poses 9 --cpu 8 --seed 42 \
  --mode vina            # or: --mode hydrated --ph 7.4
                         # add --rescore for MM/PBSA + MM/GBSA (slow)

# Headless, resumable batch redocking validation
#   panel.csv rows: name,pdb_path,ligand   e.g. ADORA2A_4EIY_Inactive,/path/to/4EIY.pdb,ZMA
python3 src/run_batch_validation.py --structures panel.csv --out validation_run \
  --seeds 1,2,3 --exhaustiveness 8,16,32 --engines vina,hydrated --box 22 --cpu 4

# Rescore already-docked poses (no re-docking)
python3 src/rescore.py stored  --run validation_run
python3 src/rescore.py screens --dirs RECEPTOR_docking_results --top-k 3

# Rank a screen two ways: raw score vs interaction-anchored
python3 src/prescreen.py --csv RECEPTOR_Screen_Results.csv --receptor_pdbqt RECEPTOR.pdbqt \
  --ref_pdb RECEPTOR.pdb --ref_ligand LIG          # or: --anchors 'ASN100,PHE200'
```

`run_batch_validation.py` and `rescore.py` resume where they stopped. Re-run the same command with
the same output folder, and runs that already finished are skipped.

## Outputs

`run_screen.py` writes, in the working directory (`rec_name` = the receptor PDB basename without
`_clean`):

- **`{rec_name}_Screen_Results.csv`**, with columns `Receptor, Compound_Name, SMILES, Pose_Rank,
  Binding_Energy_kcal_mol, Total_Contacts, PLIP_Contacts_with_Distances, Scoring_Function`
  (`vina` or `ad4`).
  - Each input compound becomes `{name}_var{N}` (top stereo/tautomer variants) or
    `{name}_fallback` (when 3D embedding fails).
  - Contacts are formatted as `Type:ResNameResNum(distÅ)`, joined by ` | `,
    e.g. `H-Bond:ASN254(2.91Å)`.
- **`{rec_name}_docking_results/`**: a multi-model `{var}_Ensemble.pdb` and a `{var}_view.pml` per
  variant, plus `{rec_name}_Master_Overlay.pml`.
- **`{rec_name}_MMPBSA.csv`** (with `--rescore`): the best MM/PBSA and MM/GBSA ΔG per compound,
  and the pose each one selects.

## Important caveats

- **Energies are not comparable across engines.** AutoDock4 (hydrated) and Vina ΔG are on
  different scales. The `Scoring_Function` column tags every row, and the dashboard warns when
  the two are mixed.
- **MM/PBSA and MM/GBSA values are for relative ranking within a series only.** They omit entropy
  and are typically large in magnitude, so they are not absolute affinities.
- **Receptor partial charges are zeroed in the PDBQT** (`fix_explosive_charges`) on purpose. Vina
  ignores receptor charges, and OpenBabel's Gasteiger charges can overflow the PDBQT columns on
  unrelaxed models.
- **Network access** is used to fetch structures from RCSB/GPCRdb, and to look up a ligand's
  SMILES in the RCSB Chemical Component Dictionary when a co-crystal ligand cannot be parsed
  locally. Both fail gracefully offline.
- **PyMOL is not a dependency.** Run the generated `.pml` scripts in your own PyMOL to render figures.

## Citing the underlying tools

If you use this pipeline, please cite the tools it builds on:

- **AutoDock Vina:** Eberhardt et al., *J. Chem. Inf. Model.* 2021; Trott & Olson, *J. Comput. Chem.* 2010
- **AutoDock4 hydrated docking:** Forli & Olson, *J. Med. Chem.* 2012
- **Meeko:** Forli lab, https://github.com/forlilab/Meeko
- **PLIP:** Adasme et al., *Nucleic Acids Res.* 2021
- **PDB2PQR / PROPKA:** Jurrus et al., *Protein Sci.* 2018; Olsson et al., *J. Chem. Theory Comput.* 2011
- **Open Babel:** O'Boyle et al., *J. Cheminform.* 2011
- **RDKit:** https://www.rdkit.org
- **AmberTools / MMPBSA.py:** Case et al., AmberTools; Miller et al., *J. Chem. Theory Comput.* 2012

## Licence

Copyright (C) 2026 Aaron Dexter

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for details.

The vendored scripts in `autodock_scripts/` are from AutoDock Vina (Apache-2.0); see
[autodock_scripts/NOTICE](autodock_scripts/NOTICE).
