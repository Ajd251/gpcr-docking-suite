#!/usr/bin/env python3
"""Two-way pre-screen ranking from a single docking run.

Reads a {rec}_Screen_Results.csv produced by run_screen.py and emits TWO ranked
hit-lists from the *same* poses:

  1. NON-interaction-anchored  — compounds ranked by best raw Vina ΔG (what the
     score alone says).  ->  {prefix}_raw_ranked.csv
  2. Interaction-anchored      — for each compound the pose whose receptor-contact
     fingerprint best matches a REFERENCE is selected, and compounds are ranked by
     that fingerprint similarity (guards against raw rank-1 mis-ranking the
     near-native pose).  ->  {prefix}_interaction_ranked.csv

The reference is either the native co-crystal ligand (--ref_pdb + --ref_ligand) or,
for a target with no co-crystal ligand, the conserved orthosteric anchor residues
(--anchors PHE100,ASN200). Uses the same geometric contact fingerprint as the
validation core (validation.contact_fingerprint, 4 Å heavy-atom, Tanimoto).

Example:
  python3 src/prescreen.py --csv REC_Screen_Results.csv --receptor_pdbqt REC.pdbqt \
      --ref_pdb REC.pdb --ref_ligand LIG
"""
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
import validation as V
from dock_utils import base_compound  # shared name-parsing helper


def parse_receptor(pdbqt: str) -> tuple:
    """All receptor residue atoms from a PDBQT: (rids, xyz, resname_map)."""
    rids, xyz, rname = [], [], {}
    for ln in open(pdbqt):
        if ln.startswith(("ATOM", "HETATM")):
            try:
                p = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            ch = ln[21].strip() or "_"
            rid = f"{ch}:{ln[22:26].strip()}"
            rids.append(rid)
            xyz.append(p)
            rname.setdefault(rid, ln[17:20].strip())
    return rids, (np.array(xyz) if xyz else np.zeros((0, 3))), rname


def label(rid: str, rname: dict) -> str:
    """Human-readable residue label from a ``chain:resnum`` id, e.g. ``'PHE100'``."""
    return f"{rname.get(rid, '?')}{rid.split(':')[1]}"


def lig_xyz_by_model(ensemble_path: str) -> dict:
    """{pose_rank: heavy-atom xyz} for the docked ligand (HETATM) in each MODEL."""
    out = {}
    if not os.path.exists(ensemble_path):
        return out
    cur, lig = None, []
    for ln in open(ensemble_path):
        if ln.startswith("MODEL"):
            try:
                cur = int(ln.split()[1])
            except (ValueError, IndexError):
                cur = None
            lig = []
        elif ln.startswith("HETATM"):
            lig.append(ln)
        elif ln.startswith("ENDMDL") and cur is not None:
            out[cur] = V._heavy_xyz("".join(lig))
            cur = None
    return out


def resolve_anchors(tokens: list, rids: list, rname: dict) -> frozenset:
    """Match anchor tokens ('PHE100', '100', 'A:100') to receptor residue ids."""
    sel = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            ch, num = tok.split(":", 1)
            target = f"{ch.strip() or '_'}:{num.strip()}"
            sel.update(r for r in rids if r == target)
        else:
            num = re.sub(r"\D", "", tok)
            nm = re.sub(r"\d", "", tok).upper()
            sel.update(r for r in rids
                       if r.split(":")[1] == num and (not nm or rname.get(r, "").upper() == nm))
    return frozenset(sel)


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-way pre-screen: raw score vs interaction-anchored.")
    ap.add_argument("--csv", required=True, help="{rec}_Screen_Results.csv from run_screen.py")
    ap.add_argument("--receptor_pdbqt", required=True, help="prepared receptor PDBQT used for the screen")
    ap.add_argument("--ref_pdb", help="structure file containing the reference co-crystal ligand")
    ap.add_argument("--ref_ligand", help="3-letter code of the reference ligand within --ref_pdb")
    ap.add_argument("--anchors", help="reference anchor residues (no co-crystal needed), e.g. 'PHE100,ASN200' or 'A:100,A:200'")
    ap.add_argument("--ensembles", help="dir of {variant}_Ensemble.pdb (default: {rec}_docking_results)")
    ap.add_argument("--out_prefix", help="output filename prefix (default: derived from --csv)")
    ap.add_argument("--cutoff", type=float, default=4.0, help="contact distance (Å); default 4.0")
    a = ap.parse_args()

    for f in (a.csv, a.receptor_pdbqt):
        if not os.path.exists(f):
            sys.exit(f"❌ not found: {f}")
    df = pd.read_csv(a.csv)
    required = {"Compound_Name", "SMILES", "Pose_Rank", "Binding_Energy_kcal_mol",
                "Total_Contacts", "PLIP_Contacts_with_Distances"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"❌ CSV missing column(s): {', '.join(sorted(missing))}")
    if df.empty:
        sys.exit("❌ CSV has no rows.")
    df["Base"] = df["Compound_Name"].apply(base_compound)

    rec = df["Receptor"].iloc[0] if "Receptor" in df.columns else os.path.basename(a.csv).split("_")[0]
    ens = a.ensembles or f"{rec}_docking_results"
    prefix = a.out_prefix or os.path.splitext(a.csv)[0].replace("_Screen_Results", "")

    rids, rxyz, rname = parse_receptor(a.receptor_pdbqt)
    if len(rxyz) == 0:
        sys.exit(f"❌ no atoms parsed from {a.receptor_pdbqt}")

    # ---- reference fingerprint ----
    if a.ref_ligand and a.ref_pdb:
        if not os.path.exists(a.ref_pdb):
            sys.exit(f"❌ ref_pdb not found: {a.ref_pdb}")
        code = a.ref_ligand.strip().upper()
        lig = [ln for ln in open(a.ref_pdb)
               if ln.startswith("HETATM") and ln[17:20].strip().upper() == code]
        if not lig:
            sys.exit(f"❌ ligand '{code}' not found as HETATM in {a.ref_pdb}")
        ref = V.contact_fingerprint(rids, rxyz, V._heavy_xyz("".join(lig)), a.cutoff)
        ref_desc = f"co-crystal ligand {code} ({len(lig)} atoms)"
    elif a.anchors:
        ref = resolve_anchors(a.anchors.split(","), rids, rname)
        if not ref:
            sys.exit(f"❌ none of the anchor residues matched the receptor: {a.anchors}")
        ref_desc = f"anchor residues ({a.anchors})"
    else:
        sys.exit("❌ provide a reference: --ref_pdb + --ref_ligand, or --anchors")
    if not ref:
        sys.exit("❌ reference fingerprint is empty (no contacts within cutoff).")

    # ---- per-pose fingerprints + Tanimoto to the reference ----
    fpc = {v: lig_xyz_by_model(f"{ens}/{v}_Ensemble.pdb") for v in df["Compound_Name"].unique()}
    tani, fps = [], []
    for r in df.itertuples():
        xyz = fpc.get(r.Compound_Name, {}).get(int(r.Pose_Rank))
        if xyz is None or len(xyz) == 0:
            tani.append(0.0); fps.append(frozenset())
        else:
            fp = V.contact_fingerprint(rids, rxyz, xyz, a.cutoff)
            tani.append(V._tanimoto(fp, ref)); fps.append(fp)
    df["FP_Tanimoto"] = tani
    df["_fp"] = fps
    if all(t == 0.0 for t in tani):
        print(f"⚠ no pose fingerprints computed — check that {ens}/ has the *_Ensemble.pdb files.",
              file=sys.stderr)

    heavy = {}
    for base, g in df.groupby("Base"):
        m = Chem.MolFromSmiles(str(g["SMILES"].iloc[0]))
        heavy[base] = m.GetNumHeavyAtoms() if m else np.nan

    raw_rows, int_rows = [], []
    for base, g in df.groupby("Base"):
        ha = heavy[base]
        raw = g.loc[g["Binding_Energy_kcal_mol"].idxmin()]
        sel = g.loc[g["FP_Tanimoto"].idxmax()]
        le = lambda e: round(abs(e) / ha, 3) if ha == ha and ha else np.nan
        differs = (raw["Compound_Name"], int(raw["Pose_Rank"])) != (sel["Compound_Name"], int(sel["Pose_Rank"]))
        anchors = sorted(sel["_fp"] & ref, key=lambda r: int(re.sub(r"\D", "", r) or 0))
        raw_rows.append({"Compound": base, "Variant": raw["Compound_Name"], "Pose_Rank": int(raw["Pose_Rank"]),
                         "Binding_Energy_kcal_mol": raw["Binding_Energy_kcal_mol"], "Heavy_Atoms": ha,
                         "Ligand_Efficiency": le(raw["Binding_Energy_kcal_mol"]),
                         "Num_Contacts": int(raw["Total_Contacts"]),
                         "FP_Tanimoto_of_this_pose": round(raw["FP_Tanimoto"], 3)})
        int_rows.append({"Compound": base, "Selected_Variant": sel["Compound_Name"], "Pose_Rank": int(sel["Pose_Rank"]),
                         "Binding_Energy_kcal_mol": sel["Binding_Energy_kcal_mol"],
                         "FP_Tanimoto": round(sel["FP_Tanimoto"], 3),
                         "N_Anchors_Reproduced": len(anchors), "Total_Ref_Anchors": len(ref),
                         "Anchors_Reproduced": ";".join(label(r, rname) for r in anchors),
                         "Heavy_Atoms": ha, "Ligand_Efficiency": le(sel["Binding_Energy_kcal_mol"]),
                         "Differs_From_Raw_Top": differs})

    raw_df = pd.DataFrame(raw_rows).sort_values("Binding_Energy_kcal_mol").reset_index(drop=True)
    raw_df.insert(0, "Rank", raw_df.index + 1)
    int_df = (pd.DataFrame(int_rows)
              .sort_values(["FP_Tanimoto", "Ligand_Efficiency"], ascending=[False, False])
              .reset_index(drop=True))
    int_df.insert(0, "Rank", int_df.index + 1)

    raw_out, int_out = f"{prefix}_raw_ranked.csv", f"{prefix}_interaction_ranked.csv"
    poses_out = f"{prefix}_poses_scored.csv"
    raw_df.to_csv(raw_out, index=False)
    int_df.to_csv(int_out, index=False)
    df.drop(columns=["_fp"]).to_csv(poses_out, index=False)

    print(f"Reference fingerprint: {ref_desc} — {len(ref)} residues:")
    print("  " + ", ".join(label(r, rname) for r in sorted(ref, key=lambda r: int(re.sub(r"\D", "", r) or 0))))
    print(f"\n── NON-interaction-anchored (raw Vina score)  →  {raw_out}")
    print(raw_df.to_string(index=False))
    print(f"\n── Interaction-anchored (fingerprint match)   →  {int_out}")
    print(int_df.to_string(index=False))
    print(f"\n(all poses + FP_Tanimoto → {poses_out})")
    ngap = int(int_df["Differs_From_Raw_Top"].sum())
    if ngap:
        print(f"\n⚠ {ngap}/{len(int_df)} compound(s): the interaction-selected pose differs from the "
              f"raw #1 pose — inspect these before trusting the raw score.")


if __name__ == "__main__":
    main()
