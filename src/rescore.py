#!/usr/bin/env python3
"""
End-point (MM/PBSA + MM/GBSA) rescoring driver — validation and novel screening.
================================================================================

One code path for every rescoring job. It rescores poses that were already docked (no
re-docking), reading them from a batch-validation run's persisted `poses/` or a
screening ensemble directory, and writes the selected-pose columns.

Key robustness properties (see mmpbsa.py):
  * ligands are protonated to the docking pH (default 7.4) before parameterisation,
    so the physics matches the chemistry that was docked;
  * SMILES comes from the SOURCE — a screening library JSON, or (for validation
    native ligands) the RCSB Chemical Component Dictionary — never assumed;
  * per-pose Amber scratch is deleted as it goes (no multi-GB accumulation);
  * resumable and crash-safe: each finished run is appended+flushed and skipped on
    re-run; a SIGKILL-ed run's orphaned Amber children are recovered with --cleanup.

Usage
-----
    # rescore a batch-validation run's stored poses at pH 7.4
    python3 src/rescore.py stored --run results/validation/my_validation_run

    # a screen whose ligands are novel (SMILES from the library JSON, no CCD)
    python3 src/rescore.py stored --run <dir> --library data/libraries/mylib.json

    # rescore ALREADY-DOCKED prospective screens (ensembles on disk, no re-docking);
    # SMILES is read from each dir's *_Screen_Results.csv. --top-k scores only the
    # best-K Vina poses per variation for a faster ranking.
    python3 src/rescore.py screens --dirs results/screens/RECA_docking_results \
        results/screens/RECB_docking_results --top-k 3

    # fold the rescored columns back into the run's results CSV
    python3 src/rescore.py stored --run <dir> --merge

    # recover Amber children orphaned by a previous `kill -9`
    python3 src/rescore.py --cleanup
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paths  # noqa: E402

_CCD_CACHE: dict = {}


def ccd_smiles(code: str) -> "str | None":
    """Canonical SMILES for a 3-letter ligand code from the RCSB CCD (cached).

    Used only for validation *native* ligands; novel screening supplies SMILES
    directly via --library.
    """
    code = (code or "").strip().upper()
    if not code:
        return None
    if code in _CCD_CACHE:
        return _CCD_CACHE[code]
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
    _CCD_CACHE[code] = smi
    return smi


def _per_pose_rmsds(poses_file: str) -> "list[float]":
    """RMSD per pose from the CSV written alongside the poses at sweep time.

    Nothing geometric is recomputed here, so the selected-pose RMSD stays numerically
    identical to the sweep's own axes.
    """
    side = os.path.splitext(poses_file)[0] + "_rmsd.csv"
    if not os.path.isfile(side):
        return []
    out = []
    for r in csv.DictReader(open(side)):
        try:
            out.append(float(r["RMSD_A"]))
        except (ValueError, KeyError, TypeError):
            out.append(float("nan"))
    return out


def _done_keys(path: str) -> set:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return set()
    try:
        import pandas as pd
        d = pd.read_csv(path)
        return {(str(r.Structure), int(r.Seed), int(r.Exhaustiveness)) for r in d.itertuples()}
    except Exception:
        return set()


def _resolve_smiles(ligand: str, library: dict) -> "str | None":
    """SMILES for a run: library JSON first (novel screening), else the CCD (natives)."""
    if library:
        for key in (ligand, str(ligand).strip()):
            if key in library:
                return library[key]
    return ccd_smiles(ligand)


def cmd_stored(a: argparse.Namespace) -> None:
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import mmpbsa
    import validation

    run_dir = os.path.abspath(a.run)
    results = os.path.join(run_dir, "batch_validation_results.csv")
    out_csv = os.path.join(run_dir, "rescored.csv")

    if a.merge:
        d = pd.read_csv(results)
        r = pd.read_csv(out_csv)
        key = ["Structure", "Seed", "Exhaustiveness"]
        d = d.drop(columns=[c for c in r.columns if c not in key and c in d.columns],
                   errors="ignore")
        d.merge(r, on=key, how="left").to_csv(results, index=False)
        print(f"merged {len(r)} rescored runs into {results}")
        return

    library = json.load(open(a.library)) if a.library else {}
    d = pd.read_csv(results)
    d = d[d.Poses_File.notna() & (d.Poses_File.astype(str) != "")]
    if a.limit:
        d = d.head(a.limit)
    skip = _done_keys(out_csv)
    todo = [r for r in d.itertuples()
            if (str(r.Structure), int(r.Seed), int(r.Exhaustiveness)) not in skip]
    print(f"{len(todo)} runs to rescore ({len(skip)} done) | pH {a.ph} indi {a.indi} "
          f"maxcyc {a.maxcyc} jobs {a.jobs}\n", flush=True)

    cache = os.path.join(run_dir, "ligand_params")
    fields = ["Structure", "Seed", "Exhaustiveness", "Ligand_Net_Charge",
              "MMPBSA_RMSD", "MMPBSA_Rank", "MMPBSA_dG",
              "MMGBSA_RMSD", "MMGBSA_Rank", "MMGBSA_dG", "N_Poses_Scored", "Rescore_Note"]
    new = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()

    try:
        for n, r in enumerate(todo, 1):
            struct, seed, exh = str(r.Structure), int(r.Seed), int(r.Exhaustiveness)
            smi = _resolve_smiles(str(r.Ligand).strip(), library)
            pdb = paths.find_structure_pdb(struct)
            note = ""
            if not smi or not pdb:
                note = "no SMILES" if not smi else "no source PDB"
                w.writerow({"Structure": struct, "Seed": seed, "Exhaustiveness": exh,
                            "N_Poses_Scored": 0, "Rescore_Note": note})
                fh.flush()
                print(f"  [{n}/{len(todo)}] {struct[:34]} s{seed}: skipped ({note})", flush=True)
                continue

            poses_file = paths.resolve(str(r.Poses_File))
            wd = paths.scratch_dir("rescore", os.path.basename(run_dir), f"{struct}_s{seed}_e{exh}")
            models = validation.split_pose_models(poses_file, "vina", wd)
            rmsds = _per_pose_rmsds(poses_file)
            rec = [l for l in open(pdb, errors="ignore") if l.startswith("ATOM")]

            # warm the AM1-BCC cache once, at the correct protonation state
            try:
                mol0 = mmpbsa.ligand_mol_from_pose(validation._pose_as_hetatm(models[0]), smi, ph=a.ph)
                mmpbsa.prepare_ligand_params(mol0, os.path.join(wd, "_warm"), cache_dir=cache)
            except Exception:
                pass

            def _one(im):
                i, block = im
                res = mmpbsa.rescore_pose(
                    rec + validation._pose_as_hetatm(block), smi,
                    os.path.join(wd, f"pose{i + 1}"),
                    ph=a.ph, indi=a.indi, maxcyc=a.maxcyc, cache_dir=cache)
                return i, res

            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                scored = dict(ex.map(_one, list(enumerate(models))))

            row = {"Structure": struct, "Seed": seed, "Exhaustiveness": exh,
                   "N_Poses_Scored": sum(1 for v in scored.values() if v.get("pb_dG") == v.get("pb_dG")),
                   "Ligand_Net_Charge": next((v.get("net_charge") for v in scored.values()
                                              if v.get("net_charge") is not None), "")}
            for mkey, pre in (("pb_dG", "MMPBSA"), ("gb_dG", "MMGBSA")):
                cand = [(i, v[mkey]) for i, v in scored.items() if v.get(mkey) == v.get(mkey)]
                if cand:
                    bi, bdg = min(cand, key=lambda t: t[1])
                    row[f"{pre}_dG"] = round(float(bdg), 2)
                    row[f"{pre}_Rank"] = bi + 1
                    row[f"{pre}_RMSD"] = (round(rmsds[bi], 2)
                                         if bi < len(rmsds) and rmsds[bi] == rmsds[bi] else "")
            if row["N_Poses_Scored"] == 0:
                # surface the first failure reason instead of a silent NaN
                errs = [v.get("error") for v in scored.values() if v.get("error")]
                row["Rescore_Note"] = (errs[0][:80] if errs else "all poses failed")
            w.writerow(row)
            fh.flush()
            print(f"  [{n}/{len(todo)}] {struct[:30]:30s} s{seed}: "
                  f"{row['N_Poses_Scored']}/{len(models)} poses | q={row.get('Ligand_Net_Charge')} | "
                  f"MMPBSA {row.get('MMPBSA_RMSD','-')} A | MMGBSA {row.get('MMGBSA_RMSD','-')} A"
                  f"{'  ['+row['Rescore_Note']+']' if row.get('Rescore_Note') else ''}", flush=True)
    finally:
        fh.close()
    print(f"\nwrote {out_csv}\n  merge:  python3 src/rescore.py stored --run {a.run} --merge")


def cmd_screens(a: argparse.Namespace) -> None:
    """Rescore stored prospective-screen ensembles (no re-docking).

    Each screen directory holds `{var}_Ensemble.pdb` complexes and a
    `{rec}_Screen_Results.csv` that already carries each variation's SMILES — so the
    ligand template comes straight from the screen output (no library file, no CCD).
    Writes/appends the companion `{rec}_MMPBSA.csv` in the same directory using the exact
    schema `run_screen.py --rescore` produces, and skips variations already scored so the
    run is resumable. `--top-k` scores only the best-K Vina poses per variation.
    """
    import glob
    import shutil
    import pandas as pd
    import mmpbsa

    COLS = ["Receptor", "Compound_Name", "Variation", "Ligand_Net_Charge",
            "Best_MMPBSA_Pose", "MMPBSA_dG", "Best_MMGBSA_Pose", "MMGBSA_dG",
            "N_Poses_Scored", "Note"]
    grand = 0
    for d in a.dirs:
        d = os.path.abspath(os.path.expanduser(d))
        srs = glob.glob(os.path.join(d, "*_Screen_Results.csv"))
        if not srs:
            print(f"!  {d}: no *_Screen_Results.csv — skipping", flush=True)
            continue
        rec = os.path.basename(srs[0]).replace("_Screen_Results.csv", "")
        smi_of = {str(r.Compound_Name): r.SMILES for r in
                  pd.read_csv(srs[0])[["Compound_Name", "SMILES"]].drop_duplicates().itertuples()}
        ens_files = sorted(glob.glob(os.path.join(d, "*_Ensemble.pdb")))

        out_csv = os.path.join(d, f"{rec}_MMPBSA.csv")
        done = set()
        if os.path.isfile(out_csv):
            try:
                done = set(pd.read_csv(out_csv)["Variation"].astype(str))
            except Exception:
                done = set()
        todo = [f for f in ens_files
                if os.path.basename(f).replace("_Ensemble.pdb", "") not in done]
        print(f"\n=== {rec}: {len(todo)}/{len(ens_files)} variations to rescore "
              f"(top_k={a.top_k or 'all'}, {a.jobs}-way) ===", flush=True)

        fh = open(out_csv, "a", newline="")
        w = csv.writer(fh)
        if not done:
            w.writerow(COLS)
        scratch_root = paths.scratch_dir("screen_rescore", rec)
        for i, ens in enumerate(todo, 1):
            var = os.path.basename(ens).replace("_Ensemble.pdb", "")
            base = var.split("_var")[0].split("_fallback")[0]
            smi = smi_of.get(var)
            if not smi:
                w.writerow([rec, base, var, "", "", "", "", "", 0, "no SMILES"]); fh.flush(); continue
            wd = os.path.join(scratch_root, var)
            try:
                poses = mmpbsa.rescore_ensemble(ens, smi, wd, ph=a.ph, indi=a.indi,
                                                maxcyc=a.maxcyc, n_jobs=a.jobs, max_poses=a.top_k)
            except Exception as exc:
                w.writerow([rec, base, var, "", "", "", "", "", 0, str(exc)[:80]]); fh.flush()
                shutil.rmtree(wd, ignore_errors=True); continue
            q = next((p.get("net_charge") for p in poses if p.get("net_charge") is not None), "")
            pb = [(p["pose_rank"], p["pb_dG"]) for p in poses if p.get("pb_dG") == p.get("pb_dG")]
            gb = [(p["pose_rank"], p["gb_dG"]) for p in poses if p.get("gb_dG") == p.get("gb_dG")]
            bpb = min(pb, key=lambda t: t[1]) if pb else ("", "")
            bgb = min(gb, key=lambda t: t[1]) if gb else ("", "")
            note = "" if pb else (next((p.get("error") for p in poses if p.get("error")), "no poses") or "no poses")
            w.writerow([rec, base, var, q, bpb[0], round(bpb[1], 2) if pb else "",
                        bgb[0], round(bgb[1], 2) if gb else "", len(pb), note[:80]])
            fh.flush()
            grand += len(pb)
            print(f"  [{i}/{len(todo)}] {var}: MM/PBSA {round(bpb[1], 2) if pb else '—'} "
                  f"MM/GBSA {round(bgb[1], 2) if gb else '—'} kcal/mol (q={q}, "
                  f"{len(pb)} poses){'  ['+note[:40]+']' if note else ''}", flush=True)
            shutil.rmtree(wd, ignore_errors=True)
        fh.close()
        print(f"=== {rec}: wrote {out_csv} ===", flush=True)
    print(f"\nDONE. poses scored this run: {grand}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cleanup", action="store_true",
                    help="kill Amber children orphaned by a previous SIGKILL, then exit")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("stored", help="rescore poses already on disk (no re-docking)")
    s.add_argument("--run", required=True, help="a batch-validation run directory")
    s.add_argument("--ph", type=float, default=7.4, help="ligand protonation pH (match the docking)")
    s.add_argument("--indi", type=float, default=2.0, help="PB solute dielectric")
    s.add_argument("--maxcyc", type=int, default=250, help="minimisation cycles")
    s.add_argument("--jobs", type=int, default=8, help="poses scored in parallel")
    s.add_argument("--library", default=None, help="JSON {ligand/compound: SMILES} for novel screens")
    s.add_argument("--limit", type=int, default=None, help="only the first N runs")
    s.add_argument("--merge", action="store_true", help="merge rescored.csv into the results CSV")

    sc = sub.add_parser("screens", help="rescore stored screening ensembles (no re-docking)")
    sc.add_argument("--dirs", nargs="+", required=True,
                    help="screen result dir(s), each with *_Screen_Results.csv + *_Ensemble.pdb")
    sc.add_argument("--ph", type=float, default=7.4, help="ligand protonation pH (match the docking)")
    sc.add_argument("--indi", type=float, default=2.0, help="PB solute dielectric")
    sc.add_argument("--maxcyc", type=int, default=250, help="minimisation cycles")
    sc.add_argument("--jobs", type=int, default=8, help="poses scored in parallel")
    sc.add_argument("--top-k", dest="top_k", type=int, default=None,
                    help="rescore only the best-K Vina poses per variation (default: all)")
    a = ap.parse_args()

    # env: env's bin on PATH (pdb2pqr/obabel), Amber pgid journal for SIGKILL recovery
    os.environ["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + os.environ.get("PATH", "")
    paths.ensure()
    os.environ.setdefault("MMPBSA_PGID_FILE", os.path.join(paths.SCRATCH, "amber_pgids.txt"))
    import mmpbsa
    if a.cleanup:
        print(f"cleaned up {mmpbsa.cleanup_stale()} orphaned process group(s)")
        return
    if a.cmd in ("stored", "screens"):
        if sys.platform == "darwin":
            try:
                subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
            except Exception:
                pass
        cmd_stored(a) if a.cmd == "stored" else cmd_screens(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
