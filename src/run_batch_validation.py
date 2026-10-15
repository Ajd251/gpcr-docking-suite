#!/usr/bin/env python3
"""Headless, resumable, sleep-proof batch redocking-validation runner.

Runs exactly the same sweep as the app's **Batch Validation** tab, but from a
terminal so it survives browser/Streamlit-session death and (by auto-holding the
system awake) keeps running with the screen off. Results are written incrementally
and already-completed runs are skipped, so if it is interrupted you just run the
same command again and it resumes where it stopped.

Sleep behaviour
---------------
On start it launches `caffeinate -ims -w <pid>`, which prevents idle/disk/system
sleep for the lifetime of this process while still allowing the **display to turn
off** (screen black is fine; the CPU keeps working). Caveat: closing the lid on
battery still forces sleep on Apple Silicon regardless — keep the Mac plugged in
with the lid open (the screen can be off). On non-macOS systems `caffeinate` is
simply skipped. Disable with `--no-awake`.

Usage
-----
    conda activate gpcr-docking
    python3 src/run_batch_validation.py --structures panel.csv --out validation_run \
        --seeds 1,5,42,100 --exhaustiveness 8,16,32 --engines vina,hydrated \
        --box 22 --n-poses 16 --cpu 4 --ph 7.4

`panel.csv` — one structure per line (a header row "name,pdb_path,ligand" is
optional and ignored):

    name,pdb_path,ligand
    ADORA2A_4EIY_Inactive,/path/to/4EIY.pdb,ZMA
    ADRB2_2RH1_Inactive,/path/to/2RH1.pdb,CAU

When it finishes, upload the resulting `batch_validation_results.csv` in the app's
Batch Validation tab (the "combine" uploader) to analyse and plot it.
"""
import argparse
import csv as _csv
import os
import subprocess
import sys

import validation

ENGINE_LABELS = {"vina": ("vina", "Vina"),
                 "hydrated": ("hydrated", "AD4 (hydrated)")}


def keep_awake():
    """Prevent idle/system sleep until this process exits (screen may still sleep)."""
    try:
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
        print("[caffeinate] system kept awake until done (screen may turn off). "
              "On battery, keep the lid open.")
    except Exception:
        print("[caffeinate] unavailable — run under `caffeinate -ims python ...` "
              "to stop the Mac sleeping.")


def read_panel(path):
    """Parse the structures CSV into [{name, pdb_path, ligand}], skipping bad rows."""
    jobs = []
    with open(path, newline="") as f:
        for r in _csv.reader(f):
            if len(r) < 3 or not r[0].strip():
                continue
            if r[0].strip().lower() in ("name", "structure"):
                continue  # header
            name, pdb_path, ligand = r[0].strip(), os.path.expanduser(r[1].strip()), r[2].strip()
            if not os.path.isfile(pdb_path):
                print(f"  ! missing PDB for {name}: {pdb_path}")
                continue
            jobs.append({"name": name, "pdb_path": pdb_path, "ligand": ligand})
    return jobs


def main():
    p = argparse.ArgumentParser(
        description="Headless, resumable, sleep-proof batch redocking validation.")
    p.add_argument("--structures", required=True,
                   help="CSV: name,pdb_path,ligand (one structure per line; header optional)")
    p.add_argument("--out", required=True,
                   help="Output folder. Resumes if it already holds a results CSV.")
    p.add_argument("--seeds", default="1,5,42,100", help="comma-separated RNG seeds")
    p.add_argument("--exhaustiveness", default="8,16,32", help="comma-separated exhaustiveness values")
    p.add_argument("--engines", default="vina", help="comma list of: vina, hydrated")
    p.add_argument("--box", type=float, default=22.0, help="grid box edge in Angstrom")
    p.add_argument("--n-poses", type=int, default=16, help="poses generated per run")
    p.add_argument("--cpu", type=int, default=4, help="CPU cores per dock (lower = cooler/quieter)")
    p.add_argument("--ph", default="7.4", help="protonation pH")
    p.add_argument("--cooldown", type=int, default=0, help="seconds to idle between runs (thermal)")
    p.add_argument("--no-overlays", action="store_true", help="skip representative PyMOL overlays")
    p.add_argument("--no-awake", action="store_true", help="do not auto-prevent system sleep")
    a = p.parse_args()

    if not a.no_awake:
        keep_awake()

    jobs = read_panel(a.structures)
    if not jobs:
        sys.exit("No valid structures found in the panel file.")
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    exh = [int(x) for x in a.exhaustiveness.split(",") if x.strip()]
    engines = [ENGINE_LABELS[e.strip()] for e in a.engines.split(",")
               if e.strip() in ENGINE_LABELS]
    if not (seeds and exh and engines):
        sys.exit("Need at least one seed, one exhaustiveness and one valid engine.")

    total = len(jobs) * len(engines) * len(seeds) * len(exh)
    print(f"Panel: {len(jobs)} structures x {len(engines)} engine(s) x "
          f"{len(seeds)} seeds x {len(exh)} exhaustiveness = {total} runs")
    print(f"Output: {os.path.abspath(a.out)}  (resumes if interrupted)\n")

    def prog(done, total, msg):
        print(f"[{done}/{total}] {msg}", flush=True)

    def log(msg):
        print(f"  ! {msg}", flush=True)

    csv_path, _ = validation.run_batch_sweep(
        jobs, a.out, seeds=seeds, exh_list=exh, engines=engines,
        box_edge=a.box, n_poses=a.n_poses, cpu=a.cpu, ph=a.ph, cooldown=a.cooldown,
        progress_cb=prog, log_cb=log, make_overlays=not a.no_overlays)

    print(f"\nDone. Results: {os.path.abspath(csv_path)}")
    print("Upload this CSV in the app's Batch Validation tab (combine uploader) to analyse.")


if __name__ == "__main__":
    main()
