#!/usr/bin/env python3
"""Receptor prep CLI: protonate a cleaned PDB with PROPKA (via pdb2pqr) and
convert it to a Vina-ready PDBQT with OpenBabel.

`propka_protonate` is also imported by hydrated.py so the standard Vina prep and
the hydrated/AD4 path protonate the receptor identically.
"""
import argparse
import subprocess
import os
import sys

def run_command(cmd: list[str], step_name: str) -> bool:
    """Run a subprocess command, printing status; return True on success, False on error."""
    print(f"🔄 Running {step_name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {step_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR during {step_name}:")
        print(e.stderr)
        return False

def propka_protonate(input_pdb: str, output_pqr: str, ph: str = "7.4") -> bool:
    """Protonate a receptor with PDB2PQR + PROPKA at the given pH, writing a PQR.

    Shared by the standard Vina prep (this script) and the hydrated/AD4 path
    (hydrated.build_ad4_maps) so both engines protonate the receptor identically
    (AMBER ff, PROPKA microenvironment pKa, chains kept, HETATM/water stripped).
    Returns True on success.
    """
    pdb2pqr_cmd = [
        "pdb2pqr",
        "--ff=AMBER",
        "--titration-state-method=propka",
        f"--with-ph={ph}",
        "--keep-chain",
        input_pdb,
        output_pqr,
    ]
    return run_command(pdb2pqr_cmd, "PDB2PQR (PROPKA Microenvironment Physics)")

def main() -> None:
    """Parse CLI args and run the two-step receptor prep (PROPKA protonation -> PDBQT conversion)."""
    parser = argparse.ArgumentParser(description="Receptor Prep: PROPKA pKa prediction & OpenBabel formatting.")
    parser.add_argument("-i", "--input", required=True, help="Input cleaned PDB file")
    parser.add_argument("-o", "--output", required=True, help="Output PDBQT file for Vina")
    parser.add_argument("-p", "--ph", default="7.4", help="Physiological pH for PROPKA")
    
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ Cannot find input file: {args.input}")
        sys.exit(1)

    base_name = os.path.splitext(args.input)[0]
    temp_pqr = f"{base_name}_protonated.pqr"

    # Step 1: PDB2PQR with PROPKA (The perfect GPCR physics engine)
    if not propka_protonate(args.input, temp_pqr, args.ph):
        sys.exit(1)

    # Step 2: OpenBabel (Bypassing MGLTools completely)
    # The -xr flag denotes a rigid receptor.
    obabel_cmd = [
        "obabel", 
        temp_pqr, 
        "-O", args.output, 
        "-xr", 
        "--partialcharge", "gasteiger"
    ]

    if not run_command(obabel_cmd, "OpenBabel (PDBQT Conversion)"):
        sys.exit(1)

    # Cleanup
    if os.path.exists(temp_pqr): os.remove(temp_pqr)

    print(f"\n🎯 Pipeline Complete! Your Vina-ready receptor is saved as: {args.output}")

if __name__ == "__main__":
    main()