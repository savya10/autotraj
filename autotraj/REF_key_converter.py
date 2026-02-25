import argparse
import sys
import numpy as np
from ase.io import read, write
from collections import defaultdict

def convert_labels(input_file, output_file, dryrun=False):
    print(f"Reading trajectory from: {input_file}...")
    try:
        traj = read(input_file, index=':')
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Loaded {len(traj)} frames.\n")

    # Tracking for the report
    missing_report = defaultdict(list)
    modified_count = 0

    for i, atoms in enumerate(traj):
        frame_modified = False
        
        # --- 1. HANDLE FORCES (Per Atom) ---
        # We try to get forces from the calculator (atoms.get_forces())
        # If successful, we save them to atoms.arrays['REF_forces']
        try:
            # this pulls from the calculator attached by unified.py
            forces = atoms.get_forces() 
            
            if not dryrun:
                # Save to arrays (standard for per-atom data)
                atoms.new_array('REF_forces', forces)
                
                # OPTIONAL: Remove the calculator so 'forces' doesn't appear twice?
                # Usually keeping the calculator is fine, but if you strictly want ONLY REF_forces:
                # atoms.calc = None 
            
            frame_modified = True
        except Exception:
            # If no calculator or no forces found
            missing_report['forces'].append(i)

        # --- 2. HANDLE STRESS (Per Frame) ---
        # We try to get stress from the calculator (atoms.get_stress())
        # If successful, we save it to atoms.info['REF_stress']
        try:
            # this pulls from the calculator (already converted to eV/A^3 by unified.py)
            stress = atoms.get_stress()
            
            if not dryrun:
                atoms.info['REF_stress'] = stress
            
            frame_modified = True
        except Exception:
            missing_report['stress'].append(i)

        # --- 3. HANDLE ENERGY (Optional but recommended) ---
        # usually people want REF_energy too
        try:
            energy = atoms.get_potential_energy()
            if not dryrun:
                atoms.info['REF_energy'] = energy
                # frame_modified = True (optional to count this)
        except:
            pass

        if frame_modified:
            modified_count += 1

    # --- Reporting ---
    print("--- Analysis Report ---")
    
    if missing_report:
        print("\nWARNING: The following properties were missing (could not be extracted from Calculator):")
        for key, frames in missing_report.items():
            if len(frames) > 10:
                frame_str = f"{frames[:5]} ... {frames[-5:]} (Total: {len(frames)})"
            else:
                frame_str = str(frames)
            print(f"  - '{key}' missing in frames: {frame_str}")
    else:
        print("\nSuccess: Forces and Stress found in all frames.")

    print("-" * 30)
    if dryrun:
        print(f"DRYRUN COMPLETE. No files written.")
        print(f"If run for real, {modified_count} frames would have REF_ labels added.")
    else:
        if output_file:
            print(f"Writing {len(traj)} frames to {output_file}...")
            # We assume you want to keep the atoms.info/arrays we just added.
            # Note: This will still write the Calculator if we didn't remove it.
            write(output_file, traj)
            print("Done.")
        else:
            print("Error: No output filename provided (use -o).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Calculator results to REF_ labels.")
    parser.add_argument("input", help="Input trajectory file")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("--dryrun", action="store_true", help="Report missing data without writing")

    args = parser.parse_args()

    if not args.dryrun and not args.output:
        parser.error("Output file (-o) is required unless using --dryrun")

    convert_labels(args.input, args.output, args.dryrun)