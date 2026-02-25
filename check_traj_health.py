import argparse
import numpy as np
from ase.io import read, Trajectory

def analyze_trajectory(traj_path):
    print(f"--- Analyzing: {traj_path} ---")
    
    try:
        # Open in read mode
        traj = Trajectory(traj_path, mode='r')
    except Exception as e:
        print(f"CRITICAL: Could not open file using ase.io.Trajectory. Error: {e}")
        return

    n_frames = 0
    n_missing_calc = 0
    n_bad_stress_shape = 0
    n_missing_forces = 0
    
    for i, atoms in enumerate(traj):
        n_frames += 1
        has_calc = atoms.calc is not None
        
        # 1. Check Calculator
        if not has_calc:
            n_missing_calc += 1
            if n_missing_calc <= 5: # Only print details for first 5 failures
                print(f"[Frame {i}] MISSING CALCULATOR")
                # Diagnose WHY
                # Check if raw data exists in info/arrays
                has_raw_e = "raw_energy_eV" in atoms.info
                has_forces = "REF_forces" in atoms.arrays or "forces" in atoms.arrays
                
                print(f"    - Has raw_energy_eV in info? {has_raw_e}")
                if has_raw_e:
                    print(f"      -> Value: {atoms.info['raw_energy_eV']}")
                print(f"    - Has forces array? {has_forces}")
                
        # 2. Check Stress Shape (Common crasher)
        # Even if calc exists, bad stress shape in info/calc results causes MACE crashes
        stress = None
        if has_calc:
            try:
                stress = atoms.get_stress()
            except Exception:
                # If get_stress fails, inspect the internal storage
                pass
        
        # If get_stress failed or we just want to check the raw storage
        if stress is None:
            # Check info for raw stress
            raw_s = atoms.info.get("REF_stress") or atoms.info.get("stress")
            if raw_s is not None:
                raw_s = np.array(raw_s)
                if raw_s.shape == (9,):
                    n_bad_stress_shape += 1
                    if n_bad_stress_shape <= 5:
                        print(f"[Frame {i}] BAD STRESS SHAPE: Found (9,), expected (3,3) or (6,).")

    print("\n=== SUMMARY REPORT ===")
    print(f"Total Frames:           {n_frames}")
    print(f"Frames w/o Calculator:  {n_missing_calc}")
    print(f"Frames w/ Bad Stress:   {n_bad_stress_shape}")
    
    if n_missing_calc > 0:
        print("\n[DIAGNOSIS]")
        print("Calculators are usually missing because 'forces' were None during parsing.")
        print("This happens if the force array shape didn't match the atom count,")
        print("or if the parser failed to extract forces from the source file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check why calculators might be missing in a .traj file")
    parser.add_argument("file", help="Path to the .traj file (e.g., combined.traj)")
    args = parser.parse_args()
    
    analyze_trajectory(args.file)