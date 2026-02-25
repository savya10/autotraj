import os
import glob
import pandas as pd
from ase.io import read, write
from pathlib import Path

def combine_files(root_dir, output_dir):
    # 1. Setup paths
    root_path = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    output_traj = out_path / 'combined_all.traj'
    output_csv = out_path / 'combined_all.csv'

    # 2. Find all directories containing the specific combined folder structure
    # We look for pattern: */combined/combined.traj
    # Using recursive glob to find the specific files
    traj_files = sorted(root_path.rglob('combined/combined.traj'))
    
    if not traj_files:
        print(f"No files found matching pattern */combined/combined.traj in {root_dir}")
        return

    print(f"Found {len(traj_files)} matching directories. Processing...")

    all_atoms = []
    all_dfs = []

    # 3. Iterate through sorted files to maintain order
    for traj_path in traj_files:
        # Construct the matching csv path
        csv_path = traj_path.with_suffix('.csv')
        
        # Check if the paired csv exists
        if not csv_path.exists():
            print(f"Warning: CSV missing for {traj_path}. Skipping this pair.")
            continue
            
        print(f"Processing: {traj_path.parent.parent.name}")

        # A. Handle Trajectory
        # read(':') reads all frames in the trajectory
        try:
            atoms_list = read(traj_path, index=':')
            all_atoms.extend(atoms_list)
        except Exception as e:
            print(f"Error reading traj {traj_path}: {e}")
            continue

        # B. Handle CSV
        try:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading csv {csv_path}: {e}")

    # 4. Write final outputs
    if all_atoms:
        print(f"Writing {len(all_atoms)} frames to {output_traj}...")
        write(output_traj, all_atoms)
    
    if all_dfs:
        print(f"Writing combined CSV to {output_csv}...")
        # concat stacks them vertically
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)

    print("Done!")

# --- Usage ---
# Change '.' to your search directory if needed
input_directory = '/home/users/industry/ucl/svyagg/scratch/PAPER2_CHARGE_AWARE/FULLY_IONIZED/PBE0/Corrected/validation'  
output_directory = '/home/users/industry/ucl/svyagg/scratch/PAPER2_CHARGE_AWARE/FULLY_IONIZED/PBE0/Corrected/validation/COMBINED'

if __name__ == "__main__":
    combine_files(input_directory, output_directory)