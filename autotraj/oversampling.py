import numpy as np
from ase.io import read, write
from collections import defaultdict
import random

# ================= CONFIGURATION =================
# 1. Your existing large training file (defects + whatever else)
DEFECTS_FILE = "/home/users/industry/ucl/svyagg/scratch/polarons_dataset/training/combined/combined.traj" 

# 2. Your file containing JUST the pristine bulk frames (one per system)
BULK_ONLY_FILE = "/home/users/industry/ucl/svyagg/scratch/bulks/bulk_traj_frames/PBE0/combined/combined.traj" 

# 3. Output file
OUTPUT_FILE = "/home/users/industry/ucl/svyagg/scratch/polarons_dataset/training/combined/combined_training_bulk.traj"

# 4. How many copies of each bulk system do you want?
TARGET_COPIES = 5
# =================================================

def main():
    print(f"📖 Reading defect dataset: {DEFECTS_FILE}...")
    defect_atoms = read(DEFECTS_FILE, index=":")
    print(f"   Found {len(defect_atoms)} defect frames.")

    print(f"📖 Reading bulk dataset: {BULK_ONLY_FILE}...")
    bulk_atoms_list = read(BULK_ONLY_FILE, index=":")
    print(f"   Found {len(bulk_atoms_list)} bulk frames.")

    # Container for the new oversampled bulk frames
    oversampled_bulks = []
    
    # Track which systems we found for logging
    systems_found = defaultdict(int)

    print(f"⚙️  Oversampling each bulk system to {TARGET_COPIES} copies...")

    for bulk_atom in bulk_atoms_list:
        formula = bulk_atom.get_chemical_formula()
        systems_found[formula] += 1
        
        # --- THE FIX IS HERE ---
        # We loop to create copies, but we MUST explicitly re-attach the calculator.
        for _ in range(TARGET_COPIES):
            new_copy = bulk_atom.copy()
            
            # ASE .copy() drops the calculator. We must put it back.
            # Since SinglePointCalculator is just static data, 
            # it is safe to point to the same calculator object.
            new_copy.calc = bulk_atom.calc
            
            oversampled_bulks.append(new_copy)
        # -----------------------

    print(f"   Created {len(oversampled_bulks)} new bulk frames across {len(systems_found)} systems.")

    # Combine everything
    print("🔗 Merging datasets...")
    # NOTE: defect_atoms were never copied, so they kept their calculators.
    final_dataset = defect_atoms + oversampled_bulks
    
    # Shuffle!
    print("🔀 Shuffling...")
    random.shuffle(final_dataset)

    print(f"💾 Saving final dataset to {OUTPUT_FILE}...")
    write(OUTPUT_FILE, final_dataset)
    
    print("✅ Done! Summary:")
    print(f"   - Original Defects: {len(defect_atoms)}")
    print(f"   - Added Bulks:      {len(oversampled_bulks)} ({TARGET_COPIES} per system)")
    print(f"   - Total Training:   {len(final_dataset)}")

if __name__ == "__main__":
    main()