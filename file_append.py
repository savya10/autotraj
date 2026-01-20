import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description="Append two trajectory files together.")
    parser.add_argument("file1", help="Path to the first .traj file (start)")
    parser.add_argument("file2", help="Path to the second .traj file (end)")
    parser.add_argument("output", help="Path for the combined output .traj file")
    
    args = parser.parse_args()

    print(f"Reading {args.file1}...")
    traj1 = read(args.file1, index=":")
    
    print(f"Reading {args.file2}...")
    traj2 = read(args.file2, index=":")
    
    combined = traj1 + traj2
    
    print(f"Writing {len(combined)} frames to {args.output}...")
    write(args.output, combined)
    print("Done.")

if __name__ == "__main__":
    main()