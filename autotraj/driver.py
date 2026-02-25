# driver.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib

from tqdm import tqdm

from .utils import path_has_high_energy
from .unified import VaspWorkflow

# --- helper functions ---

def find_calc_folders(root_dir: str, look_for_outcar: bool = False) -> List[str]:
    print(f"DEBUG: Scanning root: {root_dir}")
    calc_folders = []
    scanned_count = 0
    skipped_he = 0
    
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        scanned_count += 1
        if path_has_high_energy(dirpath):
            skipped_he += 1
            continue
        
        if look_for_outcar:
            has_file = any(f.startswith("OUTCAR") for f in filenames)
        else:
            has_file = any(f.startswith("vasprun.xml") for f in filenames)
        
        if has_file:
            calc_folders.append(dirpath)
            
    print(f"DEBUG: Scan complete. Visited {scanned_count} dirs. Found {len(calc_folders)} calc folders.")
    return sorted(calc_folders)


def pick_chunk(items: List[str], chunk_index: int, nchunks: int) -> List[str]:
    if nchunks <= 1:
        return items
    out = []
    for x in items:
        h = int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)
        if (h % nchunks) == chunk_index:
            out.append(x)
    return out


def parse_dielectric(s: str):
    if not s: return None
    s = s.strip()
    if s.lower() == "none": return None
    
    if ";" in s:
        rows = s.split(";")
        mat = [[float(x) for x in r.split(",")] for r in rows]
        return mat
    if "," in s:
        vals = [float(x) for x in s.split(",")]
        if len(vals) == 3:
            return [[vals[0], 0, 0], [0, vals[1], 0], [0, 0, vals[2]]]
        if len(vals) == 9:
            return [vals[0:3], vals[3:6], vals[6:9]]
        raise ValueError("Dielectric with commas must be length 3 or 9")
    return float(s)

# --- Top-level worker function ---
def process_folder_wrapper(folder, bulk_vasprun, bulk_outcar, dielectric, output_root, snb, n_steps, bulk_mode, max_force, skip_corrections, use_outcar_parsing):
    wf = VaspWorkflow(
        bulk_vasprun_path=bulk_vasprun,
        bulk_outcar_path=bulk_outcar,
        dielectric=dielectric,
        output_root=output_root,
        snb=snb,
        n_steps=n_steps,
        bulk_mode=bulk_mode,
        max_force_threshold=max_force,
        skip_corrections=skip_corrections,
        use_outcar_parsing=use_outcar_parsing 
    )
    return wf.process_calculation(folder)


def main():
    p = argparse.ArgumentParser(description="Parse VASP defect trajectories into traj+csv.")
    
    p.add_argument("--root", required=True, help="Root directory.")
    p.add_argument("--output", default=None, help="Output folder.")
    p.add_argument("--bulk-calculation", action="store_true", help="Enable Bulk Mode.")
    
    p.add_argument("--bulk-vasprun", default=None, help="Path to bulk vasprun.")
    p.add_argument("--bulk-outcar", default=None, help="Path to bulk OUTCAR.")
    p.add_argument("--dielectric", default=None, help="Dielectric tensor/scalar.")
    
    p.add_argument("--snb", action="store_true", help="Enable SNB-style parsing.")
    p.add_argument("--workers", type=int, default=1, help="Multiprocessing workers.")
    p.add_argument("--chunk-index", type=int, default=0, help="Chunk index.")
    p.add_argument("--nchunks", type=int, default=1, help="Total chunks.")
    p.add_argument("--merge-only", action="store_true", help="Merge only.")
    
    p.add_argument("--n-steps", type=int, default=10, help="Steps to keep.")
    p.add_argument("--max-force", type=float, default=50.0, help="Force threshold.")
    p.add_argument("--skip-corrections", action="store_true", help="Disable eFNV charge corrections.")
    p.add_argument("--use-outcar-parsing", action="store_true", help="Parse OUTCARs directly (bypassing vasprun.xml).")

    args = p.parse_args()

    # --- VALIDATION LOGIC ---
    is_standard_defect_run = (not args.bulk_calculation) and (not args.skip_corrections) and (not args.use_outcar_parsing)
    
    if is_standard_defect_run:
        missing_v = (not args.bulk_vasprun) or (args.bulk_vasprun == "None")
        missing_o = (not args.bulk_outcar) or (args.bulk_outcar == "None")
        missing_d = (not args.dielectric) or (args.dielectric == "None")

        if missing_v or missing_o or missing_d:
             p.error("Standard Defect Mode requires --bulk-vasprun, --bulk-outcar, and --dielectric.\n"
                     "To skip these, use --skip-corrections, --bulk-calculation, or --use-outcar-parsing.")

    root = Path(args.root).resolve()
    out = Path(args.output).resolve() if args.output else (root / "auto_traj_output")
    out.mkdir(parents=True, exist_ok=True)

    dielectric = parse_dielectric(args.dielectric)

    workflow = VaspWorkflow(
        output_root=str(out),
        bulk_vasprun_path=args.bulk_vasprun,
        bulk_outcar_path=args.bulk_outcar,
        dielectric=dielectric,
        snb=args.snb,
        n_steps=args.n_steps,
        bulk_mode=args.bulk_calculation,
        max_force_threshold=args.max_force,
        skip_corrections=args.skip_corrections,
        use_outcar_parsing=args.use_outcar_parsing
    )

    if not args.merge_only:
        folders = find_calc_folders(str(root), look_for_outcar=args.use_outcar_parsing)
        folders = pick_chunk(folders, args.chunk_index, args.nchunks)

        if not folders:
            print(f"No calculation folders found in chunk {args.chunk_index}.")
        else:
            if args.workers <= 1:
                for folder in tqdm(folders, desc="Parsing folders"):
                    try: 
                        workflow.process_calculation(folder)
                    except Exception as e: 
                        print(f"[WARN] Failed: {folder} -> {e}")
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            process_folder_wrapper, 
                            folder, 
                            args.bulk_vasprun, 
                            args.bulk_outcar, 
                            dielectric, 
                            str(out), 
                            args.snb,
                            args.n_steps,
                            args.bulk_calculation,
                            args.max_force,
                            args.skip_corrections,
                            args.use_outcar_parsing
                        ): folder 
                        for folder in folders
                    }
                    
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="Parsing folders (mp)"):
                        folder = futs[fut]
                        try: 
                            fut.result()
                        except Exception as e: 
                            print(f"[WARN] Failed: {folder} -> {e}")

    # Merging Logic
    if args.merge_only or args.nchunks <= 1 or args.chunk_index == (args.nchunks - 1):
        print("\nMerging results into combined outputs...")
        indiv_dir = out / "individual"
        
        if indiv_dir.exists():
            all_trajs = sorted(list(indiv_dir.glob("*.traj")))
            full_pairs = []

            for t_path in all_trajs:
                c_path = t_path.with_suffix(".csv")
                
                if args.bulk_calculation:
                    full_pairs.append((t_path, None))
                else:
                    if c_path.exists():
                        full_pairs.append((t_path, c_path))
                    else:
                        full_pairs.append((t_path, None))

            if full_pairs:
                combined_traj, combined_csv = workflow.combine_outputs(full_pairs)
                print(f"✅ Wrote combined trajectory: {combined_traj}")
                if combined_csv:
                    print(f"✅ Wrote combined csv:        {combined_csv}")
                else:
                    print(f"ℹ️  No CSV created (Bulk Mode or CSV missing).")
            else:
                print("⚠ No valid pairs found in individual/ to combine.")
        else:
             print("⚠ 'individual' directory not found. Nothing to merge.")

if __name__ == "__main__":
    main()