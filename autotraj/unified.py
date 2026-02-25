# unified.py
from __future__ import annotations

import os
import glob
import csv
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import Trajectory, read
from ase.calculators.singlepoint import SinglePointCalculator

from .utils import read_vasprun_data, path_has_high_energy, extract_defect_info
from .parallelised_charge_correction import EFNVCorrectionTrajectory

# Conversion factor: 1 eV/Ang^3 = 160.21766 GPa = 1602.1766 kBar
KBAR_TO_EV_A3 = 1.0 / 1602.1766208

def _safe_slug(s: str, maxlen: int = 160) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)
    return keep[:maxlen].rstrip("_")


class VaspWorkflow:
    def __init__(
        self,
        output_root: str,
        bulk_vasprun_path: Optional[str] = None,
        bulk_outcar_path: Optional[str] = None,
        dielectric: Any = None,
        snb: bool = True,
        gzip_ok: bool = True,
        n_steps: Optional[int] = 10,
        bulk_mode: bool = False,
        max_force_threshold: float = 50.0,
        skip_corrections: bool = False,
        use_outcar_parsing: bool = False,
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.gzip_ok = bool(gzip_ok)
        self.n_steps = n_steps
        self.bulk_mode = bool(bulk_mode)
        self.max_force_threshold = float(max_force_threshold)
        self.skip_corrections = bool(skip_corrections)
        self.use_outcar_parsing = bool(use_outcar_parsing) 
        self.adaptor = AseAtomsAdaptor()

        # If in bulk mode, we ignore defect-specific inputs
        if self.bulk_mode:
            self.bulk_vasprun = None
            self.bulk_outcar = None
            self.dielectric = None
            self.snb = False
        else:
            self.snb = bool(snb)
            self.dielectric = dielectric
            self.bulk_vasprun_path = str(bulk_vasprun_path) if bulk_vasprun_path and bulk_vasprun_path != "None" else None
            self.bulk_outcar_path = str(bulk_outcar_path) if bulk_outcar_path and bulk_outcar_path != "None" else None
            
            self.bulk_vasprun = None
            self.bulk_outcar = None
            
            if self.bulk_vasprun_path and os.path.exists(self.bulk_vasprun_path):
                 try:
                    self.bulk_vasprun = Vasprun(self.bulk_vasprun_path, parse_potcar_file=False, parse_dos=False, parse_eigen=False, exception_on_bad_xml=False)
                 except Exception: pass
            
            if self.bulk_outcar_path and os.path.exists(self.bulk_outcar_path):
                 try:
                    self.bulk_outcar = Outcar(self.bulk_outcar_path)
                 except Exception: pass

    @staticmethod
    def _find_vasp_pairs(folder: str) -> List[Tuple[Optional[str], Optional[str]]]:
        pairs = []
        all_files = os.listdir(folder)
        vasprun_files = sorted([f for f in all_files if f.startswith("vasprun")])

        for v_name in vasprun_files:
            v_path = os.path.join(folder, v_name)
            o_name = v_name.replace("vasprun", "OUTCAR")
            if v_name == "vasprun.xml" or v_name == "vasprun.xml.gz":
                 candidates = ["OUTCAR", "OUTCAR.gz"]
            else:
                 candidates = [o_name]
                 if ".xml" in o_name:
                     candidates.append(o_name.replace(".xml", ""))

            o_path = None
            for c in candidates:
                cand_path = os.path.join(folder, c)
                if os.path.exists(cand_path):
                    o_path = cand_path
                    break
            pairs.append((v_path, o_path))

        if not pairs:
            outcar_files = sorted([f for f in all_files if f.startswith("OUTCAR")])
            for o_name in outcar_files:
                pairs.append((None, os.path.join(folder, o_name)))
                
        return pairs

    @staticmethod
    def hash_dict(d: Dict[str, Any]) -> str:
        serialized = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def process_calculation(self, folder: str) -> Tuple[Path, Optional[Path]]:
        if path_has_high_energy(folder):
            raise RuntimeError("Skipped High_Energy path")

        file_pairs = self._find_vasp_pairs(folder)
        if not file_pairs:
            raise FileNotFoundError(f"No VASP files found in {folder}")

        slug = _safe_slug(folder.replace(os.sep, "__"))
        short_hash = self.hash_dict({"folder": folder, "first_file": file_pairs[0][0] or file_pairs[0][1]})[:12]
        stem = f"{slug}__{short_hash}"

        indiv_dir = self.output_root / "individual"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        traj_out = indiv_dir / f"{stem}.traj"
        csv_out = indiv_dir / f"{stem}.csv"

        final_traj = Trajectory(str(traj_out), mode="w")
        final_csv_rows = []

        for vasprun_path, outcar_path in file_pairs:
            frames_to_write = [] 
            meta_data = {}
            
            parsing_method = "vasprun"
            defect_vasprun = None

            if self.use_outcar_parsing or not vasprun_path:
                parsing_method = "outcar"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        defect_vasprun = Vasprun(
                            filename=vasprun_path,
                            parse_potcar_file=False,
                            parse_dos=False,
                            parse_eigen=False,
                            exception_on_bad_xml=False
                        )
                        if not defect_vasprun.ionic_steps:
                            raise RuntimeError("Empty steps")
                    except Exception as e:
                        print(f"[WARN] Vasprun malformed in {folder} ({os.path.basename(vasprun_path)}). Switching to OUTCAR fallback.")
                        parsing_method = "outcar"

            # ------------------------------------------------------------------
            # PATH A: OUTCAR PARSING (ASE)
            # ------------------------------------------------------------------
            if parsing_method == "outcar":
                if not outcar_path: continue 

                try:
                    traj_atoms = read(outcar_path, index=":", format="vasp-out")
                except Exception: continue 

                valid_indices = []
                for i, atoms in enumerate(traj_atoms):
                    e = atoms.get_potential_energy()
                    if e >= 0: continue
                    f = atoms.get_forces()
                    if f is not None:
                        max_f = np.max(np.linalg.norm(f, axis=1))
                        if max_f > self.max_force_threshold: continue
                    valid_indices.append(i)

                final_indices = []
                if self.n_steps and self.n_steps > 0 and len(valid_indices) > self.n_steps:
                    subsample_idx = np.linspace(0, len(valid_indices) - 1, self.n_steps, dtype=int)
                    final_indices = [valid_indices[j] for j in subsample_idx]
                else:
                    final_indices = valid_indices

                for i in final_indices:
                    atoms = traj_atoms[i]
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    # ASE OUTCAR parsing usually gives stress in correct units/shape
                    try: s = atoms.get_stress() 
                    except: s = np.zeros(6)
                    
                    atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f, stress=s)
                    
                    atoms.info["source_file"] = os.path.basename(outcar_path)
                    atoms.info["raw_energy_eV"] = e
                    atoms.info["corrected_energy_eV"] = e 
                    atoms.info["correction_term_eV"] = 0.0
                    atoms.info["total_charge"] = 0.0
                    atoms.info["charge_state"] = 0.0
                    frames_to_write.append(atoms)
                
                meta_data = {
                    "charge_state": 0.0, "total_charge": 0.0, "run_type": "outcar_fallback",
                    "vasprun_path": "None", "outcar_path": outcar_path
                }

            # ------------------------------------------------------------------
            # PATH B: VASPRUN PARSING
            # ------------------------------------------------------------------
            else:
                all_ionic_steps = defect_vasprun.ionic_steps
                n_total = len(all_ionic_steps)
                
                if not defect_vasprun.converged:
                    candidate_indices = list(range(n_total - 1))
                else:
                    candidate_indices = list(range(n_total))

                nelm = defect_vasprun.parameters.get("NELM", 60)
                valid_indices = []
                
                for i in candidate_indices:
                    step = all_ionic_steps[i]
                    if len(step.get("electronic_steps", [])) >= nelm: continue
                    
                    e_0 = step.get("e_0_energy")
                    if e_0 is None and "electronic_steps" in step and step["electronic_steps"]:
                        e_0 = step["electronic_steps"][-1].get("e_0_energy")
                    if e_0 is not None and e_0 >= 0: continue
                    
                    try:
                        forces = np.array(step.get("forces", []), dtype=float)
                        if forces.size > 0:
                            max_f = np.max(np.linalg.norm(forces, axis=1))
                            if max_f > self.max_force_threshold: continue
                    except Exception: pass 
                    
                    valid_indices.append(i)

                final_indices = []
                if self.n_steps and self.n_steps > 0 and len(valid_indices) > self.n_steps:
                    subsample_idx = np.linspace(0, len(valid_indices) - 1, self.n_steps, dtype=int)
                    final_indices = [valid_indices[j] for j in subsample_idx]
                else:
                    final_indices = valid_indices

                try:
                    if not self.bulk_mode:
                        m = read_vasprun_data(
                            vasprun_object=defect_vasprun,
                            vasprun_filepath=vasprun_path,
                            bulk_vasprun_object=self.bulk_vasprun, 
                            bulk_vasprun_filepath=self.bulk_vasprun_path, 
                            snb=self.snb,
                        )
                        meta_data = m
                    else:
                        meta_data = {"charge_state": 0.0, "total_charge": 0.0}
                except Exception:
                    meta_data = {"charge_state": 0.0, "total_charge": 0.0}

                try: c_val = float(meta_data.get("total_charge", 0.0))
                except: c_val = 0.0
                meta_data["total_charge"] = c_val
                meta_data["charge_state"] = c_val
                meta_data["vasprun_path"] = vasprun_path
                meta_data["outcar_path"] = outcar_path

                corrections = {}
                if not self.skip_corrections:
                    if final_indices and abs(c_val) > 1e-5 and outcar_path and self.bulk_vasprun and self.bulk_outcar:
                        try:
                            corr_engine = EFNVCorrectionTrajectory(
                                bulk_vasprun=self.bulk_vasprun,
                                bulk_outcar=self.bulk_outcar,
                                defect_vasprun=defect_vasprun,
                                defect_outcar=outcar_path,          
                                dielectric_tensor=self.dielectric,
                                charge=c_val,
                                already_parsed=True,
                            )
                            corrections = corr_engine.compute_all_corrections()
                        except Exception: pass

                for i in final_indices:
                    step = defect_vasprun.ionic_steps[i]
                    struct = step["structure"]
                    atoms = self.adaptor.get_atoms(struct)

                    raw_e = None
                    if "e_0_energy" in step: raw_e = float(step["e_0_energy"]) 
                    elif "e_fr_energy" in step: raw_e = float(step["e_fr_energy"])
                    else:
                        try: raw_e = float(step["electronic_steps"][-1]["e_0_energy"])
                        except: raw_e = float(getattr(defect_vasprun, "final_energy", np.nan))

                    forces = np.array(step.get("forces", []), dtype=float)
                    if forces.shape[0] != len(atoms): forces = None
                    
                    # --- FIXED STRESS HANDLING ---
                    stress = step.get("stress", None)
                    if stress is not None:
                        stress = np.array(stress, dtype=float)
                        # 1. Check shape. If 3x3, keep it. If flat 9, reshape.
                        if stress.shape == (9,):
                            stress = stress.reshape(3, 3)
                        
                        # 2. Convert Units: kBar -> eV/A^3
                        stress = stress * KBAR_TO_EV_A3
                    else:
                        # Fallback to zero (3x3 matrix safe for ASE)
                        stress = np.zeros((3, 3))

                    if raw_e is not None and forces is not None:
                        atoms.calc = SinglePointCalculator(atoms, energy=raw_e, forces=forces, stress=stress)

                    corr = float(corrections.get(i, 0.0))
                    corrected_e = raw_e + corr

                    atoms.info["source_file"] = os.path.basename(vasprun_path)
                    atoms.info["ionic_step"] = i
                    atoms.info["raw_energy_eV"] = raw_e
                    atoms.info["correction_term_eV"] = corr
                    atoms.info["corrected_energy_eV"] = corrected_e
                    atoms.info["total_charge"] = c_val
                    atoms.info["charge_state"] = c_val

                    frames_to_write.append(atoms)

            # WRITE AGGREGATED
            for atoms in frames_to_write:
                atoms.info["source_folder"] = folder
                final_traj.write(atoms)

                if not self.bulk_mode:
                    f = atoms.get_forces()
                    fmean = float(np.mean(np.linalg.norm(f, axis=1))) if f is not None else np.nan
                    try: s = atoms.get_stress()
                    except: s = np.zeros(6)
                    
                    row = {
                        **meta_data,
                        "source_folder": folder,
                        "source_file": atoms.info["source_file"],
                        "ionic_step": atoms.info.get("ionic_step"),
                        "raw_energy_eV": atoms.info["raw_energy_eV"],
                        "correction_eV": atoms.info["correction_term_eV"],
                        "corrected_energy_eV": atoms.info["corrected_energy_eV"],
                        "fmean_avg": fmean,
                        "stress_voigt": s.tolist() if s is not None else None,
                        "cell_lengths": list(map(float, atoms.cell.cellpar()[:3])),
                    }
                    final_csv_rows.append(row)

        if final_csv_rows:
            all_keys = set().union(*(r.keys() for r in final_csv_rows))
            fieldnames = sorted(list(all_keys))
            with open(csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(final_csv_rows)

        return traj_out, (None if self.bulk_mode else csv_out)

    def combine_outputs(self, traj_csv_pairs: List[Tuple[Path, Optional[Path]]]) -> Tuple[Path, Optional[Path]]:
        combined_dir = self.output_root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_traj = combined_dir / "combined.traj"
        
        out_traj = Trajectory(str(combined_traj), mode="w")
        
        count_files = 0
        count_frames = 0
        detailed_lines = []
        if self.bulk_mode:
             header = f"{'Idx':<5} | {'Energy (eV)':<12} | {'Max Force':<10} | {'Press(kBar)':<11} | {'Step':<5} | {'Source File'}"
             detailed_lines.append(header)
             detailed_lines.append("-" * len(header))

        for traj_path, _ in traj_csv_pairs:
            if not traj_path.exists(): continue
            try:
                t = Trajectory(str(traj_path), mode="r")
                n_local = 0
                for atoms in t:
                    out_traj.write(atoms)
                    
                    if self.bulk_mode:
                        try: e = atoms.get_potential_energy()
                        except: e = 0.0
                        
                        try: 
                            f = atoms.get_forces()
                            fmax = np.max(np.linalg.norm(f, axis=1)) if f is not None else 0.0
                        except: fmax = 0.0
                        
                        try:
                            s = atoms.get_stress()
                            if s is not None:
                                # Convert BACK to kBar for display only (approx 1602 * eV/A^3)
                                press_kbar = -np.mean(s[:3]) * 1602.18
                                s_str = f"{press_kbar:.1f}"
                            else: s_str = "None"
                        except Exception as e: 
                            s_str = "Err" # Still Err implies read failure, but less likely now
                            
                        src_file = atoms.info.get("source_file", "??")
                        step = str(atoms.info.get("ionic_step", "?"))
                        g_idx = count_frames + n_local
                        
                        line = f"{g_idx:<5} | {e:<12.4f} | {fmax:<10.4f} | {s_str:<11} | {step:<5} | {src_file}"
                        detailed_lines.append(line)
                    
                    n_local += 1
                
                if n_local > 0:
                    count_files += 1
                    count_frames += n_local
                    
            except Exception as e:
                print(f"[WARN] Skipping corrupted traj file: {traj_path.name} ({e})")
                continue

        combined_csv = combined_dir / "combined.csv"
        dfs = []
        total_csv_rows = 0
        
        if not self.bulk_mode:
            for _, csv_path in traj_csv_pairs:
                if csv_path and csv_path.exists() and csv_path.stat().st_size > 0:
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty: dfs.append(df)
                    except Exception: pass
            
            if dfs:
                full_df = pd.concat(dfs, ignore_index=True)
                total_csv_rows = len(full_df)
                full_df.to_csv(combined_csv, index=False)
            else:
                pd.DataFrame().to_csv(combined_csv, index=False)
        
        theoretical_max = count_files * (self.n_steps if self.n_steps else 0)
        yield_pct = (count_frames / theoretical_max * 100) if theoretical_max > 0 else 0.0

        summary_file = combined_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("=== DATASET MERGE SUMMARY ===\n")
            f.write(f"Total Folders Processed:         {count_files}\n")
            f.write(f"Total Frames Extracted:          {count_frames}\n")
            if not self.bulk_mode:
                f.write(f"Total CSV Rows:                  {total_csv_rows}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Requested Frames per File:       {self.n_steps}\n")
            
            if self.bulk_mode and detailed_lines:
                f.write("\n\n=== DETAILED FRAME DATA ===\n")
                for line in detailed_lines:
                    f.write(line + "\n")
        
        print(f"📝 Summary saved to: {summary_file}")
        return combined_traj, (None if self.bulk_mode else combined_csv)