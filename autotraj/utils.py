# utils.py
from pymatgen.io.vasp.outputs import Vasprun
from ase.io import read, write
import os
import re
import warnings
from copy import deepcopy
import numpy as np
from monty.serialization import loadfn, dumpfn
from collections import defaultdict, Counter
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

# Keep your existing imports
try:
    from doped.utils.parsing import get_defect_type_site_idxs_and_unrelaxed_structure
except ImportError:
    pass # Handle cases where doped isn't installed if necessary

def identify_hybrid_functional(incar):
    """
    Identify the hybrid functional used in the VASP calculation.
    """
    functional = "GGA"
    if incar.get("LHFCALC", True):
        # identify if AEXX and HFSCREEN are present
        aexx = incar.get('AEXX')
        if aexx is not None:
            if aexx == 0.25:
                if 'HFSCREEN' in incar:
                    functional = "HSE06"
                else:
                    functional = "PBE0"
            else:
                if 'HFSCREEN' in incar:
                    functional = f"HSE_{aexx}"
                else:
                    functional = f"PBE0_{aexx}"
        else:
            if 'HFSCREEN' in incar:
                functional = "HSE06"
            else:
                functional = "PBE0"
    return functional

def calc_type(incar, kpts):
    calc_type = "vasp-std"
    if [1, 1, 1] in kpts:
        calc_type = "vasp-gam"

    elif incar.get("LSORBIT"):
        calc_type = "vasp-ncl"

    elif incar.get("NKREDX", 1) != 1:
        calc_type = calc_type + "-nkred"

    return calc_type
    
def match_formula_dopant(reduced_formula_bulk, reduced_formula_defect):
    """
    Identify dopants by comparing bulk and defect formulas.
    """
    try:
        # Extract elements only (ignore numbers)
        bulk_elements = set(re.findall(r'[A-Z][a-z]?', reduced_formula_bulk))
        defect_elements = set(re.findall(r'[A-Z][a-z]?', reduced_formula_defect))

        # Dopants are elements in defect but not in bulk
        dopants = sorted(list(defect_elements - bulk_elements))
        if not dopants:
            return reduced_formula_bulk, "intrinsic", "None"
        else:
            dopant_str = ",".join(dopants)
            return reduced_formula_bulk, "extrinsic", dopant_str

    except Exception as e:
        print(f"[Formula Match Error] {e}")
        return reduced_formula_bulk, "Unknown", "Unknown"

def _parse_path_for_defect(vasprun_filepath):
    """
    Helper to extract defect site, type, and charge from a file path.
    Robustly handles file sync artifacts (e.g. -LAPTOP-XXXX suffixes).
    """
    defect_site = None
    defect_type = "Unknown"
    charge_state = None
    
    parts = vasprun_filepath.split(os.sep)
    for folder in parts:
        folder_lower = folder.lower()

        # NEW (1): skip file-like components so we never parse from vasprun/outcar filenames
        if folder_lower.startswith(("vasprun", "outcar")) or any(ext in folder_lower for ext in [".xml", ".gz", ".bz2", ".xz"]):
            continue

        # NEW (2): skip k-point metadata folders like "01_KPOINTS_222"
        # (These are not defects, but match the Name_<int> pattern.)
        if "kpoints" in folder_lower or folder_lower.startswith("kpoints"):
            continue

        # (Optional) If you want to be stricter with other metadata folders, uncomment:
        # if re.match(r"^\d+_", folder_lower):  # e.g. 01_KPOINTS_222, 03_SOMETHING_10
        #     continue

        # 1. Skip Distortion/Rattled folders to prevent false matching of numbers
        # e.g. "Bond_Distortion_40.0%" should not be parsed as charge 40
        if any(x in folder for x in ["Distortion", "Rattled", "Unperturbed", "Dimer", "Trimer"]):
            continue

        # 2. Clean up High_Energy tags
        folder_clean = folder.replace("_High_Energy", "").replace("_High_E", "")
        
        # 3. Regex Magic 
        # Matches: "Name_Charge" OR "Name_Charge-Suffix" OR "Name_Charge Suffix"
        match = re.match(r"^(.*)_([+-]?\d+)(?:[- ].*)?$", folder_clean)
        
        if match:
            defect_site = match.group(1)
            try:
                charge_state = int(match.group(2))
            except ValueError:
                continue

    if defect_site is not None:
        ds_str = str(defect_site).lower()
        if ds_str == "none" or (isinstance(defect_site, float) and np.isnan(defect_site)):
            defect_type = "Unknown"
        elif ds_str.startswith("v_") or ds_str.startswith("vac_"):
            defect_type = "Vacancy"
        elif any(x in ds_str for x in ["_i_", "int_", "inter_"]):
            defect_type = "Interstitial"
        else:
            defect_type = "Substitution"
            
    return defect_site, defect_type, charge_state

def extract_defect_info(vasprun_filepath, snb=True, parsed_charge_state=None, bulk_vasprun=None):
    try:
        # 1. Common Path Parsing (Site, Type, Charge)
        defect_site, defect_type, charge_state = _parse_path_for_defect(vasprun_filepath)
        
        # 2. Handle Distortion logic based on snb flag
        if snb:
            distortion = "Unknown"
            parts = vasprun_filepath.split(os.sep)
            for folder in parts:
                if any(k in folder for k in ["Rattled_from_", "Distortion_", "Bond_Distortion_"]):
                    distortion = folder
                elif folder in ["Rattled", "Unperturbed", "Dimer", "Trimer"]:
                    distortion = folder
        else:
            # For snb=False, we keep distortion Unknown as requested
            distortion = "Not distorted"
            
            # 3. Fallback: If path parsing failed to find charge, use the structure-derived one
            # (Note: This is only used if snb=False usually, or if path was totally unparseable)
            if charge_state is None:
                charge_state = parsed_charge_state if parsed_charge_state is not None else "Unknown"

        return defect_site, defect_type, charge_state, distortion

    except Exception as e:
        print(f"[Defect Info Error] Failed on path '{vasprun_filepath}': {e}")
        return "Error", "Error", "Error", "Error"

def read_vasprun_data(vasprun_object, vasprun_filepath, bulk_vasprun_object, bulk_vasprun_filepath, snb=True):
    print("Extracting parameters from vasprun...")
    
    # Use robust access for structures (handles crashy runs if object is partial)
    if vasprun_object.ionic_steps:
        initial_defect_poscar = vasprun_object.ionic_steps[0]['structure'].copy()
        final_defect_poscar = vasprun_object.ionic_steps[-1]['structure'].copy()
    else:
        # Fallback for completely empty ionic steps (unlikely if filtered upstream)
        initial_defect_poscar = vasprun_object.final_structure
        final_defect_poscar = vasprun_object.final_structure

    # Handle Bulk (allow None for bulk_vasprun_object)
    if bulk_vasprun_object and bulk_vasprun_object.ionic_steps:
        final_bulk_poscar = bulk_vasprun_object.ionic_steps[-1]['structure'].copy()
        bulk_formula = final_bulk_poscar.reduced_formula
        bulk_filename = bulk_vasprun_object.filename
    else:
        final_bulk_poscar = None
        bulk_formula = "Unknown"
        bulk_filename = "None"

    number_steps = len(vasprun_object.ionic_steps)
    reduced_formula_defect = final_defect_poscar.reduced_formula
    incar = vasprun_object.parameters
    kpoints = vasprun_object.kpoints.kpts
    potcar_symbols = vasprun_object.potcar_symbols
    converged = vasprun_object.converged
    converged_electronic = vasprun_object.converged_electronic
    run_type = vasprun_object.run_type
    hybrid_functional = identify_hybrid_functional(incar)
    calculation_type = calc_type(incar, kpoints)

    # Extract info using the updated robust function
    defect_site, defect_type, parsed_charge_state, distortion = extract_defect_info(
        vasprun_filepath=vasprun_filepath, snb=snb
    )

    # Match formula
    if final_bulk_poscar:
        formula, dopant_type, dopant = match_formula_dopant(bulk_formula, reduced_formula_defect)
    else:
        formula, dopant_type, dopant = reduced_formula_defect, "Unknown", "Unknown"

    parsed_dict = {
        "kpoints": kpoints,
        "potcar_symbols": potcar_symbols,
        "converged": converged,
        "converged_electronic": converged_electronic,
        "run_type": run_type,
        "hybrid_functional": hybrid_functional,
        "calculation_type": calculation_type,
        'ENCUT': incar.get("ENMAX"),
        'NELECT': incar.get("NELECT"),
        'AEXX': incar.get("AEXX"),
        'HFSCREEN': incar.get("HFSCREEN"),
        'NUPDOWN': incar.get("NUPDOWN"),
        'GGA': incar.get("GGA"),
        'ISMEAR': incar.get("ISMEAR"),
        'ISPIN': incar.get("ISPIN"),
        'ISIF': incar.get("ISIF"),
        'EDIFF': incar.get("EDIFF"),
        'EDIFFG': incar.get("EDIFFG"),
        'LREAL': incar.get("LREAL"),
        'NELM': incar.get("NELM"),
        'number_total_steps': number_steps,
        'number_of_atoms_defect': final_defect_poscar.num_sites,
        'reduced_formula_defect': reduced_formula_defect,
        'defect_site': defect_site,
        'defect_type': defect_type,
        'distortion': distortion,
        'total_charge': parsed_charge_state,
        'formula': formula,
        'dopant_type': dopant_type,
        'dopant': dopant,
        'vasprun_filepath': vasprun_filepath,
        'bulk_vasprun_filepath': bulk_filename
    }

    vasprun_data = {**incar, **parsed_dict}
    print("Finished reading vasprun data.")
    return vasprun_data

def path_has_high_energy(path: str) -> bool:
    """
    True if any folder component in the path is exactly 'High_Energy' or contains it.
    """
    parts = path.split(os.sep)
    return any("High_Energy" in p for p in parts)