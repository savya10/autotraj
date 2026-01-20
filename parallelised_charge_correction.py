import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from doped.analysis import _convert_dielectric_to_tensor
from pydefect.analyzer.calc_results import CalcResults
from pydefect.analyzer.defect_structure_comparator import DefectStructureComparator
from pydefect.cli.vasp.make_efnv_correction import calc_max_sphere_radius
from pydefect.corrections.efnv_correction import ExtendedFnvCorrection, PotentialSite
from pydefect.corrections.ewald import Ewald
from pydefect.defaults import defaults
from doped.utils.parsing import get_defect_type_site_idxs_and_unrelaxed_structure
from tqdm import tqdm
import warnings

def strip_supercell(bulk_supercell, defect_supercell):
    """
    Safely strip oxidation states.
    """
    # 1. Fix Lattice mismatch
    if not np.allclose(bulk_supercell.lattice.matrix, defect_supercell.lattice.matrix, atol=1e-2):
        warnings.warn(
            f"Bulk and defect supercells have different lattices!\n"
            f"Scaling bulk to match defect for eFNV calculation."
        )
    # scale bulk lattice to match defect lattice
    bulk_supercell.lattice = defect_supercell.lattice

    # 2. Unconditionally remove oxidation states
    # This is safe even if they don't exist (it just does nothing).
    bulk_supercell.remove_oxidation_states()
    defect_supercell.remove_oxidation_states()

    return bulk_supercell, defect_supercell 

class EFNVCorrectionTrajectory:
    def __init__(self, bulk_vasprun, bulk_outcar,
                 defect_vasprun, defect_outcar,
                 dielectric_tensor, charge, already_parsed=True, defect_region_radius=None):
        """Pre-load structures, potentials and build reusable objects"""
        self.charge = charge
        if already_parsed:
            self.bulk_vasprun = bulk_vasprun
            self.defect_vasprun = defect_vasprun
            self.bulk_outcar = bulk_outcar
            self.defect_outcar = Outcar(defect_outcar)
        else:
            self.bulk_vasprun = Vasprun(bulk_vasprun)
            self.defect_vasprun = Vasprun(defect_vasprun)
            self.bulk_outcar = Outcar(bulk_outcar)
            self.defect_outcar = Outcar(defect_outcar)

        # bulk = final structure, defect = trajectory
        self.bulk_supercell_raw = self.bulk_vasprun.ionic_steps[-1]["structure"]
        self.defect_supercells_raw = [s["structure"] for s in self.defect_vasprun.ionic_steps]

        # Use helper to strip the bulk and the LAST defect frame (for consistency check)
        self.bulk_supercell, _ =  strip_supercell(
            self.bulk_supercell_raw.copy(),
            self.defect_supercells_raw[-1].copy()
            )
        
        # Correctly strip list of structures (avoiding in-place None return)
        self.defect_supercells = []
        for s in self.defect_supercells_raw:
            s_clean = s.copy()
            s_clean.remove_oxidation_states()
            self.defect_supercells.append(s_clean)

        # all potentials
        self.bulk_potentials = np.array(self.bulk_outcar.read_avg_core_poten()[-1])
        self.defect_potentials = self.defect_outcar.read_avg_core_poten() # list of lists

        # dielectric + Ewald
        self.dielectric_tensor = _convert_dielectric_to_tensor(dielectric_tensor)
        self.lattice = self.defect_supercells[0].lattice 
        self.ewald = Ewald(self.lattice.matrix, self.dielectric_tensor,
                           accuracy=defaults.ewald_accuracy)
        self.unit_conversion = 180.95128169876497

        self.defect_region_radius = defect_region_radius

    def _get_defect_coords(self, defect_structure):
        defect_type, bulk_site_idx, defect_site_idx, _ = get_defect_type_site_idxs_and_unrelaxed_structure(
            self.bulk_supercell_raw, defect_structure)
        if defect_type == "vacancy":
            return self.bulk_supercell_raw[bulk_site_idx].frac_coords
        else:
            return defect_structure[defect_site_idx].frac_coords

    def _single_step_correction(self, defect_structure, defect_pots, defect_coords, defect_region_radius):
        """Lightweight doped_make_efnv_correction using cached Ewald"""
        # Note: We pass self.bulk_supercell which is already stripped
        structure_analyzer = DefectStructureComparator(defect_structure, self.bulk_supercell)

        sites, rel_coords = [], []
        for d, p in structure_analyzer.atom_mapping.items():
            specie = str(defect_structure[d].specie)
            frac_coords = defect_structure[d].frac_coords
            dist, _ = self.lattice.get_distance_and_image(defect_coords, frac_coords)
            pot = defect_pots[d] - self.bulk_potentials[p]
            sites.append(PotentialSite(specie, dist, pot, None))
            rel_coords.append([x - y for x, y in zip(frac_coords, defect_coords)])

        pc_corr = -self.ewald.lattice_energy * self.charge**2 if self.charge else 0.0
        for site, rel_coord in zip(sites, rel_coords):
            if site.distance > defect_region_radius and self.charge:
                site.pc_potential = (self.ewald.atomic_site_potential(rel_coord)
                                     * self.charge * self.unit_conversion)

        return ExtendedFnvCorrection(
            charge=self.charge,
            point_charge_correction=pc_corr * self.unit_conversion,
            defect_region_radius=defect_region_radius,
            sites=sites,
            defect_coords=tuple(defect_coords)
        )

    def compute_all_corrections(self):
        corrections = {}
        for i, defect_structure in enumerate(self.defect_supercells):
            # No tqdm here to keep logs clean in multiprocess
            defect_coords = self._get_defect_coords(self.defect_supercells_raw[i])
            defect_pots = np.array(self.defect_potentials[i])

            radius = self.defect_region_radius
            if radius is None:
                radius = calc_max_sphere_radius(defect_structure.lattice.matrix)
            corr = self._single_step_correction(defect_structure, defect_pots, defect_coords, defect_region_radius=radius)
            corrections[i] = corr.correction_energy
        return corrections