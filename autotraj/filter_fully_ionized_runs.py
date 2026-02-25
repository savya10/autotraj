"""
Filter DOPED/autotraj defect relaxation runs to only "fully ionized" charge states
(no excess carriers), using common oxidation states from pymatgen.

"Fully ionized" convention used here:
- Compute nominal defect charge q_nominal from oxidation-state bookkeeping:
    Vacancy v_X:        q_nominal = - ox(X)
    Interstitial X_i:   q_nominal = + ox(X)
    Substitution A_B:   q_nominal = ox(A) - ox(B)
- Keep the run if q_run == q_nominal

Notes:
- Uses autotraj's extract_defect_info() for consistent parsing of defect_site and charge_state.
- Defect labels may contain extra trailing annotations (e.g. Cd_i_D3d_O2.20); we parse only the
  leading token(s) that define the defect identity.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

from pymatgen.core.periodic_table import Element

from utils import extract_defect_info

VASP_XML_GLOBS = ("vasprun.xml", "vasprun.xml.gz")

# Exclude these workflow subfolders entirely:
EXCLUDED_PATH_PARTS = {"vasp_std", "vasp_nkred_std"}


@dataclass(frozen=True)
class DefectSpec:
    kind: str  # "vacancy" | "interstitial" | "substitution" | "unknown"
    outgoing: Optional[str]  # removed species (vacancy/substitution)
    incoming: Optional[str]  # added species (interstitial/substitution)


def _find_vasprun_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    for g in VASP_XML_GLOBS:
        paths.extend(root.rglob(g))

    # De-dup by resolved file path (you may still have xml + xml.gz in same folder; that's intentional)
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _canonicalize_element_symbol(sym: str) -> str:
    """
    Turn 'cu' -> 'Cu', 'SB' -> 'Sb', ' o ' -> 'O'.
    Raises ValueError for empty strings.
    """
    s = str(sym).strip()
    if not s:
        raise ValueError("Empty element symbol")
    return s[0].upper() + (s[1:].lower() if len(s) > 1 else "")


# def _parse_defect_site_label(defect_site: str) -> DefectSpec:
#     """
#     Permissive parsing of defect_site across multiple naming conventions.

#     Supported (examples):
#       - v_O, vac_O, vacancy_O
#       - vac_1_Zn
#       - Cd_i, Cd_i_D3d_O2.20
#       - Int_12_Li, Int_Cu_4, Int_S_1_-2 (charge is not part of defect_site here, but extra tokens are ok)
#       - Al_Zn
#       - sub_2_Li_on_B, sub_Li_on_B
#     """
#     s = str(defect_site).strip()
#     tokens = re.split(r"[_\-]+", s)
#     tokens = [t for t in tokens if t]
#     if len(tokens) < 2:
#         return DefectSpec(kind="unknown", outgoing=None, incoming=None)

#     low = [t.lower() for t in tokens]

#     def is_el(tok: str) -> bool:
#         return bool(re.fullmatch(r"[A-Za-z]{1,2}", tok))

#     def canon_el(tok: str) -> str:
#         return _canonicalize_element_symbol(tok)

#     # -------------------
#     # Vacancy patterns
#     # -------------------
#     # v_O / vac_O / vacancy_O
#     if low[0] in {"v", "vac", "vacancy"} and is_el(tokens[1]):
#         return DefectSpec(kind="vacancy", outgoing=canon_el(tokens[1]), incoming=None)

#     # vac_1_Zn / vac_12_O
#     if low[0] in {"v", "vac", "vacancy"} and len(tokens) >= 3:
#         # expect tokens[1] is an index, tokens[2] is element
#         if re.fullmatch(r"\d+", tokens[1]) and is_el(tokens[2]):
#             return DefectSpec(kind="vacancy", outgoing=canon_el(tokens[2]), incoming=None)

#     # -------------------
#     # Interstitial patterns
#     # -------------------
#     # Cd_i / Cd_int / Cd_inter  (and allows trailing tokens like symmetry info)
#     if low[1] in {"i", "int", "inter"} and is_el(tokens[0]):
#         return DefectSpec(kind="interstitial", outgoing=None, incoming=canon_el(tokens[0]))

#     # Int_12_Li / Int_S_1 / Int_Cu_4_2  (ignore numeric tokens)
#     if low[0] in {"int", "interstitial"}:
#         # find first element-like token after the prefix, skipping indices
#         for t in tokens[1:]:
#             if is_el(t):
#                 return DefectSpec(kind="interstitial", outgoing=None, incoming=canon_el(t))
#         return DefectSpec(kind="unknown", outgoing=None, incoming=None)

#     # -------------------
#     # Substitution patterns
#     # -------------------
#     # Al_Zn (simple)
#     if is_el(tokens[0]) and is_el(tokens[1]):
#         return DefectSpec(kind="substitution", outgoing=canon_el(tokens[1]), incoming=canon_el(tokens[0]))

#     # sub_2_Li_on_B  OR sub_Li_on_B
#     if low[0] in {"sub", "subst", "substitution"} and "on" in low:
#         # locate "on", take nearest element before and after
#         on_i = low.index("on")

#         # incoming element: scan left of "on"
#         incoming = None
#         for j in range(on_i - 1, 0, -1):
#             if is_el(tokens[j]):
#                 incoming = canon_el(tokens[j])
#                 break

#         # outgoing element: scan right of "on"
#         outgoing = None
#         for j in range(on_i + 1, len(tokens)):
#             if is_el(tokens[j]):
#                 outgoing = canon_el(tokens[j])
#                 break

#         if incoming and outgoing:
#             return DefectSpec(kind="substitution", outgoing=outgoing, incoming=incoming)

#     return DefectSpec(kind="unknown", outgoing=None, incoming=None)

def _parse_defect_site_label(defect_site: str) -> DefectSpec:
    """
    Permissive parsing of defect_site across multiple naming conventions.

    Supported (examples):
      - v_O, vac_O, vacancy_O
      - vac_1_Zn
      - Cd_i, Cd_i_D3d_O2.20
      - Int_12_Li, inter_6_Li_1, Int_Cu_4_2
      - Al_Zn
      - sub_2_Li_on_B, as_1_S_on_Zn, whateverprefix_..._A_on_B
    """
    s = str(defect_site).strip()
    tokens = re.split(r"[_\-]+", s)
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return DefectSpec(kind="unknown", outgoing=None, incoming=None)

    low = [t.lower() for t in tokens]

    def is_el(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]{1,2}", tok))

    def canon_el(tok: str) -> str:
        return _canonicalize_element_symbol(tok)

    def is_index(tok: str) -> bool:
        return bool(re.fullmatch(r"\d+", tok))

    # -------------------
    # Vacancy patterns
    # -------------------
    # v_O / vac_O / vacancy_O
    if low[0] in {"v", "vac", "vacancy"} and is_el(tokens[1]):
        return DefectSpec(kind="vacancy", outgoing=canon_el(tokens[1]), incoming=None)

    # vac_1_Zn / vac_12_O
    if low[0] in {"v", "vac", "vacancy"} and len(tokens) >= 3:
        if is_index(tokens[1]) and is_el(tokens[2]):
            return DefectSpec(kind="vacancy", outgoing=canon_el(tokens[2]), incoming=None)

    # -------------------
    # Interstitial patterns
    # -------------------
    # Cd_i / Cd_int / Cd_inter  (and allows trailing tokens like symmetry info)
    if low[1] in {"i", "int", "inter"} and is_el(tokens[0]):
        return DefectSpec(kind="interstitial", outgoing=None, incoming=canon_el(tokens[0]))

    # Int_12_Li / inter_6_Li_1 / int_S_1 / Interstitial_... (ignore numeric tokens)
    if low[0] in {"int", "inter", "interstitial"}:
        for t in tokens[1:]:
            if is_el(t):
                return DefectSpec(kind="interstitial", outgoing=None, incoming=canon_el(t))
        return DefectSpec(kind="unknown", outgoing=None, incoming=None)

    # -------------------
    # Substitution patterns
    # -------------------
    # Al_Zn (simple)
    if is_el(tokens[0]) and is_el(tokens[1]):
        return DefectSpec(kind="substitution", outgoing=canon_el(tokens[1]), incoming=canon_el(tokens[0]))

    # Generic "*_on_*" pattern (works for sub_*, as_*, etc.)
    if "on" in low:
        on_i = low.index("on")

        incoming = None
        for j in range(on_i - 1, -1, -1):
            if is_el(tokens[j]):
                incoming = canon_el(tokens[j])
                break

        outgoing = None
        for j in range(on_i + 1, len(tokens)):
            if is_el(tokens[j]):
                outgoing = canon_el(tokens[j])
                break

        if incoming and outgoing:
            return DefectSpec(kind="substitution", outgoing=outgoing, incoming=incoming)

    return DefectSpec(kind="unknown", outgoing=None, incoming=None)



def _common_oxidation_states(el: str) -> Sequence[int]:
    """
    Canonicalizes symbol and returns common oxidation states.

    If the symbol is invalid for pymatgen, returns empty tuple (treated as undeterminable).
    """
    try:
        el = _canonicalize_element_symbol(el)
        e = Element(el)
        return tuple(int(x) for x in e.common_oxidation_states)
    except Exception:
        return ()


def _choose_ox(ox_states: Sequence[int], choose: str) -> int:
    if not ox_states:
        raise ValueError("No oxidation states available")
    if choose == "first":
        return int(ox_states[0])
    if choose == "min_abs":
        return int(min(ox_states, key=lambda x: abs(x)))
    if choose == "max_abs":
        return int(max(ox_states, key=lambda x: abs(x)))
    raise ValueError(f"Unknown choose mode: {choose}")


def _expected_nominal_charge(
    defect: DefectSpec,
    *,
    choose_ox: str = "first",
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Returns (q_nominal, details) or None if cannot determine.
    """
    details: Dict[str, Any] = {"kind": defect.kind, "incoming": defect.incoming, "outgoing": defect.outgoing}

    if defect.kind == "vacancy":
        if not defect.outgoing:
            return None
        oxs = _common_oxidation_states(defect.outgoing)
        if not oxs:
            return None
        ox = _choose_ox(oxs, choose_ox)
        details.update({"ox_outgoing": ox, "ox_outgoing_candidates": list(oxs)})
        return -ox, details

    if defect.kind == "interstitial":
        if not defect.incoming:
            return None
        oxs = _common_oxidation_states(defect.incoming)
        if not oxs:
            return None
        ox = _choose_ox(oxs, choose_ox)
        details.update({"ox_incoming": ox, "ox_incoming_candidates": list(oxs)})
        return +ox, details

    if defect.kind == "substitution":
        if not defect.incoming or not defect.outgoing:
            return None
        ox_in_list = _common_oxidation_states(defect.incoming)
        ox_out_list = _common_oxidation_states(defect.outgoing)
        if not ox_in_list or not ox_out_list:
            return None
        ox_in = _choose_ox(ox_in_list, choose_ox)
        ox_out = _choose_ox(ox_out_list, choose_ox)
        details.update(
            {
                "ox_incoming": ox_in,
                "ox_outgoing": ox_out,
                "ox_incoming_candidates": list(ox_in_list),
                "ox_outgoing_candidates": list(ox_out_list),
            }
        )
        return (ox_in - ox_out), details

    return None


def run_is_fully_ionized(
    vasprun_path: Path,
    *,
    choose_ox: str = "first",
) -> Tuple[bool, Dict[str, Any]]:
    # ---- NEW: exclude vasp_std / vasp_nkred_std runs by path ----
    # Path.parts is safe for both absolute and relative paths.
    if any(part in EXCLUDED_PATH_PARTS for part in vasprun_path.parts):
        info: Dict[str, Any] = {
            "vasprun": str(vasprun_path),
            "run_dir": str(vasprun_path.parent),
            "reason": "excluded_workflow_subfolder",
        }
        return False, info

    defect_site, defect_type, charge_state, distortion = extract_defect_info(str(vasprun_path), snb=True)

    info: Dict[str, Any] = {
        "vasprun": str(vasprun_path),
        "run_dir": str(vasprun_path.parent),
        "defect_site": defect_site,
        "defect_type": defect_type,
        "charge_state": charge_state,
        "distortion": distortion,
    }

    if defect_site in (None, "Error", "Unknown"):
        info["reason"] = "could_not_parse_defect_site"
        return False, info

    if not isinstance(charge_state, int):
        info["reason"] = "could_not_parse_charge_state"
        return False, info

    defect = _parse_defect_site_label(str(defect_site))
    info["defect_kind_inferred"] = defect.kind
    info["incoming"] = defect.incoming
    info["outgoing"] = defect.outgoing

    q_nominal_and_details = _expected_nominal_charge(defect, choose_ox=choose_ox)
    if q_nominal_and_details is None:
        info["reason"] = "no_expected_nominal_charge"
        return False, info

    q_nominal, details = q_nominal_and_details
    info["q_nominal"] = int(q_nominal)
    info.update({f"nominal_{k}": v for k, v in details.items()})

    expected_q_run = int(q_nominal)
    info["expected_charge_state"] = expected_q_run

    keep = (int(charge_state) == expected_q_run)
    info["reason"] = "match" if keep else "charge_state_mismatch"
    return keep, info


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "vasprun",
        "run_dir",
        "defect_site",
        "defect_type",
        "defect_kind_inferred",
        "incoming",
        "outgoing",
        "charge_state",
        "q_nominal",
        "expected_charge_state",
        "reason",
        "distortion",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    remaining = sorted(k for k in all_keys if k not in preferred)
    fieldnames = [k for k in preferred if k in all_keys] + remaining

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root directory containing many defect run folders (searched recursively for vasprun.xml[.gz]).")
    ap.add_argument(
        "--choose-ox",
        choices=["first", "min_abs", "max_abs"],
        default="first",
        help="If multiple common oxidation states exist, choose which one to use.",
    )
    ap.add_argument("--print-keep-only", action="store_true", help="Only print kept vasprun paths.")
    ap.add_argument("--write-rejects", type=Path, default=None, help="Write rejected runs + reasons to a CSV file.")
    ap.add_argument("--write-kept", type=Path, default=None, help="Write kept runs to a CSV file (optional).")
    ap.add_argument("--copy-to", type=Path, default=None, help="Copy the parent directory of each kept vasprun into this directory.")
    ap.add_argument("--symlink-to", type=Path, default=None, help="Symlink the parent directory of each kept vasprun into this directory.")
    args = ap.parse_args()

    if args.copy_to and args.symlink_to:
        raise SystemExit("Choose only one of --copy-to or --symlink-to")

    vaspruns = _find_vasprun_paths(args.root)
    if not vaspruns:
        raise SystemExit(f"No vasprun.xml or vasprun.xml.gz found under {args.root}")

    kept: List[Tuple[Path, Dict[str, Any]]] = []
    rejected: List[Tuple[Path, Dict[str, Any]]] = []

    for vp in vaspruns:
        keep, info = run_is_fully_ionized(vp, choose_ox=args.choose_ox)
        (kept if keep else rejected).append((vp, info))

    if args.print_keep_only:
        for vp, _ in kept:
            print(vp)
    else:
        print(f"Found vaspruns: {len(vaspruns)}")
        print(f"Kept (fully ionized): {len(kept)}")
        print(f"Rejected: {len(rejected)}")

        reasons: Dict[str, int] = {}
        for _, info in rejected:
            reasons[str(info.get("reason", "unknown"))] = reasons.get(str(info.get("reason", "unknown")), 0) + 1
        if reasons:
            print("\n--- Reject reasons (count) ---")
            for k in sorted(reasons, key=lambda x: (-reasons[x], x)):
                print(f"{k}: {reasons[k]}")

    if args.write_rejects:
        _write_rows_csv(args.write_rejects, [info for _, info in rejected])

    if args.write_kept:
        _write_rows_csv(args.write_kept, [info for _, info in kept])

    outdir: Optional[Path] = args.copy_to or args.symlink_to
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        for vp, info in kept:
            run_dir = vp.parent
            dest = outdir / run_dir.name
            if dest.exists():
                continue
            if args.copy_to:
                shutil.copytree(run_dir, dest)
            else:
                os.symlink(run_dir.resolve(), dest, target_is_directory=True)


if __name__ == "__main__":
    main()