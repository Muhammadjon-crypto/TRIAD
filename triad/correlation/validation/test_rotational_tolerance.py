"""
Quantifies the rotational tolerance window around the true native
orientation for 5T35: at the TRUE native translation, how much rotational
error can be tolerated before a real, hard clash appears?

This is the key measurement from docs/TRIAD_v1_MANIFEST.md Part 9: it
reconciles Part 7 ("translation coverage isn't the bottleneck, FFT handles
it exhaustively") with Part 8's clash-veto finding by identifying WHERE the
real coverage gap actually is -- not translation (FFT-accelerated,
exhaustive), but ROTATION (still brute-force enumerated, and our current
180-rotation grid has ~35-60 degree gaps between samples, far coarser than
the ~10-20 degree tolerance window measured here).
"""
import os

import numpy as np
import pytest

from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.scoring.clash import clash_score
from triad.sampling.local_refinement import rotation_from_rotvec
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"
pytestmark = pytest.mark.skipif(not os.path.isdir(PDB_DIR), reason=f"{PDB_DIR}/ not present")


def _setup_5t35():
    entry = BENCHMARK_SET["5T35"]
    s = load_structure(f"{PDB_DIR}/5T35.pdb", structure_id="5T35")
    chain = s.chains[[c for c in s.hetero_chain_ids()
                       if s.chains[c].all_residue_names[0] == entry.ligand_code][0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)
    target_chain = s.chains[entry.target_chains[0]]
    n_copy = len(entry.ligase_chains) // 2
    ligase_chains = [s.chains[c] for c in entry.ligase_chains[:n_copy]]
    t_coords, t_elem, t_resn, t_resid, t_atomn = extract_representative_atoms(target_chain)
    lig_parts = [extract_representative_atoms(c) for c in ligase_chains]
    lig_coords = np.concatenate([p[0] for p in lig_parts])
    lig_elem = sum([p[1] for p in lig_parts], [])
    return dict(
        t_coords=t_coords, t_elem=t_elem, lig_coords=lig_coords, lig_elem=lig_elem,
        mobile_anchor=reach.ligase_warhead_centroid,
        fixed_anchor=chain.all_coords[reach.farthest_atom_idx],
    )


def test_clash_score_increases_monotonically_with_rotational_error():
    d = _setup_5t35()
    lig_centered = d["lig_coords"] - d["mobile_anchor"]
    tau_native = d["mobile_anchor"] - d["fixed_anchor"]
    axis = np.array([1.0, 0.0, 0.0])

    angles_deg = [0, 5, 10, 20, 30, 45]
    scores = []
    for angle_deg in angles_deg:
        R = rotation_from_rotvec(axis * np.radians(angle_deg))
        rotated = lig_centered @ R.T
        posed = rotated + d["fixed_anchor"] + tau_native
        scores.append(clash_score(d["t_coords"], d["t_elem"], posed, d["lig_elem"]))

    for i in range(len(scores) - 1):
        assert scores[i + 1] >= scores[i] - 1e-6, (
            f"clash score should not decrease with more rotational error: "
            f"{list(zip(angles_deg, scores))}"
        )


def test_rotational_tolerance_window_is_narrow():
    """THE KEY QUANTITATIVE FINDING (manifest Part 9): the true native
    pose tolerates only ~10-20 degrees of rotational error before a real
    clash appears. Measured, not assumed -- and substantially finer than
    our current 180-rotation grid's ~35-60 degree gaps between samples.
    """
    d = _setup_5t35()
    lig_centered = d["lig_coords"] - d["mobile_anchor"]
    tau_native = d["mobile_anchor"] - d["fixed_anchor"]
    axis = np.array([1.0, 0.0, 0.0])

    def clash_at(angle_deg):
        R = rotation_from_rotvec(axis * np.radians(angle_deg))
        rotated = lig_centered @ R.T
        posed = rotated + d["fixed_anchor"] + tau_native
        return clash_score(d["t_coords"], d["t_elem"], posed, d["lig_elem"])

    assert clash_at(10) <= 5.0, "expected 10 degrees to still be within tolerance"
    assert clash_at(20) > 5.0, "expected 20 degrees to exceed clash tolerance"


if __name__ == "__main__":
    test_clash_score_increases_monotonically_with_rotational_error()
    test_rotational_tolerance_window_is_narrow()
    print("Rotational tolerance window findings confirmed and locked in.")
