"""
Isolates whether Parts 25-38's 4.59 A result on 5T35 depended on
insufficient FFT zero-padding (a circular-correlation wraparound
artifact) rather than genuine discrimination. Runs the identical oracle-
band electrostatics search at the ORIGINAL small padding (pad =
reach_dist + 15, matching Part 25 exactly) and at several larger paddings,
everything else held fixed, and reports the best pose and RMSD for each.

If RMSD changes materially as padding increases, the small-grid result
was contaminated by wraparound and every full-search result in Parts
25-38 needs re-examination. If RMSD stays close to 4.59 A as padding
grows, the small grid was already sufficient and the discrepancy in
run_oracle_leak_test.py lies elsewhere (most likely: a real difference
between the oracle SHELL used there and here, or a bug specific to that
script), not in the validated pipeline.

Run locally: PYTHONPATH=. python3 run_padding_check.py
"""
import time
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import (
    build_receptor_potential_grid, build_charge_grid,
    build_receptor_occupancy_grid, build_ligand_shape_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

PDB_ID = "5T35"


def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)


entry = BENCHMARK_SET[PDB_ID]
s = load_structure(f"pdb_raw/{PDB_ID}.pdb", structure_id=PDB_ID)
het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0] == entry.ligand_code]
chain = s.chains[het[0]]
r = extract_ligand_mol(chain)
split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)

target_chain = s.chains[entry.target_chains[0]]
ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]

t_coords = target_chain.all_coords
t_resn = target_chain.all_residue_names
t_atomn = target_chain.all_atom_names
t_radii = np.full(len(t_coords), 1.7)
lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
lig_atomn = sum([c.all_atom_names for c in ligase_chains], [])
lig_radii = np.full(len(lig_coords), 1.7)
ligase_ca_native = np.concatenate([c.ca_coords() for c in ligase_chains])
t_charges = assign_formal_charges(t_resn, t_atomn)
lig_charges = assign_formal_charges(lig_resn, lig_atomn)

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
spacing = 1.5

rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)

PADDINGS = [
    ("original (Part 25)", reach_dist + 15.0),
    ("2x", 2 * (reach_dist + 15.0)),
    ("large", 60.0),
]

for label, pad in PADDINGS:
    lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
    hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
    n_voxels = int(np.prod(grid_shape))
    print(f"\n=== padding={pad:.1f} A ({label}), grid={grid_shape}, {n_voxels:,} voxels ===")

    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

    best_score, best_R, best_tau = -np.inf, None, None
    t0 = time.time()
    for R in rotations:
        rotated = lig_centered @ R.T
        embedded = rotated + fixed_anchor
        charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
        occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
        C_elec = -fft_correlate_3d(pot_r, charge_l)
        valid = reach_mask & build_overlap_mask(occ_r, occ_l)
        if not valid.any():
            continue
        vals = np.where(valid, C_elec, -np.inf)
        idx = np.unravel_index(np.argmax(vals), vals.shape)
        if vals[idx] > best_score:
            best_score, best_R, best_tau = float(vals[idx]), R, tau[idx].copy()

    elapsed = time.time() - t0
    posed = (ligase_ca_native - mobile_anchor) @ best_R.T + fixed_anchor + best_tau
    rmsd = raw_rmsd(posed, ligase_ca_native)
    print(f"  {elapsed/60:.1f} min | best_score={best_score:.1f} | RMSD={rmsd:.2f} A")

print("\n(Part 25's original reported result for this exact search: RMSD=4.59 A)")
