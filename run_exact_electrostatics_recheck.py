"""
THE critical re-check: does native still rank well among valid candidates
at the correct rotation (the foundational claim from Parts 8 and 16) once
electrostatics is scored with the EXACT pairwise Coulomb sum
(triad.scoring.electrostatics.coulomb_energy, validated against analytical
cases in Part 5) instead of the FFT-grid approximation now known to be
wrong (Part 40)?

Reuses the identical reach+clash candidate pool construction as every
earlier test on 5T35 at the native rotation. Only the electrostatics
SCORING of each candidate changes: FFT approximation -> exact pairwise sum.
Shape/clash grids are untouched (compactly supported, not implicated by
Part 40's diagnosis).

Run locally: PYTHONPATH=. python3 run_exact_electrostatics_recheck.py
(No full rotation search -- one rotation, exact-scoring every valid
candidate. Should take at most a few minutes.)
"""
import time
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import (
    build_receptor_shape_grid, build_ligand_shape_grid, build_receptor_occupancy_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges, coulomb_energy
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

# REPRESENTATIVE atoms, matching Parts 8/16's original foundational test
t_coords, t_elem, t_resn, t_resid, t_atomn = extract_representative_atoms(target_chain)
lig_parts = [extract_representative_atoms(c) for c in ligase_chains]
lig_coords = np.concatenate([p[0] for p in lig_parts])
lig_resn = sum([p[2] for p in lig_parts], [])
lig_atomn = sum([p[4] for p in lig_parts], [])
t_radii = np.full(len(t_coords), 1.7)
lig_radii = np.full(len(lig_coords), 1.7)
t_charges = assign_formal_charges(t_resn, t_atomn)
lig_charges = assign_formal_charges(lig_resn, lig_atomn)

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
tau_native = mobile_anchor - fixed_anchor

spacing, pad = 1.5, reach_dist + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
print(f"grid {grid_shape} (original padding, fine for shape/clash -- not implicated by Part 40)")

shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

# native rotation = identity
embedded = lig_centered + fixed_anchor
shape_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
C_shape = fft_correlate_3d(shape_r, shape_l)
overlap_mask = build_overlap_mask(occ_r, shape_l)
valid_mask = reach_mask & overlap_mask
valid_idx = np.argwhere(valid_mask)
print(f"{len(valid_idx)} valid (reach+clash) candidates at native rotation")

t0 = time.time()
exact_scores = np.empty(len(valid_idx))
for k, c in enumerate(valid_idx):
    candidate_tau = tau[tuple(c)]
    posed_lig = lig_centered + fixed_anchor + candidate_tau
    exact_scores[k] = -coulomb_energy(t_coords, t_charges, posed_lig, lig_charges)
print(f"scored {len(valid_idx)} candidates with EXACT pairwise Coulomb in {time.time()-t0:.1f}s")

native_exact = -coulomb_energy(t_coords, t_charges, lig_coords, lig_charges)
rank = int(np.sum(exact_scores > native_exact)) + 1
pct = 100.0 * rank / len(valid_idx)
print(f"\nNative exact electrostatic score: {native_exact:.1f}")
print(f"Native rank among {len(valid_idx)} valid candidates: {rank} ({pct:.1f} percentile)")
print(f"\n(Compare: Part 8/16's ORIGINAL FFT-based finding for this exact setup was")
print(f" native rank 31st of 1001, ~3.0 percentile -- the foundational claim this")
print(f" whole investigation was built on.)")
