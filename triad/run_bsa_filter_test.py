"""
Tests whether requiring a minimum real buried surface area (BSA), as a
second hard filter alongside the clash filter, eliminates the false-
positive high shape scores found at wrong rotations (manifest Part 23):
a small local patch of good overlap can score well on raw shape
correlation while the rest of the ligase body floats in open space,
contributing a real but tiny BSA compared to the true native interface.

Run locally (has pdb_raw/): PYTHONPATH=. python3 run_bsa_filter_test.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import build_receptor_shape_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask
from triad.scoring.sasa import buried_surface_area
from triad.geometry.transforms import sample_rotations
from triad.benchmark.manifest import BENCHMARK_SET

entry = BENCHMARK_SET['5T35']
s = load_structure('pdb_raw/5T35.pdb', structure_id='5T35')
het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0]==entry.ligand_code]
chain = s.chains[het[0]]
r = extract_ligand_mol(chain)
split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)
target_chain = s.chains[entry.target_chains[0]]
ligase_chains = [s.chains[c] for c in entry.ligase_chains[:3] if c in s.chains]

t_coords = target_chain.all_coords
t_elem = target_chain.all_elements
lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
lig_elem = sum([c.all_elements for c in ligase_chains], [])
t_radii = np.full(len(t_coords), 1.7)
lig_radii = np.full(len(lig_coords), 1.7)

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
tau_native = mobile_anchor - fixed_anchor
spacing, pad = 1.5, reach_dist + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

native_bsa = buried_surface_area(t_coords, t_elem, lig_coords, lig_elem, n_points=50)
print(f'Native true BSA: {native_bsa:.0f} A^2')

rotations = sample_rotations(n_axes=10, n_angles_per_axis=6)
for ri in [5, 15, 25, 35]:
    R = rotations[ri]
    rotated = lig_centered @ R.T
    embedded = rotated + fixed_anchor
    shape_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    C_shape = fft_correlate_3d(shape_r, shape_l)
    C_masked = np.where(reach_mask, C_shape, -np.inf)
    best_idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
    best_wrong_tau = tau[best_idx]
    posed_wrong = rotated + fixed_anchor + best_wrong_tau
    wrong_bsa = buried_surface_area(t_coords, t_elem, posed_wrong, lig_elem, n_points=50)
    print(f'Rotation {ri} (wrong): best shape score={C_masked[best_idx]:.0f}, BSA at that pose={wrong_bsa:.0f} A^2')
