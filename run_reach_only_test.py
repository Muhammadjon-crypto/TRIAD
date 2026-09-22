"""
Tests the reach constraint's discriminating power in isolation, without
shape at all: does the true native rotation, using ONLY the reach-distance
geometric constraint (no shape, no electrostatics), stand out from wrong
rotations? Manifest Part 24 hypothesis: reach-constraint specificity, not
shape reward, may be the real signal to lean on for Phase 4.

For each rotation tested (native + several wrong ones), report whether a
valid reach-satisfying translation exists at all, and separately, whether
combining reach + electrostatics (no shape) ranks native above the wrong
rotations' best electrostatics-only pose.

Run locally: PYTHONPATH=. python3 run_reach_only_test.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import build_receptor_potential_grid, build_charge_grid, build_receptor_occupancy_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
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
t_resn = target_chain.all_residue_names
t_atomn = target_chain.all_atom_names
lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
lig_elem = sum([c.all_elements for c in ligase_chains], [])
lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
lig_atomn = sum([c.all_atom_names for c in ligase_chains], [])
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

pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

def eval_rotation(R, label):
    rotated = lig_centered @ R.T
    embedded = rotated + fixed_anchor
    charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
    occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    C_elec = -fft_correlate_3d(pot_r, charge_l)
    overlap_mask = build_overlap_mask(occ_r, occ_l)
    valid_mask = reach_mask & overlap_mask
    n_valid = valid_mask.sum()
    if n_valid == 0:
        print(f'{label}: NO valid (reach+clash) candidates at all')
        return None
    best_elec = C_elec[valid_mask].max()
    print(f'{label}: {n_valid} valid candidates, best electrostatics-only score = {best_elec:.1f}')
    return best_elec

print('Electrostatics-only (no shape) scores, reach+clash filtered:')
native_score = eval_rotation(np.eye(3), 'NATIVE rotation')

rotations = sample_rotations(n_axes=10, n_angles_per_axis=6)
for ri in [5, 15, 25, 35]:
    eval_rotation(rotations[ri], f'Wrong rotation {ri}')
