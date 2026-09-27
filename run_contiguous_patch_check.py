"""
Tests whether the contiguous-patch rescoring correctly separates 5T35's
true native pose from the known spurious wrong-rotation pose (rotation
15, raw shape score 1105 -- one of the clearest false-positive examples
established earlier in this investigation, manifest Part 22-23).

If contiguous-patch rescoring pulls the wrong pose's score down
substantially relative to native's, that's real, direct evidence this
approach targets the actual diagnosed failure mode.

Run locally: PYTHONPATH=. python3 run_contiguous_patch_check.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import build_receptor_shape_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.contiguous_patch import contiguous_patch_score, largest_contiguous_overlap
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
lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
t_radii = np.full(len(t_coords), 1.7)
lig_radii = np.full(len(lig_coords), 1.7)

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
spacing, pad = 1.5, reach_dist + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)

def check_pose(embedded_lig_coords, label):
    shape_l = build_ligand_shape_grid(embedded_lig_coords, lig_radii, grid_shape, lo, spacing)
    C_shape = fft_correlate_3d(shape_r, shape_l)
    raw_score = C_shape[0, 0, 0]
    largest_patch = largest_contiguous_overlap(shape_r, shape_l)
    total_overlap = np.sum((shape_r > 0) & (shape_l > 0))
    rescored = contiguous_patch_score(shape_r, shape_l, raw_score)
    frac = largest_patch / total_overlap if total_overlap > 0 else 0.0
    print(f'{label}: raw_score={raw_score:.1f}, total_overlap={total_overlap}, '
          f'largest_patch={largest_patch}, contiguous_fraction={frac:.2f}, rescored={rescored:.1f}')

# native pose
check_pose(lig_coords, 'NATIVE')

# the known spurious rotation-15 pose, at its own best-scoring translation
rotations = sample_rotations(n_axes=10, n_angles_per_axis=6)
R = rotations[15]
rotated = lig_centered @ R.T
embedded = rotated + fixed_anchor
# find its best translation within the reach sphere (same as prior tests)
from triad.correlation.search import build_reach_mask
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)
shape_l_wrong = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
C_shape_wrong = fft_correlate_3d(shape_r, shape_l_wrong)
C_masked = np.where(reach_mask, C_shape_wrong, -np.inf)
best_idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
best_wrong_tau = tau[best_idx]
posed_wrong = rotated + fixed_anchor + best_wrong_tau
check_pose(posed_wrong, 'WRONG rotation 15 (best pose)')
