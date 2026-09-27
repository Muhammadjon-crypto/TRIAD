"""
Re-tests native vs known-wrong-rotation-15 with the FIXED contact channel
(dilation_radius=4.0, proximity-based rather than raw interpenetration --
manifest Part 33). If Part 32's finding was an artifact of the
interpenetration-only bug rather than a genuine size-bias refutation,
this should change the outcome.

Run locally: PYTHONPATH=. python3 run_dilated_contact_check.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.potentials.contact_potential import (
    extract_interface_contacts, classify_surface_residues, derive_contact_potential,
)
from triad.correlation.contact_channel import build_residue_type_grids, contact_potential_correlation
from triad.correlation.channels import build_receptor_shape_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask
from triad.geometry.transforms import sample_rotations
from triad.benchmark.manifest import BENCHMARK_SET

def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)

print("Re-deriving corrected contact potential...")
all_contacts, all_background = [], []
n_used = 0
for pdb_id, entry in BENCHMARK_SET.items():
    try:
        s = load_structure(f'pdb_raw/{pdb_id}.pdb', structure_id=pdb_id)
        target_chain = s.chains[entry.target_chains[0]]
        ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]
        if not ligase_chains:
            continue
        t_coords_full = target_chain.all_coords
        t_resn_full = target_chain.all_residue_names
        lig_coords_full = np.concatenate([c.all_coords for c in ligase_chains])
        lig_resn_full = sum([c.all_residue_names for c in ligase_chains], [])
        all_contacts.extend(extract_interface_contacts(t_coords_full, t_resn_full, lig_coords_full, lig_resn_full))
        t_coords_rep, _, t_resn_rep, _, _ = extract_representative_atoms(target_chain)
        lig_rep_parts = [extract_representative_atoms(c) for c in ligase_chains]
        lig_coords_rep = np.concatenate([p[0] for p in lig_rep_parts])
        lig_resn_rep = sum([p[2] for p in lig_rep_parts], [])
        t_surf = classify_surface_residues(t_coords_rep)
        lig_surf = classify_surface_residues(lig_coords_rep)
        all_background.extend([r for r, m in zip(t_resn_rep, t_surf) if m])
        all_background.extend([r for r, m in zip(lig_resn_rep, lig_surf) if m])
        n_used += 1
    except Exception:
        pass
potential = derive_contact_potential(all_contacts, all_background, n_used)
print(f"Derived from {n_used} structures.")

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
t_resn = target_chain.all_residue_names
t_radii = np.full(len(t_coords), 1.7)
lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
lig_radii = np.full(len(lig_coords), 1.7)

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
spacing, pad = 1.5, reach_dist + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

DILATION = 4.0  # Angstroms -- roughly typical residue-residue contact range
t_type_grids = build_residue_type_grids(t_coords, t_resn, t_radii, grid_shape, lo, spacing, dilation_radius=DILATION)

def check_pose(embedded_lig_coords, label):
    l_type_grids = build_residue_type_grids(embedded_lig_coords, lig_resn, lig_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
    C_contact = contact_potential_correlation(t_type_grids, l_type_grids, potential)
    score = C_contact[0, 0, 0]
    print(f'{label}: dilated_contact_score={score:.1f}')
    return score

print()
print("5T35 native vs known wrong-rotation-15 pose, DILATED (proximity-based) contact channel:")
native_score = check_pose(lig_coords, 'NATIVE')

rotations = sample_rotations(n_axes=10, n_angles_per_axis=6)
R = rotations[15]
rotated = lig_centered @ R.T
embedded = rotated + fixed_anchor
shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
shape_l_wrong = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
C_shape_wrong = fft_correlate_3d(shape_r, shape_l_wrong)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)
C_masked = np.where(reach_mask, C_shape_wrong, -np.inf)
best_idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
best_wrong_tau = tau[best_idx]
posed_wrong = rotated + fixed_anchor + best_wrong_tau
wrong_score = check_pose(posed_wrong, 'WRONG rotation 15 (best shape pose)')

print()
print(f"Native: {native_score:.1f}, Wrong pose: {wrong_score:.1f}")
print("PASS: native scores higher" if native_score > wrong_score else "FAIL: wrong pose scores higher or equal")
