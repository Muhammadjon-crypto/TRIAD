"""
THE DECISIVE TEST for Parts 33-35's finding: does a full, dense,
autonomous rotation-inclusive search, scored by the normalized,
dilation-fixed contact potential, actually recover a near-native pose --
not just win against a small set of pre-selected known-bad rotations?

Run locally: PYTHONPATH=. python3 run_contact_full_search.py
(This is a real full search -- expect several minutes, similar to the
electrostatics-only searches in Parts 25-26.)
"""
import numpy as np, time
from scipy import ndimage
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.potentials.contact_potential import (
    extract_interface_contacts, classify_surface_residues, derive_contact_potential,
)
from triad.correlation.contact_channel import build_residue_type_grids, contact_potential_correlation
from triad.correlation.channels import build_receptor_occupancy_grid, build_ligand_shape_grid
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
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
print(f"Derived from {n_used} structures.\n")

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
ligase_ca_native = np.concatenate([c.ca_coords() for c in ligase_chains])

reach_dist = reach.straight_line_distance_angstrom
mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
lig_centered = lig_coords - mobile_anchor
spacing, pad = 1.5, reach_dist + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

DILATION = 4.0
t_type_grids = build_residue_type_grids(t_coords, t_resn, t_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
occ_r_dilated = ndimage.binary_dilation(occ_r > 0, iterations=int(np.ceil(DILATION/spacing)))
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

print("Running full 3600-rotation search, scored by NORMALIZED CONTACT POTENTIAL...")
rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
best_score, best_rotation, best_tau = -np.inf, None, None
t0 = time.time()
for i, R in enumerate(rotations):
    rotated = lig_centered @ R.T
    embedded = rotated + fixed_anchor
    l_type_grids = build_residue_type_grids(embedded, lig_resn, lig_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
    C_contact = contact_potential_correlation(t_type_grids, l_type_grids, potential)

    occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    overlap_mask = build_overlap_mask(occ_r, occ_l)
    valid_mask = reach_mask & overlap_mask
    if valid_mask.any():
        # normalize per-voxel using the DILATED overlap count (matches Part 34's metric)
        dilated_overlap = occ_r_dilated & (occ_l > 0)
        total_dilated = np.sum(dilated_overlap)
        if total_dilated > 0:
            C_normalized = np.where(valid_mask, C_contact / total_dilated, -np.inf)
            idx = np.unravel_index(np.argmax(C_normalized), C_normalized.shape)
            score = C_normalized[idx]
            if score > best_score:
                best_score, best_rotation, best_tau = score, R, tau[idx]
    if (i + 1) % 600 == 0:
        print(f"  {i+1}/{len(rotations)} done ({(time.time()-t0)/60:.1f} min)")

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min. Best normalized score: {best_score:.4f}")
posed_ca = (ligase_ca_native - mobile_anchor) @ best_rotation.T + fixed_anchor + best_tau
rmsd = raw_rmsd(posed_ca, ligase_ca_native)
print(f"RMSD of best-found pose vs native: {rmsd:.2f} A")
print("(compare: shape+electrostatics gave 69.45 A; electrostatics-only gave 4.59 A on this structure)")
