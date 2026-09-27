"""
Re-tests the knowledge-based contact potential (Part 12's channel,
originally found unhelpful) through the corrected pipeline: full-atom
coordinates (Part 23's fix) and corrected chain selection (Part 16/30's
fix). Also re-derives the potential itself from full-atom contacts,
since the original derivation used the same reduced representative-atom
set now known to miss most real hydrophobic packing -- if the ORIGINAL
potential was built from incomplete contact data, Part 12's negative
result may reflect a broken potential, not a genuinely uninformative one.

Tests whether the re-derived, full-atom contact potential can distinguish
5T35's true native pose from the known spurious wrong-rotation-15 pose,
where three separate geometry-only metrics (raw overlap, BSA, contiguity)
have now all failed to do so (Parts 22, 24, 31).

Run locally: PYTHONPATH=. python3 run_contact_potential_retest.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.potentials.contact_potential import (
    extract_interface_contacts, classify_surface_residues, derive_contact_potential,
)
from triad.correlation.contact_channel import build_residue_type_grids, contact_potential_correlation
from triad.geometry.transforms import sample_rotations
from triad.benchmark.manifest import BENCHMARK_SET

def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)

print("Re-deriving contact potential from all 15 structures, FULL ATOM coordinates...")
all_contacts, all_background = [], []
n_used = 0
for pdb_id, entry in BENCHMARK_SET.items():
    try:
        s = load_structure(f'pdb_raw/{pdb_id}.pdb', structure_id=pdb_id)
        target_chain = s.chains[entry.target_chains[0]]
        ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]
        if not ligase_chains:
            continue

        t_coords = target_chain.all_coords
        t_resn = target_chain.all_residue_names
        lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
        lig_resn = sum([c.all_residue_names for c in ligase_chains], [])

        contacts = extract_interface_contacts(t_coords, t_resn, lig_coords, lig_resn)
        all_contacts.extend(contacts)
        t_surf = classify_surface_residues(t_coords)
        lig_surf = classify_surface_residues(lig_coords)
        all_background.extend([r for r, m in zip(t_resn, t_surf) if m])
        all_background.extend([r for r, m in zip(lig_resn, lig_surf) if m])
        n_used += 1
    except Exception as e:
        print(f'  {pdb_id}: skipped -- {type(e).__name__}: {e}')

print(f"Used {n_used}/15 structures, {len(all_contacts)} contacts (full atom)")
potential = derive_contact_potential(all_contacts, all_background, n_used)

# biochemistry sanity check before trusting it
print()
print("Biochemistry check:")
for pair in [('LEU','ILE'), ('LEU','LEU'), ('ASP','GLU'), ('LYS','ARG')]:
    print(f"  {pair}: {potential.get(*pair):.2f}")

# now apply to 5T35 native vs known wrong pose
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

t_type_grids = build_residue_type_grids(t_coords, t_resn, t_radii, grid_shape, lo, spacing)

def check_contact_score(embedded_lig_coords, label):
    l_type_grids = build_residue_type_grids(embedded_lig_coords, lig_resn, lig_radii, grid_shape, lo, spacing)
    C_contact = contact_potential_correlation(t_type_grids, l_type_grids, potential)
    score = C_contact[0, 0, 0]
    print(f'{label}: contact_potential_score={score:.1f}')
    return score

print()
print("5T35 native vs known wrong-rotation-15 pose, contact potential score:")
native_score = check_contact_score(lig_coords, 'NATIVE')

rotations = sample_rotations(n_axes=10, n_angles_per_axis=6)
R = rotations[15]
rotated = lig_centered @ R.T
embedded = rotated + fixed_anchor
from triad.correlation.channels import build_receptor_shape_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask
shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
shape_l_wrong = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
C_shape_wrong = fft_correlate_3d(shape_r, shape_l_wrong)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)
C_masked = np.where(reach_mask, C_shape_wrong, -np.inf)
best_idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
best_wrong_tau = tau[best_idx]
posed_wrong = rotated + fixed_anchor + best_wrong_tau
wrong_score = check_contact_score(posed_wrong, 'WRONG rotation 15 (best shape pose)')

print()
print(f"Native contact score: {native_score:.1f}, Wrong pose contact score: {wrong_score:.1f}")
print("PASS: native scores higher" if native_score > wrong_score else "FAIL: wrong pose scores higher or equal")
