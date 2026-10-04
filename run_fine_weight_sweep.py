"""
Efficient finer weight sweep, correcting the redundant-computation
inefficiency in run_exact_weight_calibration.py (which redid the entire
exact-electrostatics search once per weight, since only the combination
step actually depends on weight). This version scores every reach+clash
valid candidate ONCE per rotation with both exact electrostatics and the
contact-potential channel, storing (elec, contact, rotation_index, tau)
for every valid candidate -- not a shortlist, the same design flaw already
caught and avoided in the previous script -- then sweeps many weight
values against that single stored pool.

Memory check: Part 42's exact full search on 5T35 scored ~2.1 million
candidates total; storing 4 small values per candidate (2 floats, 1 int,
3 floats) is on the order of tens of MB, not a concern.

Tests a bracketing set of weights around the two promising regions found
in Part 43: 10-50 for 5T35 (best so far: 27.00A at 20), 50-250 for 6HAX
(best so far: 18.31A at 100, still improving at the tested edge).

Run locally: PYTHONPATH=. python3 run_fine_weight_sweep.py
(One full exact search per structure, ~20-25 min each based on Part 42-43 timing,
plus a cheap post-hoc sweep over ~10 weights per structure -- total
should be close to the cost of a SINGLE full search per structure, not
one per weight.)
"""
import time
import numpy as np
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
from triad.scoring.electrostatics import assign_formal_charges, coulomb_energy
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
        s = load_structure(f"pdb_raw/{pdb_id}.pdb", structure_id=pdb_id)
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

DILATION = 4.0

WEIGHTS_5T35 = [8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0]
WEIGHTS_6HAX = [50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0]


def run_pool_search(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f"pdb_raw/{pdb_id}.pdb", structure_id=pdb_id)
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
    spacing, pad = 1.5, reach_dist + 15.0
    lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
    hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    t_type_grids = build_residue_type_grids(t_coords, t_resn, t_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
    occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
    occ_r_dilated = ndimage.binary_dilation(occ_r > 0, iterations=int(np.ceil(DILATION / spacing)))
    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)
    rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)

    print(f"Running ONE exact search on {pdb_id} ({entry.ligase}, {entry.target}), storing full candidate pool...")
    pool_elec = []
    pool_contact = []
    pool_rot_idx = []
    pool_tau = []
    t0 = time.time()
    for ri, R in enumerate(rotations):
        rotated = lig_centered @ R.T
        embedded = rotated + fixed_anchor

        l_type_grids = build_residue_type_grids(embedded, lig_resn, lig_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
        C_contact = contact_potential_correlation(t_type_grids, l_type_grids, potential)

        occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
        valid_mask = reach_mask & build_overlap_mask(occ_r, occ_l)
        if not valid_mask.any():
            continue

        dilated_overlap = occ_r_dilated & (occ_l > 0)
        total_dilated = np.sum(dilated_overlap)
        contact_norm = C_contact / total_dilated if total_dilated > 0 else np.zeros_like(C_contact)

        valid_idx = np.argwhere(valid_mask)
        for c in valid_idx:
            candidate_tau = tau[tuple(c)]
            posed = embedded + candidate_tau
            exact_elec = -coulomb_energy(t_coords, t_charges, posed, lig_charges)
            pool_elec.append(exact_elec)
            pool_contact.append(contact_norm[tuple(c)])
            pool_rot_idx.append(ri)
            pool_tau.append(candidate_tau)

        if (ri + 1) % 600 == 0:
            print(f"  {ri + 1}/{len(rotations)} ({(time.time() - t0) / 60:.1f} min), pool size: {len(pool_elec):,}")

    elapsed = time.time() - t0
    print(f"Search done in {elapsed / 60:.1f} min. Total candidate pool: {len(pool_elec):,}\n")

    pool_elec = np.array(pool_elec)
    pool_contact = np.array(pool_contact)

    weights = WEIGHTS_5T35 if pdb_id == "5T35" else WEIGHTS_6HAX
    results = {}
    for w in weights:
        combined = pool_elec + w * pool_contact
        best_i = np.argmax(combined)
        best_R = rotations[pool_rot_idx[best_i]]
        best_tau_val = pool_tau[best_i]
        posed_ca = (ligase_ca_native - mobile_anchor) @ best_R.T + fixed_anchor + best_tau_val
        rmsd = raw_rmsd(posed_ca, ligase_ca_native)
        results[w] = rmsd
        print(f"  weight={w:6.1f}: RMSD={rmsd:.2f} A")

    return results


results_5t35 = run_pool_search("5T35")
print()
results_6hax = run_pool_search("6HAX")

print("\n=== FINE SWEEP SUMMARY ===")
print("5T35 (bracketing weight=20, prior best 27.00 A):")
for w in WEIGHTS_5T35:
    print(f"  {w:6.1f}: {results_5t35[w]:.2f} A")
print("\n6HAX (bracketing/extending past weight=100, prior best 18.31 A):")
for w in WEIGHTS_6HAX:
    print(f"  {w:6.1f}: {results_6hax[w]:.2f} A")
