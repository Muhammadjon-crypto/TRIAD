"""
Efficient weight-calibration sweep: runs ONE full 3600-rotation search per
structure, but at each rotation stores the top-5 candidates by
electrostatics AND top-5 by normalized contact potential (not just the
single combined winner). After the search, sweeps many weight values
against this stored candidate pool, computing combined = elec + w*contact
and finding the best pose for EACH weight -- without re-running the
expensive FFT correlations per weight.

APPROXIMATION, stated honestly: the true best combined-score candidate for
some weight could in principle be outside the top-5-per-channel pool this
stores. This trades exactness for making a real sweep computationally
feasible (one search per structure instead of one per weight per
structure). Top-5 per channel per rotation, across all rotations, gives a
pool of up to 36,000 candidates per structure to sweep against.

Run locally: PYTHONPATH=. python3 run_weight_calibration.py
(Two full searches -- expect ~30-40 min total.)
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
from triad.correlation.channels import build_receptor_occupancy_grid, build_ligand_shape_grid, build_receptor_potential_grid, build_charge_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
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

DILATION = 4.0
TOP_K_PER_CHANNEL = 5
WEIGHTS_TO_TEST = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

def run_calibration_search(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f'pdb_raw/{pdb_id}.pdb', structure_id=pdb_id)
    het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0]==entry.ligand_code]
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
    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
    occ_r_dilated = ndimage.binary_dilation(occ_r > 0, iterations=int(np.ceil(DILATION/spacing)))
    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

    print(f"Running calibration search on {pdb_id} ({entry.ligase}, {entry.target})...")
    rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
    pool_elec, pool_contact, pool_rotation, pool_tau = [], [], [], []
    t0 = time.time()
    for i, R in enumerate(rotations):
        rotated = lig_centered @ R.T
        embedded = rotated + fixed_anchor
        charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
        C_elec = -fft_correlate_3d(pot_r, charge_l)

        l_type_grids = build_residue_type_grids(embedded, lig_resn, lig_radii, grid_shape, lo, spacing, dilation_radius=DILATION)
        C_contact = contact_potential_correlation(t_type_grids, l_type_grids, potential)

        occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
        overlap_mask = build_overlap_mask(occ_r, occ_l)
        valid_mask = reach_mask & overlap_mask
        if not valid_mask.any():
            continue

        dilated_overlap = occ_r_dilated & (occ_l > 0)
        total_dilated = np.sum(dilated_overlap)
        contact_norm = C_contact / total_dilated if total_dilated > 0 else np.zeros_like(C_contact)

        valid_idx = np.argwhere(valid_mask)
        elec_vals = C_elec[valid_mask]
        contact_vals = contact_norm[valid_mask]

        # top-K by electrostatics
        top_elec_idx = np.argsort(-elec_vals)[:TOP_K_PER_CHANNEL]
        # top-K by contact
        top_contact_idx = np.argsort(-contact_vals)[:TOP_K_PER_CHANNEL]
        combined_idx = set(top_elec_idx.tolist()) | set(top_contact_idx.tolist())

        for k in combined_idx:
            pool_elec.append(elec_vals[k])
            pool_contact.append(contact_vals[k])
            pool_rotation.append(R)
            pool_tau.append(tau[tuple(valid_idx[k])])

        if (i + 1) % 1200 == 0:
            print(f"  {i+1}/{len(rotations)} ({(time.time()-t0)/60:.1f} min), pool size so far: {len(pool_elec)}")

    elapsed = time.time() - t0
    print(f"Search done in {elapsed/60:.1f} min. Total candidate pool: {len(pool_elec)}\n")

    pool_elec = np.array(pool_elec)
    pool_contact = np.array(pool_contact)

    results = {}
    for w in WEIGHTS_TO_TEST:
        combined = pool_elec + w * pool_contact
        best_i = np.argmax(combined)
        best_R = pool_rotation[best_i]
        best_tau_val = pool_tau[best_i]
        posed_ca = (ligase_ca_native - mobile_anchor) @ best_R.T + fixed_anchor + best_tau_val
        rmsd = raw_rmsd(posed_ca, ligase_ca_native)
        results[w] = rmsd
        print(f"  weight={w:6.1f}: RMSD={rmsd:.2f} A")

    return results

results_5t35 = run_calibration_search('5T35')
print()
results_6hax = run_calibration_search('6HAX')

print("\n=== CALIBRATION SUMMARY ===")
print(f"{'weight':8} {'5T35 RMSD':12} {'6HAX RMSD':12}")
for w in WEIGHTS_TO_TEST:
    print(f"{w:8.1f} {results_5t35[w]:12.2f} {results_6hax[w]:12.2f}")
