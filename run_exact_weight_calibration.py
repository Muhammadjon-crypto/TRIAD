"""
Redoes Part 38's weight-calibration sweep (contact potential +
electrostatics) with EXACT pairwise Coulomb scoring in place of the
broken FFT approximation (Part 42's retraction). Part 38's "no weight
rescues 6HAX" conclusion was built on corrupted electrostatics scores and
must be re-checked before being trusted either way.

DESIGN NOTE, correcting a flaw caught before this was run: an earlier
draft of this script shortlisted candidates by contact score only, then
computed exact electrostatics for just that shortlist. At weight=0 that
would not correctly reproduce the true electrostatics-only baseline
(Part 42's 61.10 A on 5T35), since the best pure-electrostatics candidate
need not appear in a contact-score-based shortlist -- silently
reintroducing a new approximation error while fixing the old one. This
version instead scores every reach+clash-valid candidate at every
rotation with BOTH the exact Coulomb sum and the contact-potential
channel, and computes the true combined argmax directly for each tested
weight. Slower (a separate full pass per weight) but not approximated.
To keep total runtime reasonable, only 5 weights are tested instead of 9.

Run locally: PYTHONPATH=. python3 run_exact_weight_calibration.py
(5 weights x 2 structures, each a full ~7 minute exact search -- expect
roughly 60-75 minutes total. Runs 5T35 completely before starting 6HAX,
so partial results are visible if interrupted.)
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
WEIGHTS_TO_TEST = [0.0, 1.0, 5.0, 20.0, 100.0]


def run_one_weight(pdb_id, weight, cached):
    """Full exact search at a single weight. `cached` holds structure-level
    setup so repeated calls for the same pdb_id don't redo it."""
    (t_coords, t_charges, lig_coords, lig_resn, lig_charges, lig_radii,
     ligase_ca_native, mobile_anchor, fixed_anchor, lig_centered,
     grid_shape, lo, spacing, reach_dist, t_type_grids, occ_r,
     occ_r_dilated, tau, reach_mask, rotations) = cached

    best_score, best_R, best_tau = -np.inf, None, None
    t0 = time.time()
    for R in rotations:
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
            combined = exact_elec + weight * contact_norm[tuple(c)]
            if combined > best_score:
                best_score, best_R, best_tau = combined, R, candidate_tau

    elapsed = time.time() - t0
    posed_ca = (ligase_ca_native - mobile_anchor) @ best_R.T + fixed_anchor + best_tau
    rmsd = raw_rmsd(posed_ca, ligase_ca_native)
    print(f"  weight={weight:6.1f}: RMSD={rmsd:.2f} A  ({elapsed/60:.1f} min)")
    return rmsd


def setup_structure(pdb_id):
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

    return (t_coords, t_charges, lig_coords, lig_resn, lig_charges, lig_radii,
            ligase_ca_native, mobile_anchor, fixed_anchor, lig_centered,
            grid_shape, lo, spacing, reach_dist, t_type_grids, occ_r,
            occ_r_dilated, tau, reach_mask, rotations)


all_results = {}
for pdb_id in ["5T35", "6HAX"]:
    entry = BENCHMARK_SET[pdb_id]
    print(f"=== {pdb_id} ({entry.ligase}, {entry.target}) ===")
    cached = setup_structure(pdb_id)
    all_results[pdb_id] = {}
    for w in WEIGHTS_TO_TEST:
        all_results[pdb_id][w] = run_one_weight(pdb_id, w, cached)
    print()

print("=== EXACT-ELECTROSTATICS CALIBRATION SUMMARY ===")
print(f"{'weight':8} {'5T35 RMSD':12} {'6HAX RMSD':12}")
for w in WEIGHTS_TO_TEST:
    print(f"{w:8.1f} {all_results['5T35'][w]:12.2f} {all_results['6HAX'][w]:12.2f}")
print("\n(Part 38's original FFT-based weight=0.0 baselines: 5T35=4.59A [now known")
print(" retracted/artifact], 6HAX=77.49A. Part 42's exact electrostatics-only")
print(" baseline for 5T35 was 61.10A. This run's weight=0.0 row should match that.)")
