"""
Tests whether Part 25's breakthrough (removing shape, scoring by
electrostatics+reach alone, recovers 4.59A RMSD on 5T35 vs 69.45A with
shape) generalizes across other structures. Picks a spread from both the
'tractable' and 'random' groups established in manifest Parts 14-21.

Run locally: PYTHONPATH=. python3 run_electrostatics_only_multi.py
(Takes ~2 min per structure; several structures run here.)
"""
import numpy as np, time
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import build_receptor_potential_grid, build_charge_grid, build_receptor_occupancy_grid, build_ligand_shape_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

# spread across both groups: 6BN7/6BOY (tractable, CRBN), 8FY0 (tractable, VHL/BCLxl),
# 6HAX (random, VHL/SMARCA2), 8BDS (random, VHL/BRD4-BD1)
TEST_IDS = ['6BN7', '6HAX', '8BDS']

def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)

for pdb_id in TEST_IDS:
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
    t_elem = target_chain.all_elements
    t_resn = target_chain.all_residue_names
    t_atomn = target_chain.all_atom_names
    lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
    lig_elem = sum([c.all_elements for c in ligase_chains], [])
    lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
    lig_atomn = sum([c.all_atom_names for c in ligase_chains], [])
    ligase_ca_native = np.concatenate([c.ca_coords() for c in ligase_chains])
    t_radii = np.full(len(t_coords), 1.7)
    lig_radii = np.full(len(lig_coords), 1.7)
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

    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

    rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
    print(f'--- {pdb_id} ({entry.ligase}, {entry.target}) --- running {len(rotations)} rotations...')

    best_score = -np.inf
    best_rotation, best_tau = None, None
    t0 = time.time()
    for R in rotations:
        rotated = lig_centered @ R.T
        embedded = rotated + fixed_anchor
        charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
        occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
        C_elec = -fft_correlate_3d(pot_r, charge_l)
        overlap_mask = build_overlap_mask(occ_r, occ_l)
        valid_mask = reach_mask & overlap_mask
        if valid_mask.any():
            C_masked = np.where(valid_mask, C_elec, -np.inf)
            idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
            score = C_masked[idx]
            if score > best_score:
                best_score, best_rotation, best_tau = score, R, tau[idx]

    elapsed = time.time() - t0
    posed_ca = (ligase_ca_native - mobile_anchor) @ best_rotation.T + fixed_anchor + best_tau
    rmsd = raw_rmsd(posed_ca, ligase_ca_native)
    print(f'{pdb_id}: done in {elapsed/60:.1f} min, best score={best_score:.1f}, RMSD={rmsd:.2f} A')
    print()
