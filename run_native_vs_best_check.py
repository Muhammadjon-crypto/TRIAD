"""
Direct mechanistic check for what separates the 'electrostatics-only
search works' group (5T35, 8BDS) from the 'fails' group (6BN7, 6HAX):
how close is native's OWN true electrostatics score to the best score
the full search found anywhere? If native's own score is close to (or
above) the global best, that predicts success; if it's far below, that
predicts failure -- a direct test of whether the mechanism is "native's
own score is competitive" vs "something else is going on."

Run locally: PYTHONPATH=. python3 run_native_vs_best_check.py
(Fast -- single native-pose evaluation per structure, no full search.)
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import build_receptor_potential_grid, build_charge_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.scoring.electrostatics import assign_formal_charges
from triad.benchmark.manifest import BENCHMARK_SET

# best score found by the full 3600-rotation search, from the prior runs
BEST_FOUND = {'5T35': 549.0, '8BDS': 520.0, '6BN7': 567.1, '6HAX': 531.3}
RESULT_GROUP = {'5T35': 'WORKS (4.59A)', '8BDS': 'WORKS (4.17A)', '6BN7': 'FAILS (75.65A)', '6HAX': 'FAILS (77.49A)'}

def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)

for pdb_id in ['5T35', '8BDS', '6BN7', '6HAX']:
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
    lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
    lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
    lig_atomn = sum([c.all_atom_names for c in ligase_chains], [])
    t_charges = assign_formal_charges(t_resn, t_atomn)
    lig_charges = assign_formal_charges(lig_resn, lig_atomn)

    reach_dist = reach.straight_line_distance_angstrom
    mobile_anchor = reach.ligase_warhead_centroid
    fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
    spacing, pad = 1.5, reach_dist + 15.0
    lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
    hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    charge_l_native = build_charge_grid(lig_coords, lig_charges, grid_shape, lo, spacing)
    C_elec_native = -fft_correlate_3d(pot_r, charge_l_native)
    native_score = C_elec_native[0, 0, 0]

    best = BEST_FOUND[pdb_id]
    ratio = native_score / best if best != 0 else float('nan')
    print(f'{pdb_id} [{RESULT_GROUP[pdb_id]}]: native_score={native_score:.1f}, best_found_anywhere={best:.1f}, ratio={ratio:.2f}')
