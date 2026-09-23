"""
Cheap extension of Part 29's preflight predictor to the 3 untested
VHL/BCL-2-family structures (8FY0, 8FY1, 8FY2) -- excluded from the
original n=11 test set only because their much larger reach distance
makes the FULL search expensive. The preflight check itself doesn't run
a search at all, so it costs seconds regardless of structure size.

This does NOT tell us the actual RMSD these structures would recover --
only their preflight score and predicted bucket. Running the full search
to check the prediction is a separate, optional, expensive follow-up.

Run locally: PYTHONPATH=. python3 run_preflight_bcl2.py
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.scoring.electrostatics import assign_formal_charges
from triad.correlation.preflight import preflight_check
from triad.benchmark.manifest import BENCHMARK_SET

for pdb_id in ['8FY0', '8FY1', '8FY2']:
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f'pdb_raw/{pdb_id}.pdb', structure_id=pdb_id)
    het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0]==entry.ligand_code]
    chain = s.chains[het[0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)
    target_chain = s.chains[entry.target_chains[0]]
    ligase_chains = [s.chains[c] for c in entry.ligase_chains[:3] if c in s.chains]

    t_coords = target_chain.all_coords
    t_resn = target_chain.all_residue_names
    t_atomn = target_chain.all_atom_names
    lig_coords = np.concatenate([c.all_coords for c in ligase_chains])
    lig_resn = sum([c.all_residue_names for c in ligase_chains], [])
    lig_atomn = sum([c.all_atom_names for c in ligase_chains], [])
    t_charges = assign_formal_charges(t_resn, t_atomn)
    lig_charges = assign_formal_charges(lig_resn, lig_atomn)

    reach_dist = reach.straight_line_distance_angstrom
    fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
    spacing, pad = 1.5, reach_dist + 15.0
    lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
    hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    result = preflight_check(t_coords, t_charges, lig_coords, lig_charges, grid_shape, lo, spacing)
    print(f'{pdb_id} ({entry.ligase}, {entry.target}): native_score={result.native_score:.1f}, '
          f'predicted_tractable={result.predicted_tractable}, confidence={result.confidence}')
