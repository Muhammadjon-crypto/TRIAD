"""
Replaces the broken FFT electrostatics approximation entirely with the
exact pairwise Coulomb sum for a full, autonomous, 3600-rotation search --
the direct, trustworthy re-test of Part 25's original claim (4.59 A on
5T35), now that Part 40 has shown the FFT version was corrupted by
insufficient zero-padding for the non-compactly-supported 1/r kernel.

No FFT is used for electrostatics scoring at all: for each rotation, the
reach+clash valid candidates are found (shape/occupancy grids only, not
implicated by Part 40), and EVERY valid candidate at that rotation is
scored with the exact pairwise sum. This is the actual claim from Part 25
re-tested without any numerical shortcut in the scoring itself.

Run locally: PYTHONPATH=. python3 run_exact_full_search.py
(Full search, all rotations, exact scoring per valid candidate -- expect
roughly 15-30 min depending on how many candidates pass per rotation.)
"""
import time
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import build_receptor_occupancy_grid, build_ligand_shape_grid
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges, coulomb_energy
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

PDB_ID = "5T35"


def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)


entry = BENCHMARK_SET[PDB_ID]
s = load_structure(f"pdb_raw/{PDB_ID}.pdb", structure_id=PDB_ID)
het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0] == entry.ligand_code]
chain = s.chains[het[0]]
r = extract_ligand_mol(chain)
split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)

target_chain = s.chains[entry.target_chains[0]]
ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]

t_coords, t_elem, t_resn, t_resid, t_atomn = extract_representative_atoms(target_chain)
lig_parts = [extract_representative_atoms(c) for c in ligase_chains]
lig_coords = np.concatenate([p[0] for p in lig_parts])
lig_resn = sum([p[2] for p in lig_parts], [])
lig_atomn = sum([p[4] for p in lig_parts], [])
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

occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)

print(f"grid {grid_shape}, {int(reach_mask.sum())} reach-valid voxels per rotation (before clash filtering)")

rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
best_score, best_R, best_tau, n_candidates_scored = -np.inf, None, None, 0
t0 = time.time()
for i, R in enumerate(rotations):
    rotated = lig_centered @ R.T
    embedded = rotated + fixed_anchor
    occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    valid_mask = reach_mask & build_overlap_mask(occ_r, occ_l)
    if not valid_mask.any():
        continue

    valid_idx = np.argwhere(valid_mask)
    for c in valid_idx:
        candidate_tau = tau[tuple(c)]
        posed = embedded + candidate_tau
        score = -coulomb_energy(t_coords, t_charges, posed, lig_charges)
        n_candidates_scored += 1
        if score > best_score:
            best_score, best_R, best_tau = score, R, candidate_tau

    if (i + 1) % 600 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(rotations)} ({elapsed/60:.1f} min), {n_candidates_scored:,} candidates scored so far")

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min. Total candidates scored: {n_candidates_scored:,}")
print(f"Best exact electrostatic score: {best_score:.1f}")

posed_ca = (ligase_ca_native - mobile_anchor) @ best_R.T + fixed_anchor + best_tau
rmsd = raw_rmsd(posed_ca, ligase_ca_native)
print(f"RMSD of best-found pose vs native: {rmsd:.2f} A")
print("\n(Part 25's original FFT-based claim for this exact search was 4.59 A --")
print(" this run answers whether that was ever real.)")
