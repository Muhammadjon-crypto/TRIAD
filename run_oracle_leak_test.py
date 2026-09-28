"""
Step 1 of the linker-feasibility direction (manifest Part 39): measure how
much of the electrostatics-only search's success depends on the reach band
being centered on the TRUE crystal distance (an oracle, unavailable to a
real user) versus constraints derived from chemistry alone.

One full search pass, same 3600 rotations, three constraints evaluated on
the identical correlation grids:

  A  ORACLE      |tau - d_native| <= 3.0 A       (what every result so far used)
  B  STRICT      |tau| <= path_bonds*1.5 + r_warhead     (hard chemistry bound)
  C  TYPICAL     |tau| <= path_bonds*1.3 + r_warhead     (plausible extended span)

r_warhead is the largest centroid-to-atom distance inside the ligase
warhead (warhead internal geometry, not ternary-complex information).
The 1.5 and 1.3 A-per-bond factors are stated assumptions: 1.5 is above
typical single-bond lengths, 1.3 is near the projected span per bond of an
all-trans chain. Neither uses the crystal ligand-to-target distance.

Run locally: PYTHONPATH=. python3 run_oracle_leak_test.py
Prints an ETA after the first 100 rotations; abort with Ctrl-C if too slow.
"""
import time
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.correlation.channels import (
    build_receptor_potential_grid, build_charge_grid,
    build_receptor_occupancy_grid, build_ligand_shape_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

PDB_ID = "5T35"
STRICT_PER_BOND = 1.5
TYPICAL_PER_BOND = 1.3


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

mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
d_native = reach.straight_line_distance_angstrom
path_bonds = reach.path_length_bonds

wh_coords = chain.all_coords[split.ligase_warhead_atoms]
r_warhead = float(np.max(np.linalg.norm(wh_coords - mobile_anchor, axis=1)))
d_strict = path_bonds * STRICT_PER_BOND + r_warhead
d_typical = path_bonds * TYPICAL_PER_BOND + r_warhead

print(f"{PDB_ID}: path_bonds={path_bonds}, r_warhead={r_warhead:.1f} A")
print(f"  A oracle : shell {d_native:.1f} +/- 3.0 A")
print(f"  B strict : ball  0 - {d_strict:.1f} A")
print(f"  C typical: ball  0 - {d_typical:.1f} A")
print(f"  native distance inside B: {d_native <= d_strict}, inside C: {d_native <= d_typical}")

lig_centered = lig_coords - mobile_anchor
spacing = 1.5
pad = max(d_strict, d_native) + 15.0
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - d_strict) - pad
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + d_strict) + pad
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
print(f"  grid {grid_shape} = {np.prod(grid_shape):,} voxels\n")

pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)

nx, ny, nz = grid_shape
ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
tau = np.stack([ix * spacing, iy * spacing, iz * spacing], axis=-1)
for d, n in enumerate(grid_shape):
    half = n * spacing / 2
    tau[..., d] = np.where(tau[..., d] > half, tau[..., d] - n * spacing, tau[..., d])
tau_mag = np.linalg.norm(tau, axis=-1)
del ix, iy, iz

bands = {
    "A oracle": np.abs(tau_mag - d_native) <= 3.0,
    "B strict": tau_mag <= d_strict,
    "C typical": tau_mag <= d_typical,
}
best = {k: (-np.inf, None, None) for k in bands}
n_valid_total = {k: 0 for k in bands}

rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
t0 = time.time()
for i, R in enumerate(rotations):
    rotated = lig_centered @ R.T
    embedded = rotated + fixed_anchor
    charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
    occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    C_elec = -fft_correlate_3d(pot_r, charge_l)
    clash_ok = build_overlap_mask(occ_r, occ_l)

    for name, band in bands.items():
        valid = band & clash_ok
        nv = int(valid.sum())
        if nv == 0:
            continue
        n_valid_total[name] += nv
        vals = np.where(valid, C_elec, -np.inf)
        idx = np.unravel_index(np.argmax(vals), vals.shape)
        if vals[idx] > best[name][0]:
            best[name] = (float(vals[idx]), R, tau[idx].copy())

    if i == 99:
        per = (time.time() - t0) / 100
        print(f"ETA: {per:.2f} s/rotation -> about {per * len(rotations) / 60:.0f} min total")
    if (i + 1) % 600 == 0:
        print(f"  {i + 1}/{len(rotations)} ({(time.time() - t0) / 60:.1f} min)")

print("\n=== RESULT (electrostatics-only, identical rotations and grids) ===")
print(f"{'constraint':12} {'valid cands':>12} {'best score':>11} {'RMSD (A)':>9}")
for name in bands:
    score, R, t = best[name]
    if R is None:
        print(f"{name:12} {'0':>12} {'-':>11} {'-':>9}")
        continue
    posed = (ligase_ca_native - mobile_anchor) @ R.T + fixed_anchor + t
    rmsd = raw_rmsd(posed, ligase_ca_native)
    print(f"{name:12} {n_valid_total[name]:>12,} {score:>11.1f} {rmsd:>9.2f}")
print("\n(earlier oracle-band result on this structure: 4.59 A)")
