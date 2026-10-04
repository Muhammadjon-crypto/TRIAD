"""
Part 39, step 1, redone correctly. Measures how much the electrostatics
search depends on the reach band being centered on the TRUE crystal
distance (unavailable to a real user) versus bounds from chemistry alone.

Differences from run_oracle_leak_test.py (whose numbers were corrupted by
the FFT-electrostatics bug, Part 40-42):
  * Final scoring of every reported pose is EXACT (coulomb_energy).
  * FFT electrostatics is used only as a PRESCREEN: top TOP_K candidates
    per rotation per band, on a grid padded by PAD=60 A. The prescreen is
    validated against exact scores before the search (see validate()).
    If it misses the exact optimum often, results are unreliable and the
    script says so.

Bands (identical rotations and grids for all three):
  A  oracle   |tau - d_native| <= 3.0 A      (what earlier searches used)
  B  strict   |tau| <= 1.5*path_bonds + r_warhead
  C  typical  |tau| <= 1.3*path_bonds + r_warhead
The per-bond factors are stated assumptions; neither uses the crystal
ligand-to-target distance. Not run on real data when written: use
QUICK=1 first to check the plumbing (every 12th rotation, a few minutes).

  QUICK=1 PYTHONPATH=. python3 run_oracle_leak_v2.py
  PYTHONPATH=. python3 run_oracle_leak_v2.py        (full, ~30-40 min)
"""
import os
import time
import numpy as np
from scipy.stats import spearmanr
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import (
    build_receptor_potential_grid, build_charge_grid,
    build_receptor_occupancy_grid, build_ligand_shape_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges, coulomb_energy
from triad.geometry.transforms import sample_rotations
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

PDB_ID = "5T35"
STRICT_PER_BOND, TYPICAL_PER_BOND = 1.5, 1.3
PAD = 60.0
TOP_K = 300
QUICK = os.environ.get("QUICK") == "1"


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

mobile_anchor = reach.ligase_warhead_centroid
fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
d_native = reach.straight_line_distance_angstrom
path_bonds = reach.path_length_bonds
wh = chain.all_coords[split.ligase_warhead_atoms]
r_warhead = float(np.max(np.linalg.norm(wh - mobile_anchor, axis=1)))
d_strict = path_bonds * STRICT_PER_BOND + r_warhead
d_typical = path_bonds * TYPICAL_PER_BOND + r_warhead
print(f"{PDB_ID}: path_bonds={path_bonds}, r_warhead={r_warhead:.1f} A, native d={d_native:.1f} A")
print(f"  A oracle shell {d_native:.1f}+/-3.0 | B strict ball 0-{d_strict:.1f} | C typical ball 0-{d_typical:.1f}")
print(f"  native inside B: {d_native <= d_strict}, inside C: {d_native <= d_typical}")

lig_centered = lig_coords - mobile_anchor
spacing = 1.5
lo = np.minimum(t_coords.min(axis=0), fixed_anchor - d_strict) - PAD
hi = np.maximum(t_coords.max(axis=0), fixed_anchor + d_strict) + PAD
grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
print(f"  grid {grid_shape} = {int(np.prod(grid_shape)):,} voxels (pad {PAD:.0f} A)\n")

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

rotations = sample_rotations(n_axes=150, n_angles_per_axis=24)
if QUICK:
    rotations = rotations[::12]
    print(f"QUICK MODE: {len(rotations)} rotations; results NOT comparable to a full run.\n")


def prepare(R):
    embedded = (lig_centered @ R.T) + fixed_anchor
    charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
    occ_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    C = -fft_correlate_3d(pot_r, charge_l)
    clash_ok = build_overlap_mask(occ_r, occ_l)
    return embedded, C, clash_ok


def exact_score(embedded, t):
    return -coulomb_energy(t_coords, t_charges, embedded + t, lig_charges)


VAL_K = 300


def validate(n_rot=8):
    """Measure how well the large-padding FFT prescreen tracks exact scores.
    For several rotations, score EVERY valid band-B candidate exactly, then
    record (a) Spearman(FFT, exact) and (b) the rank the exact-best candidate
    has in the FFT ordering. The shortlist size is then set from (b)."""
    print("Validating FFT prescreen against exact scores (band B, several rotations)...")
    picks = np.linspace(0, len(rotations) - 1, n_rot).astype(int)
    ranks = []
    for k in picks:
        embedded, C, clash_ok = prepare(rotations[k])
        valid = bands["B strict"] & clash_ok
        idx = np.argwhere(valid)
        if len(idx) < 2 * VAL_K:
            continue
        fft_vals = C[valid]
        ex = np.array([exact_score(embedded, tau[tuple(i)]) for i in idx])
        rho = spearmanr(fft_vals, ex).correlation
        order = np.argsort(-fft_vals)
        rank = int(np.where(order == int(np.argmax(ex)))[0][0]) + 1
        ranks.append(rank)
        print(f"  rotation {k}: {len(idx):,} valid | Spearman={rho:.3f} | "
              f"exact-best has FFT rank {rank}")
    return ranks


ranks = validate()
if ranks:
    TOP_K = int(min(1500, max(300, 3 * max(ranks))))
    print(f"  worst FFT rank of the exact-best candidate: {max(ranks)} "
          f"-> shortlist size TOP_K = {TOP_K} (3x worst rank, capped at 1500)")
    if 3 * max(ranks) > 1500:
        print("  WARNING: needed shortlist exceeds the cap; the search may miss the exact optimum.")
else:
    print("  could not validate; keeping TOP_K = 300, treat results with caution")
print()

best = {k: (-np.inf, None, None) for k in bands}
n_rot_with_valid = {k: 0 for k in bands}
t0 = time.time()
for i, R in enumerate(rotations):
    embedded, C, clash_ok = prepare(R)
    cache = {}
    for name, band in bands.items():
        valid = band & clash_ok
        if not valid.any():
            continue
        n_rot_with_valid[name] += 1
        vals = np.where(valid, C, -np.inf).ravel()
        k = min(TOP_K, int(valid.sum()))
        cand = np.argpartition(-vals, k - 1)[:k]
        for flat in cand:
            if not np.isfinite(vals[flat]):
                continue
            t = tau[np.unravel_index(flat, grid_shape)]
            if flat not in cache:
                cache[flat] = exact_score(embedded, t)
            sc = cache[flat]
            if sc > best[name][0]:
                best[name] = (float(sc), R, t.copy())
    if i == 19:
        per = (time.time() - t0) / 20
        print(f"ETA: {per:.2f} s/rotation -> about {per * len(rotations) / 60:.0f} min total")
    if (i + 1) % 600 == 0:
        print(f"  {i + 1}/{len(rotations)} ({(time.time() - t0) / 60:.1f} min)")

print("\n=== RESULT (exact electrostatics; identical rotations and grids) ===")
print(f"{'constraint':11} {'rot w/ valid':>12} {'best exact':>11} {'|tau| (A)':>10} {'RMSD (A)':>9}")
for name in bands:
    score, R, t = best[name]
    if R is None:
        print(f"{name:11} {n_rot_with_valid[name]:>12} {'-':>11} {'-':>10} {'-':>9}")
        continue
    posed = (ligase_ca_native - mobile_anchor) @ R.T + fixed_anchor + t
    print(f"{name:11} {n_rot_with_valid[name]:>12} {score:>11.1f} {np.linalg.norm(t):>10.1f} "
          f"{raw_rmsd(posed, ligase_ca_native):>9.2f}")
print("\nCheck: band A on a FULL run should come close to the earlier exact result"
      " (best score ~332.1, RMSD 61.10 A). A large gap means the prescreen missed the"
      " optimum or the clash masks differ slightly (different grid origin).")
print(f"native distance {d_native:.1f} A")
