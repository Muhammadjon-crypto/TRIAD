"""
Is the native pose at (or near) a local optimum of our exact electrostatic
score? Perturb the crystal ligase pose by small rigid-body moves and ask how
often the perturbed pose scores HIGHER than native.

Perturbations (ligase moved, target fixed; rotations about the ligase-warhead
centroid, the same pivot the search uses):
  * translations of 0.5, 1, 2, 3, 5 A in random directions
  * rotations of 2, 5, 10, 20 deg about random axes
  * an "approach scan": shift along the line from ligase centroid toward target
    centroid by -3..+3 A (positive = ligase moves closer)
Each pose is also tested with the SAME clash criterion the search uses
(build_overlap_mask at zero translation), so "scores higher" can be split into
"would pass the clash filter" and "would be rejected".

Score = -coulomb_energy on representative atoms (as in Parts 41-47).
Reading guide: if many small perturbations that PASS the clash filter score
higher than native, native is not near a local optimum of this score, and the
score cannot be used to rank near-native candidates.

Run: PYTHONPATH=. python3 run_native_perturbation.py   (a few minutes per structure)
"""
import numpy as np
from scipy.spatial.transform import Rotation
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import build_receptor_occupancy_grid, build_ligand_shape_grid
from triad.correlation.search import build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges, coulomb_energy
from triad.geometry.rmsd import raw_rmsd
from triad.benchmark.manifest import BENCHMARK_SET

STRUCTURES = ["5T35", "8BDS", "6BN7", "6HAX"]
N_PER_BIN = 60
TRANS = [0.5, 1.0, 2.0, 3.0, 5.0]
ROT = [2.0, 5.0, 10.0, 20.0]
APPROACH = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
rng = np.random.default_rng(0)


def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)


def unit(v):
    return v / np.linalg.norm(v)


def run(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f"pdb_raw/{pdb_id}.pdb", structure_id=pdb_id)
    het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0] == entry.ligand_code]
    chain = s.chains[het[0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)
    target_chain = s.chains[entry.target_chains[0]]
    ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]

    t_coords, _, t_resn, _, t_atomn = extract_representative_atoms(target_chain)
    parts = [extract_representative_atoms(c) for c in ligase_chains]
    lig = np.concatenate([p[0] for p in parts])
    lig_resn = sum([p[2] for p in parts], [])
    lig_atomn = sum([p[4] for p in parts], [])
    ca = np.concatenate([c.ca_coords() for c in ligase_chains])
    t_q = assign_formal_charges(t_resn, t_atomn)
    l_q = assign_formal_charges(lig_resn, lig_atomn)
    t_rad = np.full(len(t_coords), 1.7)
    l_rad = np.full(len(lig), 1.7)
    pivot = reach.ligase_warhead_centroid

    spacing, pad = 1.5, 25.0
    lo = np.minimum(t_coords.min(axis=0), lig.min(axis=0)) - pad
    hi = np.maximum(t_coords.max(axis=0), lig.max(axis=0)) + pad
    shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))
    occ_r = build_receptor_occupancy_grid(t_coords, t_rad, shape, lo, spacing)

    def evaluate(Rm, shift):
        """Pose = rotate ligase about pivot by Rm, then translate by shift."""
        move = lambda X: (X - pivot) @ Rm.T + pivot + shift
        posed = move(lig)
        score = -coulomb_energy(t_coords, t_q, posed, l_q)
        occ_l = build_ligand_shape_grid(posed, l_rad, shape, lo, spacing)
        passes = bool(build_overlap_mask(occ_r, occ_l)[0, 0, 0])
        return score, passes, raw_rmsd(move(ca), ca)

    I = np.eye(3)
    nat_score, nat_pass, _ = evaluate(I, np.zeros(3))
    print(f"\n=== {pdb_id} ({entry.ligase}, {entry.target}) | native score {nat_score:.1f} | "
          f"native passes clash filter: {nat_pass} ===")
    print(f"{'move':>10} {'CA RMSD':>8} {'n_pass':>6} {'%higher(all)':>13} {'%higher(pass)':>14} {'med dScore(pass)':>17}")

    def summarize(label, results):
        sc = np.array([x[0] for x in results]); ps = np.array([x[1] for x in results])
        rm = np.mean([x[2] for x in results])
        hi_all = 100 * np.mean(sc > nat_score)
        if ps.any():
            hi_p = f"{100 * np.mean(sc[ps] > nat_score):13.0f}%"
            med = f"{np.median(sc[ps] - nat_score):17.1f}"
        else:
            hi_p, med = f"{'n/a':>14}", f"{'n/a':>17}"
        print(f"{label:>10} {rm:8.2f} {int(ps.sum()):>6} {hi_all:12.0f}% {hi_p} {med}")
        return sc, ps

    out = {}
    for d in TRANS:
        res = [evaluate(I, d * unit(rng.normal(size=3))) for _ in range(N_PER_BIN)]
        out[f"t{d}"] = summarize(f"{d:g} A", res)
    for th in ROT:
        res = [evaluate(Rotation.from_rotvec(np.radians(th) * unit(rng.normal(size=3))).as_matrix(),
                        np.zeros(3)) for _ in range(N_PER_BIN)]
        out[f"r{th}"] = summarize(f"{th:g} deg", res)

    toward = unit(t_coords.mean(axis=0) - lig.mean(axis=0))
    print("\n  approach scan (positive = ligase moves toward target):")
    print(f"  {'shift A':>8} {'score':>8} {'passes clash':>13}")
    for d in APPROACH:
        sc, ps, _ = evaluate(I, d * toward)
        print(f"  {d:8.1f} {sc:8.1f} {str(ps):>13}")
    return nat_score, out


summary = {}
for pdb_id in STRUCTURES:
    try:
        summary[pdb_id] = run(pdb_id)
    except Exception as e:
        print(f"{pdb_id}: FAILED -- {type(e).__name__}: {e}")

print("\n=== SUMMARY: % of clash-passing perturbations scoring HIGHER than native ===")
print(f"{'':6}" + "".join(f"{k:>9}" for k in [f't{d}' for d in TRANS] + [f'r{t}' for t in ROT]))
for pdb_id, (ns, out) in summary.items():
    row = f"{pdb_id:6}"
    for k in [f"t{d}" for d in TRANS] + [f"r{t}" for t in ROT]:
        sc, ps = out[k]
        row += f"{(100 * np.mean(sc[ps] > ns)):8.0f}%" if ps.any() else f"{'n/a':>9}"
    print(row)
