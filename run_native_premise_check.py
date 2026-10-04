"""
Tests two premises from the proposed blueprint against the crystal
structures only (no search, no tuning):

 1. EXIT-VECTOR SIGN. For each native complex, define
      v_E3  = unit vector from the ligase-warhead centroid to the first
              linker/target atom bonded to the warhead
      v_T   = unit vector from the target-warhead centroid to the first
              linker atom bonded to it
    (a PROXY for "direction the pocket points out", since we have no
    explicit pocket-exit definition). The blueprint's rule v_T . v_E3 <= 0
    would wrongly reject any native that violates it; report the dot
    product for every structure.
 2. BSA CONVENTION. Buried surface area of the native interface computed
    with ALL ATOMS and with the REPRESENTATIVE-ATOM set. A fixed BSA cap
    (blueprint: 600 A^2) is only meaningful if the number is stable
    across such conventions and natives sit under it.

Run locally: PYTHONPATH=. python3 run_native_premise_check.py
(15 structures x 2 BSA calculations; a few minutes.)
"""
import numpy as np
from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.scoring.sasa import buried_surface_area
from triad.benchmark.manifest import BENCHMARK_SET


def select_ligase_chains(entry):
    n = len(entry.ligase_chains)
    return list(entry.ligase_chains[:3]) if n == 6 else list(entry.ligase_chains)


def first_exit_neighbor(mol, group, coords):
    """First atom outside `group` bonded to an atom inside it."""
    gset = set(int(i) for i in group)
    for i in sorted(gset):
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j not in gset:
                return j
    return None


rows = []
for pdb_id, entry in sorted(BENCHMARK_SET.items()):
    try:
        s = load_structure(f"pdb_raw/{pdb_id}.pdb", structure_id=pdb_id)
        target_chain = s.chains[entry.target_chains[0]]
        ligase_chains = [s.chains[c] for c in select_ligase_chains(entry) if c in s.chains]

        # --- BSA, two conventions ---
        t_full = target_chain.all_coords
        t_full_el = target_chain.all_elements
        l_full = np.concatenate([c.all_coords for c in ligase_chains])
        l_full_el = sum([c.all_elements for c in ligase_chains], [])
        bsa_full = buried_surface_area(t_full, t_full_el, l_full, l_full_el, n_points=50)

        t_rep, t_rep_el, _, _, _ = extract_representative_atoms(target_chain)
        parts = [extract_representative_atoms(c) for c in ligase_chains]
        l_rep = np.concatenate([p[0] for p in parts])
        l_rep_el = sum([p[1] for p in parts], [])
        bsa_rep = buried_surface_area(t_rep, t_rep_el, l_rep, l_rep_el, n_points=50)

        # --- exit-vector sign ---
        dot = ang = None
        try:
            het = [c for c in s.hetero_chain_ids()
                   if s.chains[c].all_residue_names[0] == entry.ligand_code]
            chain = s.chains[het[0]]
            r = extract_ligand_mol(chain)
            split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
            xyz = chain.all_coords
            lig_wh = split.ligase_warhead_atoms
            tgt = split.target_side_atoms
            j_e3 = first_exit_neighbor(r.mol, lig_wh, xyz)
            j_t = first_exit_neighbor(r.mol, tgt, xyz)
            if j_e3 is not None and j_t is not None and len(tgt) > 0:
                v_e3 = xyz[j_e3] - xyz[list(lig_wh)].mean(axis=0)
                v_t = xyz[j_t] - xyz[list(tgt)].mean(axis=0)
                v_e3 /= np.linalg.norm(v_e3)
                v_t /= np.linalg.norm(v_t)
                dot = float(v_e3 @ v_t)
                ang = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
        except Exception as e:
            print(f"  {pdb_id}: exit-vector part failed -- {type(e).__name__}: {e}")

        rows.append((pdb_id, entry.is_molecular_glue, bsa_full, bsa_rep, dot, ang))
        print(f"{pdb_id}: BSA all-atom={bsa_full:7.0f}  rep-atom={bsa_rep:7.0f}  "
              f"v_T.v_E3={'n/a' if dot is None else f'{dot:+.2f} ({ang:.0f} deg)'}")
    except Exception as e:
        print(f"{pdb_id}: FAILED -- {type(e).__name__}: {e}")

print("\n=== SUMMARY ===")
dots = [(p, d) for p, g, bf, br, d, a in rows if d is not None and not g]
viol = [p for p, d in dots if d > 0]
print(f"Exit-vector rule (v_T . v_E3 <= 0) violated by {len(viol)} of {len(dots)} PROTAC natives: {viol}")
for name, idx in (("all-atom", 2), ("representative-atom", 3)):
    vals = [(r[0], r[idx]) for r in rows]
    over = [p for p, v in vals if v > 600]
    print(f"BSA > 600 A^2 ({name}): {len(over)} of {len(vals)} natives: {over}")
ratios = [r[2] / r[3] for r in rows if r[3] > 0]
print(f"all-atom / representative-atom BSA ratio: median {np.median(ratios):.2f}, "
      f"range {min(ratios):.2f}-{max(ratios):.2f}")
print("\nReading guide: a 600 A^2 cap is only informative if it was set without these natives (use leave-one-out);\nthe exit-vector line is how many PROTAC natives a hard v_T . v_E3 <= 0 rule would reject.")
