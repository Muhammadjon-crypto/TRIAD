"""
Does the FREE-ligand conformer ensemble contain the bound conformation?

Premise behind a linker-conformer constraint (blueprint Module 1): a pose is
feasible if some conformer of the free PROTAC can bridge the two warheads.
If the free ensemble does not contain the native bound geometry, such a
constraint would reject the correct answer. This tests that premise on the
native structures only (no search).

For each benchmark ligand:
  * build the molecule from the CCD entry (expert bond orders, charges),
    heavy atoms only; assign stereochemistry from the CRYSTAL coordinates
    (otherwise ETKDG samples random stereocenters that can never match);
  * generate N ETKDGv3 conformers (random seed fixed);
  * whole_min   : min over conformers of heavy-atom RMSD to the crystal
                  ligand after optimal superposition on all matched atoms;
  * e2e_min     : min over conformers of the RMSD of the TARGET-side warhead
                  after superposing ONLY the LIGASE warhead (this is what a
                  warhead-pair feasibility test would use);
  * frac<2A     : fraction of conformers with e2e RMSD < 2 A (how easy the
                  bound geometry is to reach).
Limitations: RMSDs ignore symmetry-equivalent atom swaps (slightly
inflated); conformers are not force-field minimized; free-state ensembles
are not bound-state ensembles (that is exactly what is being measured).

  QUICK=1 python3 run_conformer_coverage.py   (60 conformers, few minutes)
  python3 run_conformer_coverage.py           (300 conformers)
  THREADS=2 by default (v2 may still be running); set THREADS=0 for all cores.
"""
import os
import time
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

_BOND = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


def heavy_mol_from_ccd(path):
    """RDKit heavy-atom molecule from a CCD .cif; returns (mol, {atom_name: idx})."""
    d = MMCIF2Dict(path)
    names = _as_list(d["_chem_comp_atom.atom_id"])
    syms = _as_list(d["_chem_comp_atom.type_symbol"])
    chg = _as_list(d["_chem_comp_atom.charge"]) if "_chem_comp_atom.charge" in d else ["0"] * len(names)
    rw = Chem.RWMol()
    idx = {}
    for n, s, c in zip(names, syms, chg):
        if s.upper() in ("H", "D"):
            continue
        a = Chem.Atom(s.capitalize())
        a.SetFormalCharge(int(c) if c not in ("?", ".") else 0)
        idx[n] = rw.AddAtom(a)
    b1 = _as_list(d["_chem_comp_bond.atom_id_1"])
    b2 = _as_list(d["_chem_comp_bond.atom_id_2"])
    bo = _as_list(d["_chem_comp_bond.value_order"])
    for a1, a2, o in zip(b1, b2, bo):
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], _BOND[o.upper()])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol, idx


def kabsch(P, Q):
    """Rotation/translation fitting P onto Q. Apply as (X - Pc) @ R.T + Qc."""
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Pc, Qc


def fit_rmsd(P, Q):
    R, Pc, Qc = kabsch(P, Q)
    A = (P - Pc) @ R.T + Qc
    return float(np.sqrt(np.mean(np.sum((A - Q) ** 2, axis=1))))


def analyze_ligand(cif_path, crystal_names, crystal_xyz, wh_names, tgt_names,
                   n_conf=300, threads=2, seed=7):
    mol, idx = heavy_mol_from_ccd(cif_path)
    cname = [str(n).strip() for n in crystal_names]
    xyz_by_name = {n: np.asarray(x, float) for n, x in zip(cname, crystal_xyz)}
    matched = [n for n in idx if n in xyz_by_name]
    out = {"n_heavy": mol.GetNumAtoms(), "n_matched": len(matched),
           "n_rot": rdMolDescriptors.CalcNumRotatableBonds(mol), "flags": []}
    if len(matched) < 0.9 * mol.GetNumAtoms():
        out["flags"].append("too few atom names matched; skipped")
        return out

    # stereo from the crystal geometry (only possible if every atom has coordinates)
    if len(matched) == mol.GetNumAtoms():
        conf = Chem.Conformer(mol.GetNumAtoms())
        for n, i in idx.items():
            conf.SetAtomPosition(i, xyz_by_name[n].tolist())
        mol.AddConformer(conf, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        mol.RemoveAllConformers()
    else:
        out["flags"].append("stereo NOT constrained (missing atoms)")

    molH = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.numThreads = threads
    ps.pruneRmsThresh = 0.5
    cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_conf, params=ps))
    if not cids:
        ps.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_conf, params=ps))
    out["n_conf"] = len(cids)
    if not cids:
        out["flags"].append("embedding failed")
        return out

    order = [idx[n] for n in matched]
    Q_all = np.array([xyz_by_name[n] for n in matched])
    whole = [fit_rmsd(molH.GetConformer(c).GetPositions()[order], Q_all) for c in cids]
    out["whole_min"], out["whole_med"] = float(np.min(whole)), float(np.median(whole))

    wh = [n for n in wh_names if n in idx and n in xyz_by_name]
    tg = [n for n in tgt_names if n in idx and n in xyz_by_name]
    if len(wh) >= 3 and len(tg) >= 3:
        wi, ti = [idx[n] for n in wh], [idx[n] for n in tg]
        Qw = np.array([xyz_by_name[n] for n in wh])
        Qt = np.array([xyz_by_name[n] for n in tg])
        e2e = []
        for c in cids:
            pos = molH.GetConformer(c).GetPositions()
            R, Pc, Qc = kabsch(pos[wi], Qw)
            T = (pos[ti] - Pc) @ R.T + Qc
            e2e.append(float(np.sqrt(np.mean(np.sum((T - Qt) ** 2, axis=1)))))
        out["e2e_min"] = float(np.min(e2e))
        out["frac_lt2"] = float(np.mean(np.array(e2e) < 2.0))
    else:
        out["flags"].append("warhead split unavailable (glue/degenerate); end-to-end test skipped")
    return out


def main():
    from triad.io.pdb_parser import load_structure
    from triad.io.ligand_prep import extract_ligand_mol
    from triad.topology.pharmacophore import split_by_ligase_pharmacophore
    from triad.benchmark.manifest import BENCHMARK_SET

    quick = os.environ.get("QUICK") == "1"
    n_conf = 60 if quick else 300
    threads = int(os.environ.get("THREADS", "2"))
    print(f"{'ID':5} {'code':4} {'heavy':>5} {'rot':>4} {'conf':>5} {'whole_min':>9} {'whole_med':>9} "
          f"{'e2e_min':>8} {'frac<2A':>8}  notes")
    for pdb_id, entry in sorted(BENCHMARK_SET.items()):
        t0 = time.time()
        try:
            cif = f"ccd_raw/{entry.ligand_code}.cif"
            s = load_structure(f"pdb_raw/{pdb_id}.pdb", structure_id=pdb_id)
            het = [c for c in s.hetero_chain_ids()
                   if s.chains[c].all_residue_names[0] == entry.ligand_code]
            ch = s.chains[het[0]]
            names = list(ch.all_atom_names)
            els = list(ch.all_elements)
            keep = [i for i, e in enumerate(els) if str(e).upper() != "H"]
            names_h = [names[i] for i in keep]
            xyz_h = np.asarray(ch.all_coords)[keep]
            wh_names, tgt_names = [], []
            try:
                r = extract_ligand_mol(ch)
                sp = split_by_ligase_pharmacophore(r.mol, entry.ligase)
                wh_names = [str(names[i]).strip() for i in sp.ligase_warhead_atoms]
                tgt_names = [str(names[i]).strip() for i in sp.target_side_atoms]
            except Exception as e:
                print(f"  ({pdb_id}: split failed: {type(e).__name__})")
            o = analyze_ligand(cif, names_h, xyz_h, wh_names, tgt_names, n_conf=n_conf, threads=threads)
            f = lambda k, w, p=2: f"{o[k]:{w}.{p}f}" if k in o else " " * (w - 1) + "-"
            print(f"{pdb_id:5} {entry.ligand_code:4} {o['n_heavy']:>5} {o['n_rot']:>4} "
                  f"{o.get('n_conf', 0):>5} {f('whole_min', 9)} {f('whole_med', 9)} "
                  f"{f('e2e_min', 8)} {f('frac_lt2', 8, 3)}  {'; '.join(o['flags'])}  ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"{pdb_id:5} FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
