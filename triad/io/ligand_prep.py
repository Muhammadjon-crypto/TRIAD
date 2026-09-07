"""
triad.io.ligand_prep
======================
Extracts a bound ligand (PROTAC, molecular glue, or any other HETATM group)
from a parsed PDB Chain into an RDKit Mol.

Two deliberate design choices, both driven by real bugs found while
building this module against the actual benchmark ligands:

1. Elements come from `Chain.all_elements` (BioPython's parse of the PDB
   file's own element column), never guessed from the atom name. Ligand
   atom names are arbitrary label strings (e.g. "CAV", "CAT") that can
   coincidentally match real element symbols (e.g. "CA" = calcium) despite
   the atom actually being carbon. This was caught on the real MZ1 ligand
   in 5T35: three carbons named CAV/CAT/CAU were briefly misclassified as
   calcium by an earlier name-guessing version of this module.

2. Connectivity comes from `triad.geometry.bond_perception.detect_bonds`
   (explicit covalent-radii cutoffs), not RDKit's `proximityBonding`, which
   was found to add spurious bonds (>2.2 A) in MZ1's fused-ring warhead
   core and raise a valence exception. See bond_perception.py for the
   verification against real coordinates.

What this module does NOT attempt: correct bond ORDERS (single/double/
aromatic). Connectivity is chemically meaningful and directly checkable;
bond order assignment from geometry alone is a harder, separate problem.
Every returned mol has approximate bond orders (RDKit's best guess during
sanitization) and should be treated as reliable for topology (which atoms
connect the two warhead ends and the linker) but not yet for anything that
depends on exact valence/aromaticity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Geometry import Point3D

from triad.io.pdb_parser import Chain
from triad.geometry.bond_perception import detect_bonds, bond_length_sanity_check


@dataclass
class LigandExtractionResult:
    mol: Chem.Mol | None
    chain_id: str
    resname: str
    n_atoms: int
    n_bonds: int
    warnings: list[str] = field(default_factory=list)


def extract_ligand_mol(chain: Chain, bond_tolerance: float = 0.45) -> LigandExtractionResult:
    """Build an RDKit Mol from a hetero Chain's real coordinates and
    elements.

    Steps: detect bonds via covalent-radii cutoffs -> build an editable
    RWMol with those atoms/bonds/3D positions -> sanitize (assigns bond
    orders, aromaticity, valence check) -> report the mol plus any warnings
    encountered along the way. A mol is still returned even when sanitize
    fails partially, so callers can inspect what went wrong rather than
    silently losing the structure.
    """
    warnings: list[str] = []
    n = len(chain.all_coords)
    resname = chain.all_residue_names[0] if chain.all_residue_names else ""

    bonds = detect_bonds(chain.all_coords, chain.all_elements, tolerance=bond_tolerance)
    length_warnings = bond_length_sanity_check(bonds)
    warnings.extend(length_warnings)

    rw = Chem.RWMol()
    for elem in chain.all_elements:
        symbol = elem.strip().capitalize() if elem.strip() else "C"
        try:
            atom = Chem.Atom(symbol)
        except RuntimeError:
            warnings.append(
                f"unrecognized element symbol {elem!r} — defaulted to carbon; "
                f"verify this atom manually"
            )
            atom = Chem.Atom("C")
        rw.AddAtom(atom)

    for i, j, _dist in bonds:
        # detect_bonds can't distinguish bond order, so every bond starts as
        # SINGLE; sanitize() below upgrades to aromatic/double where the
        # resulting valence pattern implies it.
        rw.AddBond(i, j, Chem.BondType.SINGLE)

    conf = Chem.Conformer(n)
    for i, coord in enumerate(chain.all_coords):
        conf.SetAtomPosition(i, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        warnings.append(
            f"RDKit sanitize raised {type(e).__name__}: {e} — connectivity "
            f"may still be usable for topology purposes, but valence/"
            f"aromaticity are not trustworthy for this mol"
        )
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE
                              ^ Chem.SANITIZE_PROPERTIES)
        except Exception:
            pass  # leave mol as-is; caller sees the warning above

    return LigandExtractionResult(
        mol=mol,
        chain_id=chain.chain_id,
        resname=resname,
        n_atoms=mol.GetNumAtoms(),
        n_bonds=mol.GetNumBonds(),
        warnings=warnings,
    )
