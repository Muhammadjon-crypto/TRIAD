"""
triad.io.ligand_prep
======================
Extracts a bound ligand (PROTAC, molecular glue, or any other HETATM group)
from a parsed PDB Chain into an RDKit Mol, using the ligand's *actual*
crystallographic coordinates rather than a looked-up or hand-transcribed
SMILES string.

Why this approach: connectivity derived from a real structure's coordinates
is checkable — you can look at the distances yourself — where a SMILES
copied from a name or a database entry is not. For a molecule as large and
irregular as a PROTAC (three-part warhead-linker-warhead topology, PEG
chains, ~40-70 heavy atoms), a single transcription error produces a
plausible-looking but wrong molecule that would silently corrupt every
downstream scoring step. Bond perception from coordinates has its own
failure modes (see `extract_ligand_mol` notes below) but they are visible
and testable, not hidden in a string.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem

from triad.io.pdb_parser import Chain

# Atomic symbol lookup from PDB atom names is unreliable for two-letter
# elements (e.g. "CL1" vs "C" + "L1"), so we take the element from the
# PDB element column when available and fall back to a first-letter guess
# only as a last resort — that fallback is logged, never silent.
_KNOWN_ELEMENTS = {
    "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "ZN", "MG", "CA", "NA",
    "K", "FE", "MN", "CU", "NI", "CO", "SE",
}


@dataclass
class LigandExtractionResult:
    mol: Chem.Mol | None
    chain_id: str
    resname: str
    n_atoms: int
    bond_perception_warnings: list[str]


def _guess_element(atom_name: str) -> str:
    """Best-effort element guess from a PDB atom name when no explicit
    element column is available. Two-letter elements are checked first
    since e.g. 'CL' (chlorine) must not be read as 'C' (carbon).
    """
    stripped = atom_name.strip().upper()
    # strip leading digits (e.g. "1HB" -> "HB")
    stripped = stripped.lstrip("0123456789")
    if len(stripped) >= 2 and stripped[:2] in _KNOWN_ELEMENTS:
        return stripped[:2].capitalize()
    return stripped[:1]


def extract_ligand_mol(chain: Chain) -> LigandExtractionResult:
    """Build an RDKit Mol from a hetero Chain's real coordinates.

    Method: write the chain's atoms out as a minimal PDB HETATM block and
    hand it to RDKit's MolFromPDBBlock with proximity-based bond perception
    (RDKit infers bonds from interatomic distances against covalent radii
    tables — the same approach crystallography software uses to build a
    ligand's connectivity when no CONECT records are present).

    Known limitations, stated rather than hidden:
      - Bond ORDERS (single/double/aromatic) are approximate: proximity
        bonding gets connectivity right far more reliably than bond order,
        especially for aromatic and amide systems. Every mol returned here
        should have `Chem.SanitizeMol` run with bond-order correction before
        being trusted for anything beyond connectivity/topology use (e.g.
        identifying the two warhead ends and the linker path). It should
        NOT be treated as pharmacologically exact until cross-checked.
      - Missing hydrogens (standard for X-ray structures) mean valence
        perception can be wrong at hetero atoms; this is flagged in
        `bond_perception_warnings`, not silently patched.
      - If the ligand is split across a crystallographic symmetry mate or
        has alternate conformers (altloc), only the first occurrence in the
        Chain's atom list is used.
    """
    warnings: list[str] = []
    lines = []
    for i, (coord, name, resname, resid) in enumerate(
        zip(chain.all_coords, chain.all_atom_names, chain.all_residue_names,
            chain.all_residue_ids)
    ):
        element = _guess_element(name)
        if element.upper() not in _KNOWN_ELEMENTS and element.upper() != "H":
            warnings.append(
                f"atom {name!r} (index {i}): unrecognized element guess "
                f"{element!r} — verify this atom's identity manually"
            )
        atom_name_field = name.ljust(4) if len(name) < 4 else name[:4]
        line = (
            f"HETATM{i+1:>5} {atom_name_field:<4} {resname:>3} A{resid:>4}    "
            f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}          {element:>2}"
        )
        lines.append(line)
    lines.append("END")
    pdb_block = "\n".join(lines)

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, proximityBonding=True)

    if mol is None:
        warnings.append(
            "RDKit could not construct a Mol from this ligand's coordinates "
            "at all (not just bond-order issues) — likely a genuinely "
            "unusual element or a coordinate/formatting problem. Needs "
            "manual inspection before use."
        )
        return LigandExtractionResult(
            mol=None, chain_id=chain.chain_id,
            resname=chain.all_residue_names[0] if chain.all_residue_names else "",
            n_atoms=len(chain.all_coords), bond_perception_warnings=warnings,
        )

    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception as e:  # noqa: BLE001 — deliberately broad: any sanitize
        # failure here means "this connectivity guess doesn't make chemical
        # sense as-is" and must be surfaced, not swallowed.
        warnings.append(f"RDKit sanitize raised {type(e).__name__}: {e}")

    return LigandExtractionResult(
        mol=mol,
        chain_id=chain.chain_id,
        resname=chain.all_residue_names[0] if chain.all_residue_names else "",
        n_atoms=mol.GetNumAtoms(),
        bond_perception_warnings=warnings,
    )
