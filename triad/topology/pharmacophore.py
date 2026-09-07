"""
triad.topology.pharmacophore
==============================
Conserved E3-ligase-binding warhead patterns for VHL and CRBN, and the
pose-independent warhead/linker split built on them.

WHY THESE PATTERNS EXIST AS SMARTS, NOT HAND-WRITTEN: earlier attempts at
this module tried (a) proximity-based contact classification, which was
empirically wrong on every tested structure (95-100% of each ligand's atoms
fell within a naive contact cutoff of *some* protein, leaving ~0 "linker"
atoms) and conceptually circular (it requires already knowing the correct
bound pose, which is what the docking search is trying to find); and
(b) considered hand-transcribing a published VHL/CRBN warhead SMILES, which
carries the same transcription-error risk flagged in ligand_prep.py.

The approach used instead: discover each ligase's conserved binding motif
via maximum common substructure (MCS) across multiple benchmark ligands that
share the ligase but differ in TARGET (so the only thing they have in
common is the ligase-binding end, not a shared target scaffold). Each
pattern was then validated against a HELD-OUT structure not used in
discovery, confirming it generalizes rather than overfitting the discovery
set:

  VHL_WARHEAD_SMARTS: discovered from 5T35 (BRD4-BD2), 6HAX (SMARCA2),
    8BDS (BRD4-BD1) -- 22 atoms, matches the known (2S,4R)-hydroxyproline
    + tert-leucine + phenyl VHL-ligand scaffold. Validated against 8FY0
    (BCL-xL, not in discovery set): full 22-atom match.

  CRBN_WARHEAD_SMARTS: discovered from 6BN7 (BRD4-BD1, a PROTAC) and 5HXB
    (GSPT1, a molecular glue) -- 18 atoms, matches the glutarimide-
    isoindolinone (thalidomide/lenalidomide-family) scaffold shared by all
    IMiD-based CRBN binders. Validated against 5FQD (CK1-alpha, not in
    discovery set): full 18-atom match -- and since lenalidomide itself is
    only 19 heavy atoms, this confirms it as a near-complete molecular glue
    with essentially no separate linker/target-warhead structure.

These SMARTS are frozen constants (re-derivable via
`discover_conserved_substructure` below if the benchmark set grows and a
tighter/looser pattern is warranted), not re-computed at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdFMCS

# Frozen, validated patterns -- see module docstring for provenance.
VHL_WARHEAD_SMARTS = (
    "[#6&!R](-&!@[#6&!R]-&!@[#6&!R](-&!@[#6&!R])-&!@[#6&!R])"
    "(-&!@[#7]1-&@[#6](-&!@[#6&!R](-&!@[#8&!R])-&!@[#7&!R]-&!@[#6&!R]"
    "-&!@[#6]2-&@[#6]-&@[#6]-&@[#6]-&@[#6]-&@[#6]-&@2)"
    "-&@[#6]-&@[#6](-&@[#6]-&@1)-&!@[#8&!R])-&!@[#8&!R]"
)
CRBN_WARHEAD_SMARTS = (
    "[#6]1-&@[#6]-&@[#6]-&@[#6]2-&@[#6](-&@[#6]-&@1)"
    "-&@[#6](-&@[#7](-&@[#6]-&@2)-&!@[#6]1-&@[#6]-&@[#6]-&@[#6]"
    "(-&@[#7]-&@[#6]-&@1-&!@[#8&!R])-&!@[#8&!R])-&!@[#8&!R]"
)

_LIGASE_PATTERNS = {
    "VHL": Chem.MolFromSmarts(VHL_WARHEAD_SMARTS),
    "CRBN": Chem.MolFromSmarts(CRBN_WARHEAD_SMARTS),
}


def discover_conserved_substructure(
    mols: list[Chem.Mol], timeout: int = 30
) -> str:
    """Re-derive a conserved-substructure SMARTS via MCS across a list of
    mols. Used to originally derive VHL_WARHEAD_SMARTS / CRBN_WARHEAD_SMARTS
    above (see their provenance in the module docstring) and available here
    for re-deriving a pattern if the benchmark set grows.

    IMPORTANT: mols must share the ligase but differ in target, or this will
    discover whatever scaffold IS shared -- which could be the target
    warhead instead of the ligase warhead if all inputs happen to target the
    same protein (this exact mistake was caught during development: an
    initial 4-structure VHL discovery set all happened to target BRD4-BD2/
    SMARCA2, and returned the shared JQ1-like target scaffold instead).
    """
    mcs = rdFMCS.FindMCS(
        mols, timeout=timeout, ringMatchesRingOnly=True, completeRingsOnly=True
    )
    return mcs.smartsString


@dataclass
class LigaseWarheadSplit:
    ligase_warhead_atoms: list[int]
    linker_atoms: list[int]
    target_side_atoms: list[int]      # target warhead + anything not classified as linker
    exit_atoms: list[int]             # atoms bonded to the ligase warhead but outside it
    is_fully_ligase_warhead: bool     # True when the whole ligand matched (e.g. a molecular glue)
    warnings: list[str] = field(default_factory=list)


def _build_adjacency(mol: Chem.Mol) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def split_by_ligase_pharmacophore(mol: Chem.Mol, ligase: str) -> LigaseWarheadSplit:
    """Split a degrader's atoms into ligase-warhead / linker / target-side
    using the conserved ligase-binding substructure match, NOT proximity to
    any bound protein -- this makes the split an intrinsic property of the
    molecule, valid for any candidate pose, not just the crystallized one.

    Linker definition: starting from each atom just outside the matched
    ligase warhead ("exit atoms"), walk outward through the bond graph while
    atoms are acyclic (not part of any ring) -- this captures typical
    PEG/alkyl/amide linker chemistry. The walk stops at the first ring atom
    encountered, which marks the start of the target-binding warhead's ring
    system. Everything else is "target_side" (a coarser bucket than a true
    target-warhead boundary, since target chemotypes vary per project and
    aren't chemically conserved the way the ligase warhead is).
    """
    warnings: list[str] = []
    if ligase not in _LIGASE_PATTERNS:
        raise ValueError(f"unknown ligase {ligase!r}, expected one of {list(_LIGASE_PATTERNS)}")

    pattern = _LIGASE_PATTERNS[ligase]
    match = mol.GetSubstructMatch(pattern)
    if not match:
        warnings.append(
            f"no {ligase} pharmacophore match found -- this ligand may not "
            f"actually be {ligase}-recruiting, or connectivity perception "
            f"failed for this structure"
        )
        return LigaseWarheadSplit(
            ligase_warhead_atoms=[], linker_atoms=[],
            target_side_atoms=list(range(mol.GetNumAtoms())),
            exit_atoms=[], is_fully_ligase_warhead=False, warnings=warnings,
        )

    warhead_set = set(match)
    n_atoms = mol.GetNumAtoms()

    if len(warhead_set) == n_atoms:
        return LigaseWarheadSplit(
            ligase_warhead_atoms=sorted(warhead_set), linker_atoms=[],
            target_side_atoms=[], exit_atoms=[],
            is_fully_ligase_warhead=True, warnings=warnings,
        )

    adj = _build_adjacency(mol)
    ring_info = mol.GetRingInfo()
    in_ring = [ring_info.NumAtomRings(i) > 0 for i in range(n_atoms)]

    exit_atoms = sorted({
        nbr for atom in warhead_set for nbr in adj[atom]
        if nbr not in warhead_set and mol.GetAtomWithIdx(nbr).GetSymbol() != "H"
    })
    if not exit_atoms:
        warnings.append(
            "ligase warhead match has no external connections -- ligand may "
            "be disconnected or the match spans the whole connected component"
        )

    linker: set[int] = set()
    frontier = set(exit_atoms)
    visited = set(warhead_set)
    while frontier:
        next_frontier = set()
        for atom in frontier:
            if atom in visited:
                continue
            visited.add(atom)
            if mol.GetAtomWithIdx(atom).GetSymbol() == "H":
                continue  # hydrogens are valence filling, not topology
            if in_ring[atom]:
                continue  # ring atom: target-side scaffold begins here, don't expand past it
            linker.add(atom)
            next_frontier |= adj[atom] - visited
        frontier = next_frontier

    target_side = sorted(set(range(n_atoms)) - warhead_set - linker)

    if len(exit_atoms) > 1:
        warnings.append(
            f"{len(exit_atoms)} exit points from the ligase warhead found "
            f"(expected 1 for a typical linear PROTAC) -- this ligand may "
            f"have a branched or unusual topology; linker classification "
            f"may be incomplete"
        )

    return LigaseWarheadSplit(
        ligase_warhead_atoms=sorted(warhead_set),
        linker_atoms=sorted(linker),
        target_side_atoms=target_side,
        exit_atoms=exit_atoms,
        is_fully_ligase_warhead=False,
        warnings=warnings,
    )
