"""
triad.io.pdb_parser
====================
BioPython-backed loading of PDB structures and extraction of per-chain
coordinate arrays for use by the geometry/sampling/scoring modules.

Design note: BioPython's Structure/Chain objects are convenient for parsing
but awkward to rigid-body-transform repeatedly during a rotational search
(thousands of orientations per docking run). So this module extracts each
chain into a lightweight `Chain` dataclass holding plain NumPy coordinate
arrays plus the residue/atom bookkeeping needed to write results back out.
All heavy transform work happens on the NumPy arrays via
`triad.geometry.transforms`, not on the BioPython object.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from triad.geometry.transforms import apply_transform


@dataclass
class Chain:
    """One polymer or heteroatom chain extracted from a PDB structure.

    `all_coords`/`all_atom_names`/`all_residue_ids` are parallel arrays: index
    i of each refers to the same atom. `ca_mask` marks which of those atoms
    are alpha-carbons, used for fast backbone-level scoring and RMSD without
    re-filtering on every call.
    """

    chain_id: str
    all_coords: np.ndarray            # (N, 3)
    all_atom_names: list[str]         # length N
    all_residue_names: list[str]      # length N
    all_residue_ids: list[int]        # length N (author seqid)
    ca_mask: np.ndarray               # (N,) bool
    is_hetero: bool = False

    def ca_coords(self) -> np.ndarray:
        """Alpha-carbon coordinates only, in residue order."""
        return self.all_coords[self.ca_mask]

    def ca_residue_ids(self) -> list[int]:
        return [rid for rid, is_ca in zip(self.all_residue_ids, self.ca_mask) if is_ca]

    def copy(self) -> "Chain":
        return Chain(
            chain_id=self.chain_id,
            all_coords=self.all_coords.copy(),
            all_atom_names=list(self.all_atom_names),
            all_residue_names=list(self.all_residue_names),
            all_residue_ids=list(self.all_residue_ids),
            ca_mask=self.ca_mask.copy(),
            is_hetero=self.is_hetero,
        )

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Chain":
        """Return a new Chain with all coordinates under a rigid-body
        transform applied. Non-mutating — the rotational search evaluates
        many candidate orientations and must not corrupt the original.
        """
        new_chain = self.copy()
        new_chain.all_coords = apply_transform(self.all_coords, R, t)
        return new_chain


@dataclass
class LoadedStructure:
    """A parsed PDB entry: one Chain per author chain ID, plus header
    metadata pulled straight from the PDB HEADER/REMARK 2 records.
    """

    structure_id: str
    chains: dict[str, Chain]
    resolution_angstrom: float | None = None
    header_title: str = ""

    def chain_ids(self) -> list[str]:
        return list(self.chains.keys())

    def protein_chain_ids(self) -> list[str]:
        return [cid for cid, ch in self.chains.items() if not ch.is_hetero]

    def hetero_chain_ids(self) -> list[str]:
        return [cid for cid, ch in self.chains.items() if ch.is_hetero]


def _extract_resolution(structure_header: dict) -> float | None:
    res = structure_header.get("resolution")
    return float(res) if res is not None else None


def load_structure(path: str, structure_id: str | None = None) -> LoadedStructure:
    """Parse a PDB file into a LoadedStructure.

    Heteroatom residues (ligands, the PROTAC/glue molecule itself, ions) are
    split into their own pseudo-chains keyed as "{chain_id}_HET_{resname}",
    since a single author chain can carry both the protein and its bound
    ligand and we need to address them independently downstream (e.g. to
    exclude the ligand from backbone RMSD, or isolate it for linker sampling).

    Waters (HOH) are dropped entirely — irrelevant to rigid-body docking and
    they bloat every downstream computation for no benefit.
    """
    structure_id = structure_id or path.split("/")[-1].split(".")[0].upper()

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PDBConstructionWarning)
        bio_structure = parser.get_structure(structure_id, path)
    if caught:
        print(
            f"[triad.io.pdb_parser] {path}: {len(caught)} BioPython "
            f"construction warning(s) (missing/duplicate atoms, etc.) — "
            f"see REMARK 465/470 in the file for details."
        )

    chains: dict[str, Chain] = {}

    model = next(bio_structure.get_models())  # first model only (X-ray: always 1)
    for bio_chain in model:
        protein_coords, protein_names, protein_resnames, protein_resids, protein_ca = (
            [], [], [], [], []
        )
        het_groups: dict[str, list] = {}  # resname -> list of atom records

        for residue in bio_chain:
            hetflag, resseq, icode = residue.id
            resname = residue.get_resname()

            if hetflag == "W" or resname == "HOH":
                continue  # drop waters

            if hetflag.strip() != "":  # heteroatom residue (ligand, ion, etc.)
                het_groups.setdefault(resname, [])
                for atom in residue:
                    het_groups[resname].append(
                        (atom.get_coord(), atom.get_name(), resname, resseq)
                    )
                continue

            for atom in residue:
                protein_coords.append(atom.get_coord())
                protein_names.append(atom.get_name())
                protein_resnames.append(resname)
                protein_resids.append(resseq)
                protein_ca.append(atom.get_name() == "CA")

        if protein_coords:
            chains[bio_chain.id] = Chain(
                chain_id=bio_chain.id,
                all_coords=np.asarray(protein_coords, dtype=np.float64),
                all_atom_names=protein_names,
                all_residue_names=protein_resnames,
                all_residue_ids=protein_resids,
                ca_mask=np.asarray(protein_ca, dtype=bool),
                is_hetero=False,
            )

        for resname, records in het_groups.items():
            coords = np.asarray([r[0] for r in records], dtype=np.float64)
            names = [r[1] for r in records]
            resnames = [r[2] for r in records]
            resids = [r[3] for r in records]
            key = f"{bio_chain.id}_HET_{resname}"
            chains[key] = Chain(
                chain_id=key,
                all_coords=coords,
                all_atom_names=names,
                all_residue_names=resnames,
                all_residue_ids=resids,
                ca_mask=np.zeros(len(records), dtype=bool),
                is_hetero=True,
            )

    header = bio_structure.header
    return LoadedStructure(
        structure_id=structure_id,
        chains=chains,
        resolution_angstrom=_extract_resolution(header),
        header_title=header.get("name", "").strip(),
    )
