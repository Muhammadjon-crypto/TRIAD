"""
triad.topology.warhead_split
==============================
Splits a bound degrader molecule (PROTAC or molecular glue) into three
functional parts: the target-binding warhead, the E3-ligase-binding
warhead, and the linker connecting them.

This is deliberately built on CONNECTIVITY + 3D PROXIMITY only, not bond
order or aromaticity — see triad/io/ligand_prep.py for why bond-order
perception is unreliable on these heavy-atom-only crystal structures.
Topology (which atoms touch which protein, and the shortest bonded path
between the two contact regions) doesn't need bond order at all, so this
module is fully unblocked by that limitation.

Method:
  1. For every ligand atom, compute the minimum distance to any atom in the
     target protein chains and to any atom in the ligase protein chains
     (using the real bound-complex coordinates — this is a crystal
     structure, so these are the *actual* contacts, not predicted ones).
  2. Atoms within `contact_cutoff` of one partner (and closer to it than the
     other) are seed "contact atoms" for that side.
  3. Each contact set is expanded a few bonds outward via BFS on the
     ligand's bond graph, to capture the full warhead ring system rather
     than just the handful of atoms making the closest contacts.
  4. The shortest bonded path between the two (expanded) contact regions is
     the linker backbone; its bond-length and straight-line end-to-end
     distance are exactly the two numbers that will later constrain which
     relative target/ligase orientations are physically reachable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem


@dataclass
class WarheadLinkerSplit:
    target_contact_atoms: list[int]
    ligase_contact_atoms: list[int]
    target_warhead_atoms: list[int]   # contact atoms expanded outward by BFS
    ligase_warhead_atoms: list[int]
    linker_atoms: list[int]           # everything not in either warhead set
    linker_path: list[int] | None     # shortest bonded path between the two warheads
    linker_path_length_bonds: int | None
    end_to_end_distance_angstrom: float | None
    warnings: list[str] = field(default_factory=list)


def _build_adjacency(mol: Chem.Mol) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _min_distances_to_chain(ligand_coords: np.ndarray, other_coords: np.ndarray) -> np.ndarray:
    """Minimum distance from each ligand atom to any atom in `other_coords`."""
    diffs = ligand_coords[:, None, :] - other_coords[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    return dists.min(axis=1)


def _bfs_expand(adj: dict[int, set[int]], seeds: set[int], n_bonds: int) -> set[int]:
    """Expand a seed atom set outward by up to n_bonds bond-hops."""
    frontier = set(seeds)
    visited = set(seeds)
    for _ in range(n_bonds):
        next_frontier = set()
        for atom in frontier:
            next_frontier |= adj[atom] - visited
        visited |= next_frontier
        frontier = next_frontier
    return visited


def _multi_source_shortest_path(
    adj: dict[int, set[int]], sources: set[int], targets: set[int]
) -> list[int] | None:
    """Shortest path (in bonds) from any atom in `sources` to any atom in
    `targets`, via BFS seeded from all sources simultaneously.
    """
    if sources & targets:
        common = next(iter(sources & targets))
        return [common]

    parent: dict[int, int | None] = {s: None for s in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if node in targets:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for nbr in adj[node]:
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    return None


def split_target_ligase_linker(
    ligand_coords: np.ndarray,
    ligand_mol: Chem.Mol,
    target_coords: np.ndarray,
    ligase_coords: np.ndarray,
    contact_cutoff: float = 4.5,
    warhead_expand_bonds: int = 1,
) -> WarheadLinkerSplit:
    """Classify a bound degrader's atoms into target-warhead / linker /
    ligase-warhead, using real 3D contacts from the crystal structure.

    Parameters
    ----------
    ligand_coords : (N, 3) array, same atom order as ligand_mol
    ligand_mol : RDKit Mol with correct connectivity (bond order not required)
    target_coords, ligase_coords : (M, 3) arrays of the bound protein chains'
        atom coordinates, from the SAME structure as the ligand (this is a
        crystal structure, so these are observed contacts, not predictions)
    contact_cutoff : Angstrom distance defining a "contact" (4.5 A is a
        standard heavy-atom contact distance in structural biology,
        covering direct contacts plus a typical H-bond geometry margin)
    warhead_expand_bonds : how many bonds outward to expand from the seed
        contact atoms, to capture the full ring system rather than only the
        literal closest atoms
    """
    warnings: list[str] = []
    n_atoms = len(ligand_coords)
    if ligand_mol.GetNumAtoms() != n_atoms:
        raise ValueError(
            f"ligand_mol has {ligand_mol.GetNumAtoms()} atoms but "
            f"ligand_coords has {n_atoms} — must be the same molecule"
        )

    dist_to_target = _min_distances_to_chain(ligand_coords, target_coords)
    dist_to_ligase = _min_distances_to_chain(ligand_coords, ligase_coords)

    target_contacts = {
        i for i in range(n_atoms)
        if dist_to_target[i] <= contact_cutoff and dist_to_target[i] < dist_to_ligase[i]
    }
    ligase_contacts = {
        i for i in range(n_atoms)
        if dist_to_ligase[i] <= contact_cutoff and dist_to_ligase[i] < dist_to_target[i]
    }

    if not target_contacts:
        warnings.append(
            f"no ligand atoms within {contact_cutoff} A of the target chain — "
            f"check that target_coords is the correct chain set"
        )
    if not ligase_contacts:
        warnings.append(
            f"no ligand atoms within {contact_cutoff} A of the ligase chain — "
            f"check that ligase_coords is the correct chain set"
        )

    overlap = target_contacts & ligase_contacts
    if overlap:
        warnings.append(
            f"{len(overlap)} atom(s) equidistant to both partners within "
            f"tie-breaking tolerance: {sorted(overlap)} — check contact_cutoff"
        )

    adj = _build_adjacency(ligand_mol)
    target_warhead = _bfs_expand(adj, target_contacts, warhead_expand_bonds) if target_contacts else set()
    ligase_warhead = _bfs_expand(adj, ligase_contacts, warhead_expand_bonds) if ligase_contacts else set()

    both_warheads = target_warhead | ligase_warhead
    linker_atoms = sorted(set(range(n_atoms)) - both_warheads)

    path = None
    path_len = None
    end_to_end = None
    if target_contacts and ligase_contacts:
        path = _multi_source_shortest_path(adj, target_contacts, ligase_contacts)
        if path is None:
            warnings.append(
                "no bonded path found between target-contact and ligase-contact "
                "atoms — the ligand graph may be disconnected (check bond "
                "perception on this structure)"
            )
        else:
            path_len = len(path) - 1
            end_to_end = float(
                np.linalg.norm(ligand_coords[path[0]] - ligand_coords[path[-1]])
            )

    return WarheadLinkerSplit(
        target_contact_atoms=sorted(target_contacts),
        ligase_contact_atoms=sorted(ligase_contacts),
        target_warhead_atoms=sorted(target_warhead),
        ligase_warhead_atoms=sorted(ligase_warhead),
        linker_atoms=linker_atoms,
        linker_path=path,
        linker_path_length_bonds=path_len,
        end_to_end_distance_angstrom=end_to_end,
        warnings=warnings,
    )
