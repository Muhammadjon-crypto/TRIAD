"""
triad.topology.linker_geometry
=================================
Computes how far a degrader's structure extends from its verified
ligase-binding warhead -- both in bond-path length and straight-line
distance. This is the actual geometric constraint the rotational search
will use: two candidate target/ligase orientations are only feasible if the
distance between where each protein would sit is compatible with how far
this specific PROTAC's architecture can physically reach.

Deliberately built on graph eccentricity from the ligase warhead (the
farthest atom, by bond-path length, from the verified warhead match) rather
than trying to pin down one exact "exit atom" -- the warhead/linker boundary
has known ambiguity at the edges (see pharmacophore.py), but "how far does
the molecule extend, maximally, from its verified ligase-binding core" is a
robust question regardless of exactly where decoration ends and linker
begins.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem


@dataclass
class ReachMetrics:
    farthest_atom_idx: int
    path_length_bonds: int              # graph eccentricity from the ligase warhead
    straight_line_distance_angstrom: float   # observed (crystal) reach to that atom
    ligase_warhead_centroid: np.ndarray
    mean_bond_length_along_path: float  # sanity-check: should be ~1.3-1.6 A for real bonds


def _build_adjacency(mol: Chem.Mol) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _multi_source_bfs_distances(
    adj: dict[int, set[int]], sources: set[int]
) -> tuple[dict[int, int], dict[int, int | None]]:
    """BFS from all `sources` simultaneously. Returns (distance, parent) maps
    over every atom reachable from the source set.
    """
    dist = {s: 0 for s in sources}
    parent: dict[int, int | None] = {s: None for s in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                parent[nbr] = node
                queue.append(nbr)
    return dist, parent


def compute_reach_from_warhead(
    ligand_coords: np.ndarray,
    ligand_mol: Chem.Mol,
    warhead_atoms: list[int],
) -> ReachMetrics:
    """Find the atom farthest (by bond-path length) from the given warhead
    atom set, and report both that bond-path length and the real straight-
    line distance to it in the observed (crystal) conformation.
    """
    if not warhead_atoms:
        raise ValueError("warhead_atoms is empty -- nothing to measure reach from")

    adj = _build_adjacency(ligand_mol)
    dist, parent = _multi_source_bfs_distances(adj, set(warhead_atoms))

    unreached = set(range(ligand_mol.GetNumAtoms())) - set(dist.keys())
    if unreached:
        # not necessarily an error -- could be a genuinely disconnected
        # fragment (e.g. a counter-ion) -- but worth knowing about
        pass  # caller can inspect via ligand_mol.GetNumAtoms() vs len(dist) if needed

    farthest_atom = max(dist, key=dist.get)
    path_length = dist[farthest_atom]

    warhead_coords = ligand_coords[warhead_atoms]
    centroid = warhead_coords.mean(axis=0)

    straight_line = float(np.linalg.norm(ligand_coords[farthest_atom] - centroid))

    # reconstruct path and check mean bond length as a sanity check that
    # we're walking real bonds, not something corrupted
    path = [farthest_atom]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    bond_lengths = [
        float(np.linalg.norm(ligand_coords[path[k]] - ligand_coords[path[k + 1]]))
        for k in range(len(path) - 1)
    ]
    mean_bond_length = float(np.mean(bond_lengths)) if bond_lengths else 0.0

    return ReachMetrics(
        farthest_atom_idx=farthest_atom,
        path_length_bonds=path_length,
        straight_line_distance_angstrom=straight_line,
        ligase_warhead_centroid=centroid,
        mean_bond_length_along_path=mean_bond_length,
    )
