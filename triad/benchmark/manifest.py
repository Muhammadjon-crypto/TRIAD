"""
triad.benchmark.manifest
=========================
Ground-truth metadata for the TRIAD benchmark set.

Every field here — chain-to-protein assignment, ligand codes, resolutions —
was read directly off the DBREF/HEADER records of the actual files fetched
from RCSB (see benchmark/fetch_benchmark_set.sh), not inferred by pattern-
matching similar entries. That distinction mattered in practice: 6HAX and
6HAY target SMARCA2 and 6HR2 targets SMARCA4, not BRD4 — an assumption based
on "it's VHL-based like the others" would have been wrong for 3 of 15
entries.

`ligase_chains` includes the full obligate E3-ligase assembly (VHL + Elongin
B/C, or CRBN + DDB1) since these dock as one rigid unit against the target,
not the catalytic subunit alone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkEntry:
    pdb_id: str
    ligase: str                 # "VHL" or "CRBN"
    target: str                 # protein being degraded
    degrader_name: str          # PROTAC or molecular glue name (informal)
    ligand_code: str            # PDB het code for the degrader itself
    ligase_chains: tuple[str, ...]   # full ligase assembly (+ Elongin B/C or DDB1)
    target_chains: tuple[str, ...]
    resolution_angstrom: float
    is_molecular_glue: bool = False


BENCHMARK_SET: dict[str, BenchmarkEntry] = {
    "5T35": BenchmarkEntry(
        pdb_id="5T35", ligase="VHL", target="BRD4-BD2", degrader_name="MZ1",
        ligand_code="759",
        ligase_chains=("B", "C", "D", "F", "G", "H"), target_chains=("A", "E"),
        resolution_angstrom=2.70,
    ),
    "6SIS": BenchmarkEntry(
        pdb_id="6SIS", ligase="VHL", target="BRD4-BD2", degrader_name="degrader (LFE)",
        ligand_code="LFE",
        ligase_chains=("B", "C", "D", "F", "G", "H"), target_chains=("A", "E"),
        resolution_angstrom=3.50,
    ),
    "6HAX": BenchmarkEntry(
        pdb_id="6HAX", ligase="VHL", target="SMARCA2", degrader_name="degrader (FWZ)",
        ligand_code="FWZ",
        ligase_chains=("B", "C", "D", "F", "G", "H"), target_chains=("A", "E"),
        resolution_angstrom=2.35,
    ),
    "6HAY": BenchmarkEntry(
        pdb_id="6HAY", ligase="VHL", target="SMARCA2", degrader_name="degrader (FX8)",
        ligand_code="FX8",
        ligase_chains=("B", "C", "D", "F", "G", "H"), target_chains=("A", "E"),
        resolution_angstrom=2.24,
    ),
    "6HR2": BenchmarkEntry(
        pdb_id="6HR2", ligase="VHL", target="SMARCA4", degrader_name="degrader (FWZ)",
        ligand_code="FWZ",
        ligase_chains=("B", "C", "D", "F", "G", "H"), target_chains=("A", "E"),
        resolution_angstrom=1.76,
    ),
    "6BN7": BenchmarkEntry(
        pdb_id="6BN7", ligase="CRBN", target="BRD4-BD1", degrader_name="dBET23",
        ligand_code="RN3",
        ligase_chains=("A", "B"), target_chains=("C",),
        resolution_angstrom=3.50,
    ),
    "6BOY": BenchmarkEntry(
        pdb_id="6BOY", ligase="CRBN", target="BRD4-BD1", degrader_name="dBET6",
        ligand_code="RN6",
        ligase_chains=("A", "B"), target_chains=("C",),
        resolution_angstrom=3.33,
    ),
    "7KHH": BenchmarkEntry(
        pdb_id="7KHH", ligase="VHL", target="BRD4-BD1", degrader_name="compound 9",
        ligand_code="WEP",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=2.28,
    ),
    "8BDS": BenchmarkEntry(
        pdb_id="8BDS", ligase="VHL", target="BRD4-BD1", degrader_name="PROTAC 4",
        ligand_code="QIY",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=1.72,
    ),
    "8BEB": BenchmarkEntry(
        pdb_id="8BEB", ligase="VHL", target="BRD4-BD1", degrader_name="PROTAC 49",
        ligand_code="QIK",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=3.18,
    ),
    "5HXB": BenchmarkEntry(
        pdb_id="5HXB", ligase="CRBN", target="GSPT1", degrader_name="CC-885",
        ligand_code="85C",
        ligase_chains=("B", "C", "Y", "Z"), target_chains=("A", "X"),
        resolution_angstrom=3.60, is_molecular_glue=True,
    ),
    "5FQD": BenchmarkEntry(
        pdb_id="5FQD", ligase="CRBN", target="CK1-alpha", degrader_name="lenalidomide",
        ligand_code="LVY",
        ligase_chains=("A", "B", "D", "E"), target_chains=("C", "F"),
        resolution_angstrom=2.45, is_molecular_glue=True,
    ),
    "8FY0": BenchmarkEntry(
        pdb_id="8FY0", ligase="VHL", target="BCL-xL", degrader_name="753b",
        ligand_code="YF8",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=2.94,
    ),
    "8FY1": BenchmarkEntry(
        pdb_id="8FY1", ligase="VHL", target="BCL-2", degrader_name="753b",
        ligand_code="YF8",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=2.56,
    ),
    "8FY2": BenchmarkEntry(
        pdb_id="8FY2", ligase="VHL", target="BCL-2", degrader_name="WH244",
        ligand_code="YFH",
        ligase_chains=("A", "B", "C"), target_chains=("D",),
        resolution_angstrom=2.98,
    ),
}
