#!/usr/bin/env bash
# Fetches the real Docking Benchmark 5 (Vreven et al. 2015, J. Mol. Biol.),
# a maintained clean mirror of 231 non-redundant, published protein-protein
# complexes -- used in docs/TRIAD_v1_MANIFEST.md Part 13 to test whether a
# large, diverse (but domain-general) contact potential improves PROTAC
# ternary complex discrimination. It does not -- see Part 13 for the
# finding and its interpretation (likely domain mismatch: natural PPI
# statistics vs. PROTAC-induced neo-interfaces).
#
# Unlike RCSB, github.com IS reachable from Claude's sandbox, so this can
# be run there directly -- included here for reproducibility on your own
# machine too.

set -euo pipefail
git clone --depth 1 https://github.com/haddocking/BM5-clean.git
echo "Done. Real bound-complex structures are in BM5-clean/structures-matched/"
echo "({PDBID}_r_b-matched.pdb = receptor, {PDBID}_l_b-matched.pdb = ligand, per complex)"
