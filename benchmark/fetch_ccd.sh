#!/usr/bin/env bash
# Fetches the RCSB Chemical Component Dictionary (CCD) entry for every
# unique ligand code in the TRIAD benchmark set. These files carry
# expert-curated bond orders and aromaticity -- the authoritative chemistry
# that geometric bond-order perception cannot reliably reconstruct without
# explicit hydrogens (see triad/io/ligand_prep.py module docstring).
#
# Run this LOCALLY, not in Claude's sandbox (RCSB is unreachable from there).

set -euo pipefail
OUTDIR="ccd_raw"
mkdir -p "$OUTDIR"

# Unique ligand codes across all 15 benchmark structures (see triad/benchmark/manifest.py)
CODES=(759 LFE FWZ FX8 RN3 RN6 WEP QIY QIK 85C LVY YF8 YFH)

for code in "${CODES[@]}"; do
  echo "Fetching CCD entry for $code..."
  curl -sSL "https://files.rcsb.org/ligands/download/${code}.cif" -o "$OUTDIR/${code}.cif"
done

echo "Done. $(ls "$OUTDIR" | wc -l) CCD files in $OUTDIR/"
