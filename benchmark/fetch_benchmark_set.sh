#!/usr/bin/env bash
# TRIAD benchmark set — solved PROTAC / molecular-glue ternary complexes.
# Run this LOCALLY on your own machine (RCSB is not reachable from Claude's
# sandbox network). Requires only curl.
#
# Sources for this list:
#   - Gadd et al. 2017, Nat Chem Biol 13:514-521 (5T35, first PROTAC ternary structure)
#   - Benchmark set used in "In Silico Modeling and Scoring of PROTAC-Mediated
#     Ternary Complex Poses" (PMC13242788): 5T35, 6SIS, 6BOY, 6HAX, 6BN7,
#     6HAY, 6HR2, 7KHH
#   - Additional VHL/BRD4-BD1 set (PMC11317996): 8BEB, 8BDS
#   - CRBN molecular glue structures: 5HXB (CC-885/GSPT1), 5FQD (lenalidomide/CK1a)
#   - VHL/BCL-2 family PROTAC set: 8FY0, 8FY1, 8FY2

set -euo pipefail
OUTDIR="pdb_raw"
mkdir -p "$OUTDIR"

# id:ligase:target:degrader — comment only, not parsed
IDS=(
  "5T35"   # VHL   / BRD4-BD2 / MZ1            — Gadd 2017, first PROTAC ternary structure
  "6HAX"   # VHL   / SMARCA2  / ternary
  "6SIS"   # VHL   / ternary complex (benchmark set)
  "6BOY"   # CRBN  / BRD4-BD1 / dBET6
  "6BN7"   # CRBN  / BRD4-BD1 / dBET23
  "6HAY"   # VHL   / ternary complex
  "6HR2"   # VHL   / ternary complex
  "7KHH"   # VHL   / BRD4-BD1 / compound 9
  "8BEB"   # VHL   / BRD4-BD1 / PROTAC 49
  "8BDS"   # VHL   / BRD4-BD1 / PROTAC 4
  "5HXB"   # CRBN  / GSPT1    / CC-885 (molecular glue)
  "5FQD"   # CRBN  / CK1a     / lenalidomide (molecular glue)
  "8FY0"   # VHL   / BCL-xL   / 753b
  "8FY1"   # VHL   / BCL-2    / 753b
  "8FY2"   # VHL   / BCL-2    / WH244
)

for id in "${IDS[@]}"; do
  echo "Fetching $id..."
  curl -sSL "https://files.rcsb.org/download/${id}.pdb" -o "$OUTDIR/${id}.pdb"
done

echo "Done. $(ls "$OUTDIR" | wc -l) structures in $OUTDIR/"
echo "Zip them with: zip -r triad_benchmark.zip $OUTDIR"
