# TRIAD Engineering Manifest — v1.0 Baseline & Phase 2 Blueprint

**Status date:** end of v1.0 build session
**Scope:** PROTAC / molecular-glue ternary complex structure prediction engine

---

## Part 1 — Verified v1.0 Core (frozen baseline)

Every module below was tested against real RCSB crystal structures (not synthetic
fixtures alone) and has a corresponding test file that passes. "Verified" means
tested against ground truth, real data, or an analytically known answer — not
merely "runs without crashing."

| Module | Path | What it does | Verified by | Known limitations |
|---|---|---|---|---|
| Rigid-body transforms | `triad/geometry/transforms.py` | SO(3) sampling (Fibonacci-lattice axis-angle grid), rigid transform application, Kabsch alignment | `tests/test_rmsd.py` | — |
| RMSD / Kabsch | `triad/geometry/rmsd.py` | Optimal superposition + RMSD, with reflection correction | `tests/test_rmsd.py` | — |
| PDB ingestion | `triad/io/pdb_parser.py` | BioPython-backed parsing; per-chain `Chain` objects; real element column (not name-guessed); hetero/water handling | `tests/test_pdb_parser.py`, `tests/test_benchmark_set.py` | — |
| Bond perception | `triad/geometry/bond_perception.py` | Covalent-radii distance-based connectivity, outperforms RDKit's default `proximityBonding` on fused-ring systems | `tests/test_bond_perception.py` | Bond ORDER not determined (see below) |
| Ligand extraction | `triad/io/ligand_prep.py` | Builds RDKit Mol from real crystal coordinates + verified bonds | Manual verification across 15/15 benchmark ligands | No reliable bond order/aromaticity without explicit H (13/15 structures lack them) |
| Benchmark manifest | `triad/benchmark/manifest.py` | 15 solved PROTAC/glue ternary structures, DBREF-verified chain assignments (not pattern-guessed) | `tests/test_benchmark_set.py` | 6SIS: documented bond-perception artifact (spurious cycle) |
| Ligase pharmacophore ID | `triad/topology/pharmacophore.py` | VHL (22-atom) and CRBN (18-atom) warhead SMARTS, discovered via MCS across diverse-target structures, validated on held-out data | `tests/test_pharmacophore.py` | Linker/decoration boundary approximate at the edges |
| Reach geometry | `triad/topology/linker_geometry.py` | Graph-eccentricity reach distance from ligase warhead — the real geometric constraint for orientation search | `tests/test_linker_geometry.py` | — |
| Side-chain representation | `triad/topology/sidechain_repr.py` | CA + Cb + functional heteroatoms per residue (reduced but chemically honest) | Manual verification (LYS 349 REMARK 470 cross-check) | Not full-atom; omits aliphatic side-chain carbons between Cb and functional terminus |
| Clash scoring | `triad/scoring/clash.py` | Bondi VdW radii, count + continuous overlap-depth score | Manual validation: near-zero clashes on real crystal structures | — |
| Buried surface area | `triad/scoring/sasa.py` | Shrake-Rupley SASA, validated against analytical sphere-area cases | `tests/test_sasa.py` (5 analytical checks) | Empirically found NOT to discriminate near-native poses at CA or side-chain resolution (see 1.1) |
| Electrostatics | `triad/scoring/electrostatics.py` | Formal-charge Coulomb term, distance-dependent dielectric | `tests/test_electrostatics.py` (8 analytical checks incl. salt-bridge realism) | Empirically found NOT to discriminate near-native poses, with or without distance cutoff (see 1.1) |
| Rotational search (v1) | `triad/sampling/rotational_search.py` | Combinatorial rotation x placement-direction candidate generation | Manual testing on 5T35 | Fundamental limitation, see 1.1. Superseded by Phase 2. |
| Local refinement | `triad/sampling/local_refinement.py` | Finite-difference numerical optimization with backtracking line search, clash + reach-constraint objective | `tests/test_local_refinement.py` (4 synthetic checks incl. multi-seed convergence) | Correctly converges to nearest local optimum but cannot escape a bad starting basin |

### 1.1 — The honest negative-result chain (why Phase 2 exists)

This is the most valuable output of the v1.0 session and deserves to be preserved
in full, not summarized away:

1. Naive contact-count "linker/warhead" classification: **failed** (0-4 atoms
   classified as linker across every structure tested — proximity-based splitting
   is circular for actual docking, since it needs the correct pose to define itself).
2. Pivoted to pharmacophore-based (chemical identity) classification: **worked**,
   validated on held-out data.
3. Rotational search + clash-score-only ranking: weak correlation with RMSD-to-
   native (0.34), best pose found at 6.34 A but ranked 598th of 2160.
4. Added naive distance-window contact-count as interface bonus: **actively
   harmful** — pushed the near-native pose from rank 4 to rank ~1620.
5. Replaced with real, analytically-validated buried surface area (Shrake-Rupley):
   correlation only 0.30, still failed to rank near-native poses well.
6. Diagnosed CA-only resolution as the likely cause (real recognition happens via
   side chains) -> built side-chain representative-atom extraction.
7. Re-tested clash + electrostatics at side-chain resolution: **clash improved**
   (0.34 -> 0.446 correlation); **electrostatics still failed** (-0.07 to -0.11
   correlation, tested with and without distance cutoffs).
8. Diagnosed search density as the dominant remaining bottleneck. Tested 10x
   denser combinatorial grid (24,000 candidates): best RMSD improved 7.25->4.06 A
   but **zero candidates landed within 4 A**.
9. Diagnosed rotation-pivot placement (34 A from the ligase's own center of mass)
   as a possible efficiency killer. **Tested and refuted** — centroid-pivoted
   search gave identical best RMSD (6.34 A) for the same candidate count.
10. **Conclusion, evidence-backed rather than assumed:** naive combinatorial
    (rotation x translation-direction) enumeration is an inefficient algorithm
    class for 6D rigid-body search, full stop — not a parameter-tuning problem.
    This is precisely why production docking tools (Katchalski-Katzir et al. 1992;
    ZDOCK; PIPER) use FFT correlation search instead of explicit enumeration.

---

## Part 2 — Phase 2 Blueprint: 3D FFT Correlation Docking Engine

### 2.1 — Why this specific fix, precisely

Explicit enumeration tries N_rotations x N_translations candidate poses one at a
time. FFT correlation search instead fixes a rotation, then evaluates **every
possible translation simultaneously** via a single 3D Fourier transform pair —
turning an O(N_translations) inner loop into O(N log N) per rotation, where N is
the number of grid voxels. This is not a heuristic speedup; it is an exact
re-formulation of the same brute-force translational scan as a convolution,
computed by the FFT instead of direct summation.

Prior art (real, established methods, not novel to TRIAD):
- Katchalski-Katzir, Shariv, Eisenstein, Friesem, Aflalo, Vakser (1992),
  PNAS — the original FFT correlation docking method (shape complementarity
  only).
- Gabb, Jackson, Sternberg (1997), J. Mol. Biol. — extended the method with an
  electrostatic correlation channel.
- ZDOCK, PIPER, and other modern rigid-body docking tools — same core algorithm,
  additional pairwise-potential channels and post-hoc re-ranking.

### 2.2 — Mathematical specification

**Voxelization.** Both bodies are discretized onto a 3D grid of spacing h
(Angstrom/voxel; typical values 1.0-1.2 A) and dimension N x N x N (N chosen with
small prime factors — powers of 2 preferred — for FFT efficiency; N=64 or 128
depending on molecule sizes and required translational search range).

**Property channels** (each is its own real-valued 3D array):

1. Shape/steric channel (Katchalski-Katzir formulation):
   - Receptor grid rho_r(x): interior (buried) atoms get a large positive
     penalty value rho (discourages any overlap there — a hard clash);
     surface-layer atoms get a small positive value delta; solvent-exposed
     empty space is 0.
   - Ligand grid rho_l(x): occupied voxels (any atom present) = 1, else 0.
   - This reproduces shape complementarity: a good pose gives high correlation
     from surface-surface overlap (small x small, mildly favorable) while any
     receptor-interior/ligand-occupied overlap incurs the large penalty rho,
     just as in the original 1992 method.

2. Electrostatic channel (reusing our validated
   triad/scoring/electrostatics.py formal-charge scheme):
   - Receptor grid: electrostatic potential computed from formal charges
     (Coulomb sum, or Debye-Hueckel screened, evaluated at each grid point).
   - Ligand grid: point charges placed at their real voxel positions.
   - Correlated separately from the shape channel, combined by weighted sum
     (weight is a free parameter to calibrate against the benchmark set).

3. (Deferred) desolvation / knowledge-based contact channel — not built in
   Phase 2 v1; noted here as the natural Phase 3 extension once shape +
   electrostatics correlation is validated end-to-end (matches PIPER's actual
   development history: shape+electrostatics first, pairwise potentials added
   once the core FFT machinery was proven).

**Correlation via the correlation theorem.** For two real 3D grids A
(receptor, fixed) and B (ligand, at a fixed rotation), the cross-correlation

    C(tau) = sum_x  A(x) . B(x + tau)

— i.e. "the score at every possible translation tau simultaneously" — is
computed as:

    C = IFFT3D( FFT3D(A) elementwise-times conj(FFT3D(B)) )

This is the correlation theorem — closely related to, and derivable from, the
convolution theorem by conjugating one operand's transform (equivalent to
spatially reflecting one grid before an ordinary convolution). Each channel
(shape, electrostatic) gets its own correlation grid C_shape, C_elec; the
combined score grid is C_total = C_shape + w * C_elec.

**Per-rotation procedure:**
1. Take a rotation R from our existing, verified sample_rotations() grid
   (triad/geometry/transforms.py — reused, not rebuilt).
2. Rotate the ligand's real atom coordinates by R.
3. Re-voxelize the rotated ligand onto its grid(s).
4. Run the FFT correlation for each channel; combine.
5. Find the voxel(s) with the best combined score — these are the best
   translations *for this rotation*, found in one shot rather than by trying
   each translation individually.
6. Record the top-K poses (rotation + translation) across all rotations
   tried, for downstream local refinement (Part 1's local_refinement.py,
   reused unchanged — it operates on continuous coordinates regardless of how
   the coarse candidate was generated).

**Complexity.** Per rotation: O(N^3 log N) via FFT, vs. O(N^3 . M) for direct
summation over M candidate translations. For realistic grids (N~64-128) and
realistic translational search ranges (M in the thousands to millions of
discrete shifts), this is the difference between seconds and hours per
rotation — the actual, structural fix for the bottleneck diagnosed in Part 1.1,
not an incremental improvement.

**The rotation loop remains explicit** (FFT accelerates translation search
only, not rotation, in this base formulation) — reusing our verified
sample_rotations() grid is appropriate and requires no new validation of the
rotation-sampling logic itself, only of the new voxelization/correlation
machinery built around it.

### 2.3 — Validation plan (same discipline as v1.0, non-negotiable)

Before this touches any real protein data:
1. Synthetic two-sphere test: two spheres of known radius, correlate at a
   known-correct relative translation; confirm the FFT correlation peak lands
   exactly where direct (brute-force) correlation says it should, for a small
   grid where brute-force is still checkable directly.
2. FFT round-trip test: confirm IFFT(FFT(x)) ~= x for a random grid (catches
   basic implementation errors before they hide inside a docking result).
3. Cross-check against brute-force correlation on a small grid (e.g. 16^3):
   direct O(N^6) summation vs. FFT O(N^3 log N) should agree to numerical
   precision. This is the single most important test — any subtle indexing or
   normalization bug in a from-scratch FFT correlation implementation will
   silently produce a plausible-looking but wrong docking result otherwise.
4. Only after 1-3 pass: re-run the 5T35 benchmark case and check whether
   correlation search actually produces a candidate within the true native
   basin (< 4 A), where v1.0's combinatorial search could not.

### 2.4 — Planned file structure (hooks for Phase 2)

    triad/
      correlation/                    # NEW in Phase 2 -- not yet created
        __init__.py
        grid.py                       # voxelization: atoms + properties -> 3D grid
        channels.py                   # shape channel, electrostatic channel definitions
        fft_dock.py                   # core correlation search loop (rotation outer
                                       # loop, FFT correlation inner step per rotation)
        validation/
          test_fft_roundtrip.py       # IFFT(FFT(x)) == x
          test_bruteforce_crosscheck.py   # FFT correlation vs. direct summation
          test_two_sphere_synthetic.py    # known-answer synthetic docking case

      sampling/
        rotational_search.py          # v1.0 combinatorial search -- KEPT for
                                       # reference/comparison benchmarking, not
                                       # deleted; fft_dock.py supersedes it as the
                                       # primary search
        local_refinement.py           # UNCHANGED -- reused as-is for Phase 2
                                       # polish step, since it operates on
                                       # continuous coordinates regardless of how
                                       # the coarse candidate arose

      scoring/
        electrostatics.py             # UNCHANGED logic, but charge assignment
                                       # will feed channels.py's electrostatic
                                       # grid construction
        clash.py, sasa.py             # kept for candidate re-scoring/filtering
                                       # after FFT search narrows to a manageable
                                       # candidate set

No existing v1.0 file needs to change to accommodate this — Phase 2 is purely
additive, sitting alongside the frozen baseline and reusing transforms.py's
rotation sampling and local_refinement.py's polishing step directly.

---

## Part 3 — Repo state at freeze

All test suites passing as of this checkpoint:
test_rmsd.py, test_pdb_parser.py, test_bond_perception.py,
test_benchmark_set.py, test_pharmacophore.py, test_linker_geometry.py,
test_sasa.py, test_electrostatics.py, test_local_refinement.py.

Benchmark data: 15 real PROTAC/molecular-glue ternary structures in
pdb_raw/ (fetched via benchmark/fetch_benchmark_set.sh), DBREF-verified
chain assignments in triad/benchmark/manifest.py. CCD chemical-component
fetch script (benchmark/fetch_ccd.sh) prepared but not yet consumed by any
module — reserved for whenever bond-order-accurate chemistry is needed.

**Next session starts at:** Part 2.3, step 1 — the synthetic two-sphere FFT
correlation test.
