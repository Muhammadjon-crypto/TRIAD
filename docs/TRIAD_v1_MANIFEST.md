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

---

## Part 4 — Phase 2 progress update

**Status: all three foundational validation gates (Part 2.3, steps 1-3) now PASS.**

Two real bugs were found and fixed in the process — both worth keeping visible
rather than summarizing away, since the second one is a genuinely important
lesson:

1. **Reflection bug**: the first FFT correlation formula tried,
   `IFFT(FFT(A)*conj(FFT(B)))`, was cross-checked against a brute-force
   reference and failed — the correlation peak showed up at `N - tau` instead
   of `tau` for every case tested.
2. **Wrong-ground-truth bug (more important)**: the "fix" applied at the time
   (swapping which operand gets conjugated) made the FFT formula match the
   brute-force reference — but the brute-force reference itself used an
   arbitrary, self-chosen correlation convention (`B(x + tau)`) that was never
   checked against what the codebase actually needs. Building the actual
   two-sphere synthetic docking test exposed this: the "fixed" formula placed
   the correlation peak at the wrong location relative to the real, physically
   meaningful docking question ("if I translate the mobile shape by tau, does
   it land on the target?"). The genuinely correct convention is
   `C[tau] = sum_x A(x) * B(x - tau)`, matching
   `triad.geometry.transforms.apply_transform`'s convention
   (`new_coords = old_coords + tau` implies `new_shape(x) = old_shape(x - tau)`).
   The ORIGINAL (pre-"fix") formula was correct all along; the mistake was
   validating it against a convention that was internally consistent but
   answered a different, less useful question. Both `fft_dock.py`'s docstrings
   and `test_bruteforce_crosscheck.py` now state and justify the corrected
   convention explicitly, tied to `apply_transform`'s convention, not just
   "whichever formula happened to pass."

**Files completed this update:**
- `triad/correlation/fft_dock.py` — `fft_correlate_3d`, `brute_force_correlate_3d`,
  `find_best_translation_index`, all using the corrected, justified convention.
- `triad/correlation/grid.py` — `voxelize_sphere` (periodic/wraparound-aware,
  for synthetic testing), `voxelize_atoms` (real atom coordinates → occupancy
  grid, not yet exercised on real data).
- `triad/correlation/validation/test_fft_roundtrip.py` — 5 checks, all pass.
- `triad/correlation/validation/test_bruteforce_crosscheck.py` — 4 checks, all
  pass (including the physically meaningful self-correlation peak check).
- `triad/correlation/validation/test_two_sphere_synthetic.py` — 4 checks, all
  pass (including recovery of known offsets near the periodic wraparound edge,
  and a quantitative peak-value check, not just peak-location).

**Not yet done:** `triad/correlation/channels.py` (shape/electrostatic channel
construction from real atom data) is still a stub. The rotation-loop-plus-FFT-
per-rotation search driver (tying `sample_rotations()` to `fft_correlate_3d`
across real target/ligase grids) has not been built. Real-data validation
against 5T35 (the actual point of Phase 2) has not yet been attempted — the
synthetic gates had to pass first, and they now have.

**Next session starts at:** building `channels.py`'s shape-complementarity
channel (Katchalski-Katzir style: receptor interior = high penalty, receptor
surface = small reward, per Part 2.2) using real 5T35 atom coordinates, then
wiring it into a rotation-loop driver and checking — for the first time with
real data — whether FFT correlation search finds a candidate within the true
native basin (<4 Å), where v1.0's combinatorial search could not.

---

## Part 5 — Phase 2 channels complete: shape + electrostatics built, validated, honestly characterized

**Status: both planned Phase 2 channels (Part 2.2) are now built and tested
against real 5T35 data, at the correct native rotation.**

**A real modeling bug found and fixed along the way:** the first version of
`build_receptor_shape_grid` defined "surface reward" as voxels AT atom
positions classified as surface-exposed. This is wrong — those voxels are
still inside the receptor's own solid body, so it was rewarding the ligand
for mildly clashing with the receptor's surface atoms, not for occupying the
complementary empty space beside them. Caught by testing against real data:
native alignment scored far below other, non-native translations under the
wrong definition (score 1.0 vs. a global best of 17.0). Fixed using the
correct, standard Katchalski-Katzir definition — voxel-adjacency geometry
(interior = solid voxels with all 26 neighbors also solid, via binary
erosion; surface shell = empty voxels touching the solid, via binary
dilation minus the solid) rather than atom-level burial classification.
Native's score improved to 38.0 after the fix — directionally correct, but
still far from optimal (see below).

**Electrostatic channel**: built via free-space Coulomb convolution
(charge grid convolved with a regularized 1/r kernel, reusing
`triad.scoring.electrostatics`'s `COULOMB_CONSTANT` and formal-charge
scheme for consistency). Validated against exact analytical Coulomb decay
(1/r matched to 6 decimal places at r=1,2,4,8 Å) before touching real data.

**Honest real-data finding (locked in as a permanent regression test,
`test_real_data_5t35.py`):** at 5T35's correct native rotation, restricted
to reach-constrained candidate translations (2,436 of 266,448 total
voxels), neither shape alone nor shape+electrostatics ranks the true native
pose at the top (native ranked ~1,468th and ~1,322nd of 2,436 respectively,
depending on electrostatic weight). This is NOT a contradiction of Phase
2's value — the point of FFT correlation was fixing search *efficiency*
(and it does: the entire reach-constrained shape+electrostatics search over
266,448 translations runs in well under a second, versus the hours a
combinatorial approach would need for equivalent coverage). It confirms,
with real evidence rather than assumption, that shape+electrostatics is
*necessary but insufficient* for discrimination — exactly the same
conclusion v1.0 reached with BSA and pairwise electrostatics, now confirmed
in the new, efficient search framework. This directly matches the real
history of production docking tools: Katchalski-Katzir (1992, shape only)
→ Gabb et al. (1997, +electrostatics) → PIPER and successors (+ a
knowledge-based pairwise-contact statistical potential, which is what
finally gives strong discrimination in practice).

**Files added/completed this update:**
- `triad/correlation/fft_dock.py` — added `fft_convolve_3d` (validated
  against brute-force convolution; needed for the Poisson-style potential
  solve, distinct from `fft_correlate_3d`'s conjugated product)
- `triad/correlation/channels.py` — `build_receptor_shape_grid` (corrected),
  `build_ligand_shape_grid`, `build_charge_grid`, `build_coulomb_kernel`,
  `build_receptor_potential_grid` — all complete, no longer stubs
- `triad/correlation/validation/test_real_data_5t35.py` — 2 tests: exact
  Coulomb-physics check, and the honest "native doesn't rank best" finding
  as a permanent regression (so a future change that suddenly ranks native
  #1 gets flagged for verification, not silently accepted)

**Next session: Phase 3 — knowledge-based pairwise-contact potential.**
This is the third channel every production docking tool needed to add for
real discrimination. Concretely: derive residue-pair contact propensities
from a database of known protein-protein interfaces (not just our 15
structures — a broader interface database is needed for meaningful
statistics), score candidate poses by how well their specific contacts
match learned favorable residue-pair patterns, and add this as a third
correlation channel alongside shape and electrostatics. This was flagged as
option 1 of the two credible paths forward as far back as Part 1.1, step
10's discussion — the evidence built since then (both here and in v1.0)
consistently points to this as the missing piece, not a guess.

---

## Part 6 — Phase 3 contact potential: built, validated, real incremental improvement

**Deliberate choice, stated up front:** rather than hand-transcribing a
published statistical potential (e.g. Miyazawa-Jernigan) from memory —
carrying the same transcription-error risk flagged for hand-written SMILES
in `ligand_prep.py` — this derives a genuine, if small-sample, potential
directly from our own 15 verified structures.

**A real modeling bug found and fixed:** the first derivation used ALL
residues (including deeply buried core) as the "expected by chance"
background population. This inflated expected hydrophobic-pair frequency
(hydrophobic residues are abundant in cores, which can never be at any
interface), making genuine hydrophobic packing look *unfavorable* relative
to chance — directly contradicting one of the most basic facts in protein
biochemistry. Caught by a biochemistry sanity check before this touched any
docking score, not discovered downstream. Fixed by restricting the
background population to surface-exposed residues only (reusing the same
neighbor-count classification approach validated earlier in the project).

**Result after the fix** (`tests/test_contact_potential.py`):
- Hydrophobic-hydrophobic pairs (Leu-Ile, Leu-Leu, Phe-Leu, Val-Ile): all
  strongly favorable, matching textbook biochemistry. **Strict pass,
  asserted.**
- Like-charge pairs (Asp-Glu, Lys-Arg): both strongly unfavorable, matching
  expected electrostatic repulsion. **Strict pass, asserted.**
- Individual salt-bridge pairs (Asp-Lys, Glu-Arg, Asp-Arg): inconsistent
  sign, most likely genuine small-sample noise on these narrower bins
  (15 structures, heavily correlated by shared VHL/CRBN ligases) rather
  than a remaining bug. **Documented as a known limitation, not asserted
  as passing** — the test records the values without a pass/fail
  requirement, honestly separating "this works" from "this doesn't yet."

**Practical impact, tested on 5T35** (same reach-constrained candidate pool
as Part 5's shape+electrostatics test, at the correct native rotation):
adding the contact potential as a third term produced a real, monotonic
improvement — native's rank moved from 1322nd (shape+electrostatics alone)
to 908th of 2436 as the contact-potential weight increased. This is genuine
additional discriminating signal, not noise, though — stated plainly — still
far from making native the top-ranked candidate. Every channel added so far
(shape → +electrostatics → +contact potential) has provided real,
measurable, incremental improvement without yet solving discrimination
outright, which is consistent with how long real production tools took to
mature, not a sign this approach is failing.

**Files added:**
- `triad/potentials/contact_potential.py` — `extract_interface_contacts`,
  `classify_surface_residues`, `derive_contact_potential`
- `tests/test_contact_potential.py` — biochemistry validation (strict on
  hydrophobic packing and charge repulsion, honestly documented as
  unresolved on salt bridges)

**What's still not done:** the contact potential is applied as a direct
per-candidate re-scoring step (looping over reach-valid translations), not
yet wired into the FFT correlation machinery itself as a true fourth-style
channel (which would need the per-residue-type grid decomposition described
in Part 2.2's discussion — up to 20 residue-type grids per body, combined
via the linearity trick: precompute weighted combination grids per receptor
residue type, then correlate). The rotation outer loop (trying more than
just the native rotation) has also still not been exercised — every Phase 2
and Phase 3 test so far has evaluated translational discrimination at the
correct rotation only, to isolate that variable. **Next session: build the
rotation loop, and re-run this same discrimination test across many
rotations, not just the native one — that's the final missing piece before
TRIAD can be tested as an actual, complete pose-prediction pipeline.**

---

## Part 7 — Full rotation loop built and tested: search infrastructure verified correct, discrimination confirmed as the sole remaining bottleneck

**The rotation loop is now built and run end-to-end on 5T35** (180 rotations
via the existing verified `sample_rotations()`, each with a freshly-built
ligand shape+electrostatic grid, FFT-correlated against the precomputed
receptor grids, masked to the reach-constrained sphere, best translation
extracted per rotation, global best kept across all 180). Rotating the
ligand about its own attachment point (rather than an arbitrary pivot) and
embedding the template at the target's attachment point makes the reach
constraint reduce to the elegant `|tau| ≈ reach_distance` — a fixed mask
computed once and reused across every rotation, not recomputed per-rotation.

**Result: 71.98 Å RMSD for the globally best-scoring pose** — far worse
than v1.0's combinatorial search ever found. This demanded a bug hunt
before being accepted as a real finding, not assumed:

1. Verified the pose-application formula itself is exactly correct: feeding
   in the TRUE native rotation and TRUE native translation reproduces RMSD
   of 9×10⁻¹⁶ Å (machine precision zero). No bug in the transform math.
2. Verified with the true native rotation forced into the search: the FFT
   correlation's own best-scoring translation on the reach sphere is NOT
   the true native translation — it picks a different spot 19.76 Å away,
   because shape complementarity alone genuinely scores that other spot
   higher. This is the exact same weakness already found and documented in
   Part 5 (native ranked ~1468th of 2436 by shape alone), now confirmed via
   a completely independent code path (the rotation-loop driver, not the
   earlier direct ranking test) — cross-validation of the same conclusion
   by two different measurements, not a repeated assumption.
3. Conclusion: the 71.98 Å full-search result is fully explained by this
   already-known scoring weakness, COMPOUNDING across a larger search
   space. With 180 rotations × ~2,484 reach-valid translations each, there
   are far more chances for some biologically meaningless (rotation,
   translation) combination to accidentally out-score the true native pose
   (which itself only scores mediocrely, per point 2) than there were when
   testing translations at a single fixed correct rotation. More search
   breadth makes a weak scoring function's failure mode WORSE, not better —
   an important, general lesson: search infrastructure and discrimination
   power are separate problems, and fixing one exposes the other more
   starkly rather than compensating for it.

**This is a genuinely complete, verified conclusion for the full v1.0 +
Phase 2 + Phase 3 system as it stands:**
- Search infrastructure (rotation sampling, FFT-accelerated translation
  search, reach-constraint masking): **verified correct**, no remaining
  bugs found after direct mathematical cross-checking.
- Discrimination power (shape + electrostatics + small-sample contact
  potential): **confirmed, via two independent measurements, as the sole
  remaining bottleneck** — not search coverage, not a plumbing bug.

**Next session's actual task, now unambiguous:** improve discrimination,
not search mechanics. Concrete options, roughly in order of tractability:
(a) a genuinely larger contact-potential training set (the small-sample
15-structure derivation is a real, stated limitation from Part 6); (b) a
real desolvation term (burying a hydrophobic surface without adequate
hydrophobic partner contact is energetically costly — not yet modeled at
all); (c) the previously-discussed 3D Zernike shape descriptors for finer
curvature-based complementarity, appropriately deprioritized earlier this
session until the core engine existed, which it now does and is verified.
Building more search infrastructure (finer rotation grids, more directions)
is NOT the next step — that avenue has been tested and shown not to be
where the problem lives.

---

## Part 8 — Major correction: a hard clash veto changes the picture substantially, plus two more real bugs found

**This significantly revises, without invalidating, Part 7's conclusion.**
Part 7 found native ranked poorly even at the correct rotation. Investigating
the actual top-scoring "competitor" pose revealed it had **237 clashing atom
pairs** — a physically impossible steric overlap that the shape grid's soft
`INTERIOR_PENALTY` (-15.0) failed to reject. Applying a proper HARD clash
veto (using the already-validated `triad.scoring.clash.clash_score`, not
the grid's soft penalty) as a post-filter on the reach-constrained candidate
pool at the correct rotation:

- Of 2,484 reach-valid candidates, only **1,001 are physically valid**
  (clash_score ≤ 5.0) — more than half were clash artifacts inflating the
  apparent competition.
- Among those 1,001 genuinely valid candidates, **native ranks 31st — the
  top 3%**, a dramatic improvement from its earlier (clash-polluted) rank
  of ~1,322-1,468 of 2,436.

**This means shape+electrostatics discrimination is meaningfully BETTER
than Part 7 concluded** — the earlier finding wasn't wrong (native still
isn't rank #1), but its severity was substantially overstated by comparing
native against physically impossible noise.

**Two further real bugs found while fixing this properly, worth keeping
visible:**

1. **`sample_rotations()` redundancy**: rotation angle 0° is generated for
   EVERY sampled axis (`np.linspace(0, 2*pi, n_angles_per_axis,
   endpoint=False)` includes 0), and a 0° rotation about any axis is
   identity regardless of axis choice. For `n_axes=30`, this means 30 of
   180 "rotations" (1 in 6) are literally duplicate identity matrices —
   wasted search budget, not genuine orientation diversity. Not yet fixed
   in `triad/geometry/transforms.py` (flagged here; fixing it changes a
   function several other verified tests depend on, so it needs its own
   careful re-verification pass, not a rushed edit).
2. **Raw shape score is clash-blind for genuinely wrong rotations, even at
   greatly strengthened penalty.** Tested `INTERIOR_PENALTY = -2000` (vs.
   the original -15): at two genuinely different rotations (120° from
   identity, confirmed via rotation-angle calculation — not accidentally
   identity again, unlike the first attempt to test this, which
   mistakenly used rotation index 0, itself identity because of bug #1
   above), the top-30 shape-ranked candidates STILL had zero clash-free
   survivors. This is not a tunable-parameter problem: the receptor-only
   interior/surface voxel classification doesn't correctly penalize a
   badly-oriented ligand that spreads many atoms across the (thin)
   surface shell without individual atoms registering as deep "interior"
   violations. This is an architectural limitation of the simplified
   Katchalski-Katzir-style grid, not something a bigger penalty constant
   fixes.

**Consequence for the search driver:** `triad/correlation/search.py` now
applies a real, explicit hard clash veto (via `clash_score`, threshold
`HARD_CLASH_THRESHOLD = 5.0`) as a post-filter on the top-K FFT-ranked
candidates per rotation, rather than trusting the grid's soft penalty
alone. A real silent-failure bug was also caught and fixed here: the
original implementation, when NO candidate survived the clash filter for
ANY tested rotation, silently returned its uninitialized default
(identity rotation, zero translation, score -inf) as if it were a real
result — this was caught because the reported RMSD (10.31 Å) suspiciously
equaled the reach distance almost exactly, which is what a zero-translation
default would produce. `SearchResult` now has a `found_valid_pose: bool`
field that callers MUST check.

**Honest current state:** with the hard clash veto in place, a full
180-rotation search (many of which are genuinely wasted per bug #1)
struggled to find ANY valid candidate at most non-native rotations at all
(per bug #2) — meaning the full multi-rotation search's real-world
performance has not yet been honestly re-measured end-to-end with all of
this understood. **Next session's concrete tasks, in order:**
1. Fix the `sample_rotations()` redundancy (deduplicate or restructure
   axis/angle sampling so angle=0 isn't repeated per axis), with full
   re-verification of `tests/test_rmsd.py`'s rotation-grid tests.
2. Redesign the shape grid's clash-detection to be genuinely two-body-aware
   (e.g., an explicit ligand-atom-vs-receptor-atom real clash check folded
   into the correlation itself, or a much more conservative interior
   definition) rather than relying on a fixed penalty constant.
3. Only then re-run the full rotation-loop discrimination test and get an
   honest measurement of end-to-end pose recovery — Part 7's 71.98 Å
   number and this session's mid-fix numbers should both be considered
   provisional until 1 and 2 are done.

**Also built and validated this session, status noted for completeness:**
`triad/scoring/desolvation.py` — a simplified (nonpolar-vs-polar, not a
transcribed Eisenberg-McLachlan table) atomic solvation term, validated
against correct physical direction (hydrophobic burial favorable, polar
burial unfavorable, deeper burial more favorable) in
`tests/test_desolvation.py`, all passing. Its first real-data test (on the
same "top-scoring wrong pose" from earlier) is what surfaced the 237-clash
discovery above — the term itself reported an absurd -48,006 energy for
that pose, which was the tip-off that something upstream (the pose being
scored) was physically invalid, not that desolvation itself was broken.
Genuinely applying desolvation to ranking is deferred until the clash-veto
and rotation-sampling fixes above are done — testing it against still-
partially-invalid candidate pools isn't a fair trial of the term itself.

---

## Part 9 — The reconciling finding: rotational sampling density, not shape-grid architecture, is the real remaining gap

**Step 1 of Part 8's fix list is done:** `sample_rotations()`'s angle-0
redundancy bug is fixed (shifted the angle grid by half a bin-width so no
sampled angle is ever exactly 0), verified to produce zero duplicate
rotations where 30 of 180 existed before, and locked in as a permanent
regression (`tests/test_rmsd.py::test_rotation_grid_has_no_duplicate_identity_rotations`).

**Investigating step 2 (redesigning the shape grid) led to a more precise
and more useful finding than the redesign itself would have.** Testing
`top_k_per_rotation=300` (10x the original) at several genuinely different
rotations (30-150 degrees from identity, confirmed by direct angle
calculation) still found **zero** clash-free candidates. That's far more
restrictive than a shape-grid tuning issue would produce — it pointed at
something more fundamental, so the investigation went straight to the
source: **how much rotational error does the true native pose actually
tolerate, at the TRUE native translation?**

Measured directly (`triad/correlation/validation/test_rotational_tolerance.py`):
clash score at the exact correct translation, as a function of pure
rotational deviation from the true native orientation:

| Rotational error | Clash score |
|---|---|
| 0 deg | 0.00 |
| 5 deg | 0.14 |
| 10 deg | 1.98 |
| 20 deg | 13.95 |
| 30 deg | 25.83 |
| 45 deg | 75.00 |

**The tolerance window is narrow: somewhere between 10 and 20 degrees.**
Our 180-rotation grid (30 Fibonacci-sphere axes x 6 angles) has neighboring
samples roughly 35-60 degrees apart — coarser than the tolerance window by
a factor of 2-4x. This is not a coincidence explaining the earlier
zero-survivors findings; it's the direct, quantified cause of them.

**This reconciles, rather than contradicts, Part 7's conclusion.** Part 7
found that FFT-accelerated TRANSLATION search is exhaustive and not the
bottleneck — that remains true; every translation on the reach-constrained
sphere is genuinely evaluated via the correlation. What Part 7 didn't
isolate is that ROTATION is a completely separate dimension, still
brute-force enumerated (FFT never touches it), and our rotation grid's
resolution is simply too coarse relative to how narrow the real geometric
tolerance is. Part 8's clash-veto finding (native ranks 31st of 1,001 at
the CORRECT rotation) and this session's finding (the correct rotation is
surrounded by only a ~10-20 degree valid window) fit together precisely:
discrimination among translations at the right rotation is genuinely
strong; the missing piece is landing in that rotation window at all.

**This is also not a surprising or novel result in the field** — it is the
literal, well-known reason production tools (ZDOCK, PIPER) use enormous
rotation grids (ZDOCK's standard is on the order of 54,000 rotations) —
but it is now a measured, structure-specific fact about our own data,
not an assumption borrowed from the literature.

**Next session's concrete task, now precisely scoped:** substantially
increase rotational sampling density (targeting angular gaps well under
10 degrees between neighboring grid rotations — likely requiring several
thousand to tens of thousands of rotations, consistent with field-standard
tools) and re-run the full search end-to-end with the (now-fixed)
`sample_rotations()`, the hard clash veto already in `search.py`, and the
validated shape+electrostatics scoring — this is the first test that can
honestly answer "does the complete system recover a near-native pose,"
since every previous attempt was confounded by either the clash-veto gap
(Part 8) or coarse rotational sampling (this section). Redesigning the
shape-grid architecture (Part 8's original step 2) is now deprioritized —
the evidence points at sampling density as the dominant remaining lever,
not grid architecture.

---

## Part 10 — Full honest arc closed: infrastructure is now bug-free, and the conclusion returns to discrimination

**Two more real bugs found and fixed while running the denser search Part 9
called for:**

1. **The top-K clash-check strategy silently missed almost all valid
   poses.** Running `n_axes=60, n_angles_per_axis=20` (1,200 rotations,
   genuinely denser than before) with the existing top-K-then-real-clash-
   check approach found **zero** valid poses across all 1,200 rotations.
   Diagnosis: Part 8 already showed raw shape+electrostatics score does
   NOT correlate with clash-validity (top-30 by score were 100% clash
   artifacts even when 40% of ALL candidates were genuinely valid).
   Restricting the real clash check to only the top-K score-ranked
   candidates therefore means the actual valid candidates — which don't
   score especially well by the flawed raw metric — are essentially never
   even considered.

2. **The fix**: replaced the top-K-then-expensive-real-clash-check strategy
   with a THIRD FFT correlation channel — correlating simple binary
   occupancy grids for receptor and ligand gives the exact overlapping-
   voxel count for EVERY translation simultaneously, at the same O(N log N)
   cost as the shape or electrostatic channels. Validated directly against
   real `clash_score` before use: native pose showed 1 overlapping voxel
   (real clash_score 0.0); the known 237-real-clash-pair pose showed 194
   overlapping voxels (real clash_score 182.3) — strong, confirmed
   agreement. This makes exhaustive clash filtering essentially free
   (`OVERLAP_VOXEL_THRESHOLD = 10.0`, a first-pass calibration on this one
   structure, stated as provisional) instead of requiring a per-candidate
   loop, and as a side effect made the search dramatically FASTER (~71ms
   per rotation vs. 177-402ms with the old approach).

**With both the rotation-redundancy fix (Part 9) and this exhaustive,
validated clash channel in place, a genuinely dense search was run:**
150 axes x 24 angles = 3,600 rotations (~9.4 degree axis spacing, 15
degree angle steps — matching the ~10-20 degree tolerance window measured
in Part 9), completed in 4.4 minutes.

**Result: best pose found was 69.45 Å from native.** `found_valid_pose`
correctly returned True (the search did find and correctly identify
genuinely clash-free poses, unlike the earlier failed 1,200-rotation
attempt) — but among all the valid poses found across all 3,600 rotations,
the one shape+electrostatics ranked highest was still far from native.

**This is the honest, now well-supported conclusion:** every plausible
infrastructure explanation has been tested and fixed — rotation sampling
redundancy (Part 9), clash-detection efficiency and completeness (this
section), reach-constraint correctness (Part 7), and pose-application math
(Part 7, exact to machine precision). None of those fixes changed the
fundamental outcome. **The limitation is genuinely in scoring
discrimination** — shape complementarity and simplified electrostatics,
even correctly and exhaustively applied across a properly dense rotational
search with zero known plumbing bugs, do not reliably rank the true native
pose above alternatives. This returns to and reinforces Part 7's original
conclusion, but now on much more solid footing: it can no longer be
attributed to coarse sampling, clash-checking gaps, or rotation redundancy,
because all three have been directly tested, fixed, and shown not to be
sufficient.

**What this means for next steps, concretely:** further infrastructure
work (finer rotation grids still, different clash thresholds, grid spacing
tuning) is very unlikely to be the lever that matters most — that avenue
has now been pushed hard and hit a real ceiling. The path forward is
improving the SCORING itself:
1. Properly integrating the contact potential as a true correlation
   channel (per-residue-type grids, per Part 2.2's original design),
   rather than the current post-hoc re-scoring approach — this hasn't
   been done yet and could plausibly help, especially combined with a
   larger, less small-sample-limited training set.
2. The desolvation term (built and validated in Part 9, not yet applied
   to ranking) — genuinely testing it now makes sense, since the
   candidate pools it would be tested against are finally clash-filtered
   correctly.
3. Accepting that full ab initio shape+physics-based ternary complex
   prediction, without any statistical/ML component trained on a large
   real interface database, may be genuinely at its ceiling with the
   techniques tried so far — which is itself a legitimate, well-earned
   scientific conclusion given the thoroughness of what's been tested,
   not a failure to find the "right" fix.

---

## Part 11 — Desolvation tested on the properly clash-filtered pool: another honest negative result

With the exhaustive overlap-mask now correctly identifying 1,112 genuinely
valid candidates at the true native rotation (close to, and consistent
with, the earlier clash_score-based count of 1,001 — the small difference
is an expected consequence of the two being different, correlated but not
identical, metrics), option 2 from above was tested directly: does adding
the validated desolvation term to the top 100 shape+electrostatics-ranked
valid candidates improve native's rank?

**Result: no — it makes rank worse, monotonically, as its weight increases.**
Native started at rank 27 of these 100 (shape+electrostatics alone,
consistent with the ~31st-of-1,001 finding in Part 8). Adding desolvation:

| Desolvation weight | Native's rank |
|---|---|
| 0.0 (none) | 27 |
| 0.001 | 30 |
| 0.005 | 39 |
| 0.01 | 53 |

This is a genuine, disappointing, but honestly-reported finding — not
hidden because it's unwelcome. Plausible reasons, stated as hypotheses
rather than confirmed diagnoses (not chased further given time already
invested and the pattern already being well-established): the simplified
two-category (nonpolar/polar) solvation model may be too coarse to capture
the specific pattern that favors the true interface over shape-plausible
alternatives, or the atomic-resolution SASA calculation at
representative-atom resolution may not correspond well enough to real
burial geometry to add a correctly-signed correction here.

**This reinforces, rather than complicates, Part 10's conclusion.** Every
physics-based term tried so far — shape complementarity, simplified
electrostatics, a small-sample knowledge-based contact potential (which DID
help, modestly, in Part 6), and now desolvation (which did not help) — has
been individually validated against real physics or real biochemistry, and
tested honestly against real discrimination performance. The overall
picture is consistent: incremental physics-based terms provide, at best,
modest improvement, and the ceiling for ab initio shape+physics scoring
without a substantial statistical/learned component appears real, not an
artifact of any single term being poorly implemented.

**This is a legitimate, well-earned stopping point for the "add more
physics terms" avenue.** The two remaining credible paths, both requiring
substantially more investment than a single further term: (a) a properly
integrated, per-residue-type FFT correlation channel for the contact
potential (not yet built, could still meaningfully help since post-hoc
re-scoring is a weaker test than true joint optimization); (b) a genuinely
larger, more diverse contact-potential training set, addressing the
small-sample limitation stated since Part 6, which would require
structural data beyond this project's 15-structure benchmark.

---

## Part 12 — The properly-joint contact potential channel: built, validated, tested honestly, and closing this investigation

Path (a) from Part 11 was pursued directly: `triad/correlation/contact_channel.py`
implements the true per-residue-type FFT correlation channel described in
Part 2.2's original design, rather than the post-hoc re-scoring used in
Parts 6 and 11.

**The linearity trick**, stated precisely: the full contact-potential score
`sum_{i,j} potential(i,j) * correlate(Receptor_i, Ligand_j)` — naively up to
400 correlations per rotation (20 receptor residue types x 20 ligand
residue types) — reduces to exactly 20 correlations via `correlate(Receptor_i,
sum_j potential(i,j)*Ligand_j)`, precomputing one "weighted ligand
combination grid" per receptor residue type first. **Validated against
brute-force computation to machine precision** on both a sparse hand-built
potential and a dense randomized one covering nearly all 210 residue pairs
(`triad/correlation/validation/test_contact_channel.py`, 2/2 passing) before
touching any real data — the same discipline as every other from-scratch
correlation reformulation this session.

**Tested honestly on 5T35** (same correct rotation, same clash-filtered
1,112-candidate pool as Parts 8 and 11):

| Contact-potential weight | Native's rank |
|---|---|
| 0.0 (shape+elec only) | 54 |
| 1.0 | 52 |
| 3.0 | 70 |
| 5.0 | 80 |
| 10.0 | 120 |

**Result: a negligible improvement at low weight (52 vs. 54 — noise-level),
then monotonically worse beyond that.** This is a materially different,
and more rigorous, test than Part 6's post-hoc version (which showed a
more encouraging-looking rank improvement from 1322 to 908) — the
difference is almost certainly because Part 6's candidate pool still
included clash artifacts that this properly-joint, correctly clash-
filtered test does not. Once compared on a fair, physically-valid
candidate pool, the earlier apparent improvement mostly evaporates.

**This closes the "add more physics/statistical terms" investigation
honestly.** Four separately-validated terms have now been tested against
correctly clash-filtered real data: shape (foundational), electrostatics
(modest, real contribution — Part 8's clash-veto finding), a small-sample
knowledge-based contact potential (negligible to negative, both as post-hoc
re-scoring and as a properly-joint correlation channel), and desolvation
(negative, Part 11). The consistent picture across all four, tested with
equal rigor: **shape complementarity plus a simplified electrostatic term
is very close to the ceiling of what ab initio physics-based scoring
achieves here without a substantially larger statistical foundation.**

**What remains genuinely untried, for a real future session:** a contact
potential trained on a large, diverse protein-protein interface database
(hundreds to thousands of independent structures, not 15 correlated ones)
is the one lever in this whole investigation that hasn't actually been
built and tested — everything else in the "improve scoring" category has
now been tried, validated, and honestly found insufficient. That remains
the most credible next step, and it is fundamentally a data problem, not
an algorithm problem — the correlation machinery to use such a potential,
once available, already exists and is verified (`contact_channel.py`).

---

## Part 13 — The large-dataset lever tested directly: also insufficient, but for an interesting, specific reason

**The one untried credible lever from Part 12 was pursued and tested.**
`github.com` (unlike RCSB) is reachable from this sandbox, which made it
possible to directly clone `github.com/haddocking/BM5-clean` — a maintained
mirror of the real, published Docking Benchmark 5 (Vreven et al. 2015,
*J. Mol. Biol.* 427:3031-3041): **231 non-redundant, independent, diverse
protein-protein complexes**, fetchable via `benchmark/fetch_bm5_interfaces.sh`.

**A real methodology bug was caught before trusting this data**: the first
processing attempt fed raw all-atom coordinates into `classify_surface_residues`
(which was calibrated for the reduced representative-atom resolution used
elsewhere in this project), producing only 946 "surface" background
residues across all 231 complexes — implausibly low, since a full-atom
neighbor-count threshold tuned for a sparser representation classifies
almost everything as buried. Fixed by consistently using
`extract_representative_atoms` for both the BM5 processing and the original
15-structure derivation, giving a fair, apples-to-apples comparison:
25,404 background residues and 18,115 contact observations — a genuine,
~47x increase in contact data over the original 383.

**Biochemistry validation on the larger dataset**: hydrophobic-hydrophobic
pairs remain robustly favorable (LEU-ILE +1.26, LEU-LEU +1.23, PHE-LEU
+2.26, VAL-ILE +1.00) and like-charge pairs remain robustly unfavorable
(ASP-GLU -1.41, LYS-ARG -1.72) — both categories that aggregate signal
across many residue-type combinations. Salt bridges, however, **remain
inconsistent even with 47x more data** (ASP-LYS -0.16, GLU-ARG +0.12,
ASP-ARG +0.70, GLU-LYS -0.91) — ruling out small sample size as the
explanation for that specific inconsistency; it appears to be a genuine
property of simple residue-type contact-frequency statistics (specific
salt bridges require precise mutual geometric orientation that a
type-frequency count doesn't capture), not a data-quantity artifact.

**The decisive test: does the BM5-derived potential improve discrimination
on 5T35?** Tested identically to Part 12's methodology (same correct
rotation, same 1,112-candidate clash-filtered pool):

| Contact-potential weight (BM5-derived) | Native's rank |
|---|---|
| 0.0 | 54 |
| 0.5 | 56 |
| 1.0 | 60 |
| 2.0 | 74 |
| 5.0 | 111 |

**No improvement — the same monotonically-worsening pattern as the
small-sample version.** This definitively rules out "not enough training
data" as the explanation for the contact potential's weak performance.

**The most credible interpretation, and the actual insight this test
provides:** Docking Benchmark 5 is built almost entirely from **natural,
evolutionarily-selected protein-protein interfaces** (antibody-antigen,
enzyme-inhibitor, and similar complexes that evolution optimized for
binding). A PROTAC-induced ternary complex is fundamentally different — a
**"neo-interface"** forced into proximity by a small-molecule linker,
never subject to any evolutionary selection for interface quality. The
statistical patterns that describe how natural interfaces pack may simply
not transfer across that domain boundary, regardless of how much natural-
PPI data is used. This reframes the earlier "small sample" concern: the
original 15-structure, PROTAC-specific dataset may actually be the more
*relevant* data source for this exact problem — just still too small (and
too internally correlated, sharing only two ligases) to derive reliable
statistics from alone. **The real missing dataset is not "more protein-
protein interfaces in general" but "more PROTAC/molecular-glue ternary
complexes specifically"** — a genuinely scarce resource, since the whole
field has solved only a few dozen such structures to date.

**This closes the physics/statistical scoring investigation completely,
with a specific, well-reasoned conclusion rather than an open question:**
five terms tested (shape, electrostatics, small-sample contact potential,
desolvation, large-sample general-PPI contact potential), all individually
validated against real physics, real biochemistry, or brute-force
computation, all tested with equal rigor against genuinely clash-filtered
real data. The ceiling is real, and its most likely cause — a domain
mismatch between general protein-protein interface statistics and PROTAC-
specific neo-interfaces — is itself a legitimate, citable scientific
finding about why this problem remains hard, not a dead end reached by
process of elimination alone.

---

## Part 14 — CRITICAL CORRECTION: Parts 7-13 were tested on a sample size of ONE structure

**This is a significant methodological gap, caught and fixed, not a
minor addendum.** Every discrimination test in Parts 7 through 13 — the
clash-veto finding, the rotational tolerance window, the desolvation
result, the BM5 domain-mismatch conclusion — was run exclusively on 5T35.
No claim about "the ceiling of physics-based scoring" was ever tested
against more than one data point. This should have been caught much
earlier, and it changes the overall picture substantially.

**The exact same methodology (correct rotation, reach-constrained,
exhaustively clash-filtered via the occupancy-overlap channel, ranked by
shape+electrostatics) was run across all 14 non-6SIS benchmark structures.**
Native's percentile rank among genuinely valid candidates:

| Structure | Percentile | Group |
|---|---|---|
| 8FY0 | 4.3% | TRACTABLE |
| 5T35 | 4.9% | TRACTABLE |
| 6BN7 | 6.4% | TRACTABLE |
| 6BOY | 8.2% | TRACTABLE |
| 8FY2 | 10.9% | TRACTABLE |
| 8FY1 | 12.0% | TRACTABLE |
| 5HXB | 14.8% | TRACTABLE |
| 8BDS | 26.7% | TRACTABLE |
| 6HAX | 52.9% | RANDOM |
| 7KHH | 55.5% | RANDOM |
| 5FQD | 55.7% | RANDOM |
| 8BEB | 56.6% | RANDOM |
| 6HR2 | 58.9% | RANDOM |
| 6HAY | 64.0% | RANDOM |

**This directly overturns Part 13's "uniform domain mismatch" conclusion.**
8 of 14 structures (57%) show genuinely good discrimination — native
lands in the top 5-27% of a physically valid candidate pool, which is a
real, useful signal for a pre-filtering tool, not random noise. The other
6 (43%) show near-random discrimination (50-64th percentile). Part 13's
conclusion was drawn from the single worst-documented structure treated
as if it were representative; it was an overstated generalization from
n=1, now corrected by n=14.

**What actually distinguishes the two groups is, honestly, not yet known.**
The cleanest test case is 8BDS (tractable, 26.7%) vs. 8BEB (random,
56.6%) — same ligase (VHL), same target (BRD4-BD1), nearly identical reach
distance (5.81 vs 5.76 A), and an IDENTICAL valid-candidate pool size
(820). Five candidate explanatory features were checked directly against
real data, not assumed:
- Reach distance: no clean relationship (correlation not dominant; e.g.
  8BDS and 8BEB are nearly identical yet oppositely classified)
- Linker path length (bonds): weak, inconsistent (correlation -0.224;
  6HAX has only 7 linker atoms and is RANDOM, 8FY2 has only 6 and is
  TRACTABLE)
- Ligase identity (VHL vs. CRBN): both groups contain both ligases
- Molecular glue vs. PROTAC: 5FQD (glue) is RANDOM, but this is a single
  data point, not a pattern
- Crystal resolution: moderate correlation (-0.558) but with a clear
  counter-example (6HR2 at 1.76 A, excellent resolution, is RANDOM)

**None of these five single-feature hypotheses cleanly explains the
split.** This is itself useful, honestly-reported negative information —
it rules out several "obvious" explanations rather than settling on the
first plausible-looking one. The real driver is likely either a
multivariate combination of these features, or something in the specific
3D shape/packing geometry not captured by any of these scalar summaries
(e.g. how much of the true interface area the linker itself directly
contributes versus how much is contributed by the two rigid warhead-
adjacent regions alone).

**Corrected overall conclusion:** TRIAD's shape+electrostatics scoring is
NOT uniformly at a hard ceiling for PROTAC ternary complexes — it works
genuinely well for a majority of tested real cases. What determines
whether a specific case is tractable remains a real, open, well-scoped
question for future investigation, not a solved problem and not a wall.
This is a substantially more interesting and more actionable place to be
than Part 13's conclusion suggested.

---

## Part 15 — Cross-platform floating-point sensitivity found and fixed in the new benchmark test

Running the Part 14 benchmark on a different machine (Apple Silicon Mac,
vs. this session's Linux x86_64 sandbox) surfaced a real, informative
issue: `n_valid` (the count of reach+clash-valid candidates) differed by a
handful of candidates for 10 of 14 structures — e.g. 115 vs. 118, 2501 vs.
2509, 26943 vs. 26904. Diagnosed before assuming anything was broken:

1. Confirmed the computation is perfectly deterministic across repeated
   runs WITHIN one environment (3 consecutive runs on the same sandbox
   gave identical results) — ruling out any randomness bug in the code
   itself.
2. The most likely explanation: `reach_mask` and `overlap_mask` both use
   hard `<=` threshold comparisons on continuous FFT-derived values. FFT
   implementations are not guaranteed bit-identical across different
   BLAS/hardware backends (e.g. Apple's Accelerate framework vs. a Linux
   BLAS build) — tiny (~1e-10 relative) differences in the FFT output can
   push a handful of candidates that sit almost exactly at a hard
   threshold boundary to opposite sides on different machines. This is a
   well-known, expected class of behavior in scientific computing, not a
   correctness bug — the differences observed are consistently a small
   fraction of a percent of the total candidate count.

**Real bug also found in the test's own design, independent of the
floating-point issue**: the original test asserted `n_valid` equality
BEFORE checking rank, so when `n_valid` mismatched (as it did for 10 of 14
structures on the Mac), the test never even reported what rank was
actually measured — meaning the cross-platform run gave no visibility
into whether the actually important finding (tractable vs. random
classification) held up at all. Fixed by:
- Always computing and reporting both metrics regardless of pass/fail
- Replacing exact-equality assertions with a tolerance band (n_valid
  within 2%, percentile within 5 percentage points of the baseline) —
  appropriate given the floating-point finding above, and sufficient to
  still catch a genuine regression (a real scoring change would move
  percentiles by far more than 5 points, as seen throughout this
  manifest's own history of real improvements and regressions)

**Practical implication for anyone running this benchmark:** don't expect
bit-exact reproduction of `n_valid` or `rank` across different machines —
expect the percentile classification (tractable vs. random) to hold, and
treat a difference larger than a few percentage points as worth
investigating, not a difference of a handful of counts at a threshold
boundary.

**Confirmed empirically on a second, independent machine** (Apple Silicon
Mac, vs. this session's Linux x86_64 sandbox): 12 of 14 structures matched
within the original 2% tolerance outright; the other 2 (5FQD, 6HR2)
exceeded it only marginally (2.01%, 2.61%) while their percentiles moved
by under 1 point and stayed classified identically (both remained in the
"random" group). Tolerance widened to 3% to reflect this as normal,
harmless cross-platform variation rather than a borderline failure.

**Most importantly: the qualitative Part 14 finding — 8 of 13 real PROTAC
structures show genuinely good discrimination, 5 show near-random — held
EXACTLY, structure-for-structure, on both machines.** This is now a
cross-platform-verified result, not an artifact of one environment's
floating-point behavior.

---

## Part 16 — CRITICAL BUG: 3 of 15 structures had target/ligase chains that weren't actually in contact, invalidating the 8BDS-vs-8BEB investigation

**While investigating what distinguishes 8BDS (tractable) from 8BEB
(random) — the "cleanest matched pair" from Part 14 — a direct sanity
check revealed the real native pose showed a shape-correlation score of
exactly 0.0 (floating-point noise) for BOTH structures.** Rather than
accept this and move on, it was chased to the actual root cause: at what
was being treated as "8BDS's native pose," the minimum distance between
ANY target atom and ANY ligase atom was **21.9 Å** — nowhere near a real
bound complex (real interfaces have direct contact, <5 Å).

**Checked across all 15 structures immediately given the severity:**
exactly 3 (7KHH, 8BDS, 8BEB) showed this same failure (19.5-21.9 Å instead
of <4 Å). All three share the same structural feature: their
`ligase_chains` manifest entry has exactly 3 chains (ElonginB + ElonginC +
VHL, one complete assembly), while 5 other structures (5T35, 6HAX, 6HAY,
6HR2, 6SIS) have 6 chains (two full copies of that same 3-chain assembly).

**Root cause, found and fixed**: the test script's chain-selection logic
(`n_copy = len(ligase_chains)//2 if len(ligase_chains) > 2 else ...`)
assumed any list of more than 2 chains must represent two copies and
halved it. This is correct for the genuine 6-chain, two-copy cases, but
WRONG for the 3-chain, single-copy cases — it truncated a complete
functional assembly down to just the first chain (ElonginB alone,
dropping ElonginC and VHL). ElonginB never directly contacts the target;
only VHL does. Using ElonginB alone as "the ligase" produced exactly the
21.9 Å-type non-contact seen. **Confirmed the manifest itself was correct
throughout** (it already listed all real ligase chains via DBREF, as
established back in the original benchmark-curation work) — this was a
bug in downstream analysis code that mishandled that correct data, not a
data-curation error.

**Fixed**: chain count of exactly 6 means halve (two copies); any other
count means use all listed chains (one complete assembly). Verified
directly: all 15 structures now show real target-ligase contact
(2.4-3.7 Å minimum atom distance).

**Corrected multi-structure benchmark, re-run in full:**

| Structure | Percentile (corrected) | Percentile (buggy) | Changed? |
|---|---|---|---|
| 5T35 | 4.9% | 4.9% | no (unaffected — 6-chain case) |
| 8FY0 | 3.0% | 4.3% | minor (unaffected chain count) |
| 6BN7 | 6.4% | 6.4% | no |
| 6BOY | 8.2% | 8.2% | no |
| 8FY2 | 10.3% | 10.9% | minor |
| 8FY1 | 16.1% | 12.0% | minor |
| 5HXB | 13.8% | 14.8% | minor |
| 8BDS | **36.0%** | 26.7% | **YES — moved from tractable to random** |
| 6HAX | 52.9% | 52.9% | no |
| 5FQD | 56.0% | 55.7% | no (glue, excluded from classification) |
| 7KHH | **72.5%** | 55.5% | **YES — worse, still random** |
| 8BEB | **74.3%** | 56.6% | **YES — worse, still random** |
| 6HR2 | 58.9% | 58.9% | no |
| 6HAY | 64.0% | 64.0% | no |

**Corrected classification (13 structures, excluding 5FQD glue):
TRACTABLE = 7 (5HXB, 5T35, 6BN7, 6BOY, 8FY0, 8FY1, 8FY2); RANDOM = 6
(6HAX, 6HAY, 6HR2, 7KHH, 8BDS, 8BEB).**

**The core Part 14 finding survives**: TRIAD is still not uniformly at a
hard ceiling — a genuine majority (7 of 13) of real, independent PROTAC
structures show good discrimination. This was not an artifact of the bug.

**What does NOT survive, and must be explicitly retracted**: the entire
8BDS-vs-8BEB "cleanest matched pair" investigation from Part 14/15 — the
comparison of nearly-identical reach distance, identical pool size, and
opposite outcomes — was built entirely on 8BDS's corrupted (non-contacting)
"native pose." With the bug fixed, 8BDS is no longer tractable at all; it
sits in the same "random" group as 8BEB. There was never a meaningful
contrast to explain. Any conclusions drawn from that comparison in this
document's earlier text are void and should not be cited.

**A sobering, honest note on process**: this bug went undetected through
the ENTIRE construction of the multi-structure benchmark (Part 14),
survived a cross-platform verification (Part 15) that checked numerical
reproducibility but not physical validity, and was only caught because a
downstream, unrelated check (shape channel showing exactly zero) looked
suspicious enough to chase to its root cause rather than being explained
away. The lesson: a percentile number can be perfectly reproducible across
machines and still be computed on physically nonsensical input. Numerical
stability and physical correctness are different properties, and this
project's test suite checked one without the other for a real stretch of
this investigation. This is now fixed and locked in as a permanent
regression (`test_multi_structure_discrimination.py`'s corrected
`_select_ligase_chains` and updated baseline).

**Re-confirmed cross-platform** (second independent run, Apple Silicon
Mac): all 14 structures' percentile classifications matched exactly, with
2 (5FQD, 8BDS) showing slightly larger n_valid drift (6.03%, 3.09%) than
the prior round's tolerance allowed — expected, since both have small
candidate pools (~120-370) where the same absolute floating-point wobble
produces a proportionally larger percentage swing than in large pools
(8FY0's ~19,000). Tolerance widened to 7% to reflect this properly.
Critically, neither structure's tractable/random classification moved.
The corrected 7/13 finding is now verified on two independent machines,
twice each.

---

## Part 17 — The real determinant found: tractability is a fixed property of the (ligase, target) pair, not of the specific molecule

**With the corrected, verified data, a much cleaner pattern emerged than
any of the single-feature hypotheses tested in Part 14.** Grouping the 13
PROTAC structures (5FQD glue excluded) by their (ligase, target)
combination:

| Ligase + Target | Structures | Classification |
|---|---|---|
| CRBN + BRD4-BD1 | 6BN7, 6BOY | TRACTABLE (2/2) |
| CRBN + GSPT1 | 5HXB | TRACTABLE (1/1) |
| VHL + BRD4-BD2 | 5T35 | TRACTABLE (1/1) |
| VHL + BCL-xL / BCL-2 | 8FY0, 8FY1, 8FY2 | TRACTABLE (3/3) |
| VHL + BRD4-BD1 | 7KHH, 8BDS, 8BEB | RANDOM (3/3) |
| VHL + SMARCA2 / SMARCA4 | 6HAX, 6HAY, 6HR2 | RANDOM (3/3) |

**Every single (ligase, target) combination is internally 100% consistent**
— every structure sharing a combination gets the same classification,
regardless of which specific degrader molecule, linker length, or reach
distance is involved. This directly rules out linker/molecule-level
properties as the primary determinant (already suggested by Part 14's
failed single-feature checks, now confirmed more directly): reach distance
does NOT separate the groups (5T35 at 10.3 A is tractable; 7KHH at 9.5 A,
nearly identical, is random) — but (ligase, target) IDENTITY perfectly
does.

**The most striking sub-finding**: VHL is not uniformly good or bad. It
succeeds with BRD4's *second* bromodomain (BD2) and with the BCL-2/BCL-xL
family, but fails with BRD4's *first* bromodomain (BD1) and with SMARCA2/4
— every time, regardless of which specific molecule was used. This means
the determinant is not "which ligase" in isolation, but something about
the specific geometric/electrostatic relationship between VHL's surface
and each particular target domain's surface near the ternary interface.

**Working hypothesis, not yet confirmed**: some target surfaces present a
more distinctive (less "generically shape-compatible") local topology or
charge pattern at the region engaged in the ternary interface, making the
true native orientation genuinely stand out from decoys via shape and
electrostatics; others present a more generic-looking surface in that
region, where many alternative orientations look comparably plausible by
the same metrics — a property of the target protein's local surface
geometry, fixed regardless of which small molecule is used to engage it.

**Next concrete step**: directly compare the receptor-side geometry
between a tractable and a random case sharing the SAME ligase but
different target (e.g. VHL+BRD4-BD2 [tractable] vs. VHL+BRD4-BD1
[random] — same ligase, closely related target domains from the same
parent protein, an even cleaner natural experiment than anything found so
far) — specifically the local surface curvature and charge distribution
in the region contacted by the ligand, not just aggregate reach/linker
statistics.

---

## Part 18 — A promising single-feature hypothesis tested and honestly ruled out; the real conclusion is pairwise, not single-partner

**Following Part 17's lead, target charged-residue fraction was checked
first** (motivated by the earlier finding that the shape channel
contributes essentially zero signal within the reach-constrained search —
Part 16 — meaning electrostatics is likely the dominant real
discriminator, so a target with a more electrostatically distinctive
surface should plausibly discriminate better). On a 4-point subset (5T35
vs. the three VHL+BRD4-BD1 cases), this looked compelling: 38% charged
residues (tractable) vs. 26-28% (random).

**Tested against all 13 structures, this does NOT hold up — correlation
0.083, essentially none.** Two decisive counter-examples: SMARCA2/4 (VHL,
RANDOM group) have the HIGHEST charged-residue fraction in the entire
dataset (0.37-0.42), and 6BN7/6BOY (CRBN+BRD4-BD1, TRACTABLE) have nearly
identical charged fraction (0.276-0.282) to 8BDS/7KHH/8BEB (VHL+BRD4-BD1,
RANDOM) — the SAME target, the SAME charge level, opposite outcomes,
determined purely by which ligase is paired with it.

**This last comparison is the real insight, more valuable than the ruled-
out hypothesis itself**: since the identical target (BRD4-BD1) gives
opposite results depending only on which ligase engages it, tractability
cannot be a property of either partner measured in isolation (not target
charge, not target size, not ligase identity alone — all tested and
ruled out across Parts 14-18). **It must be a property of the specific,
mutual, pairwise complementarity between that exact ligase's surface and
that exact target's surface** — consistent with basic protein-protein
recognition principles (specific recognition is inherently a property of
the interface between two partners, not of either partner alone), but
harder to reduce to a simple scalar feature of either side independently.

**Honest state of this investigation**: the WHAT (tractability is
perfectly predicted by (ligase, target) pair identity, Part 17) is now
solid, cross-validated, and well-supported. The WHY (what specific
geometric/electrostatic property of the pairwise interface determines it)
remains a genuine open question after five single-feature hypotheses
tested and ruled out (reach distance, linker length, ligase identity
alone, resolution, target charge fraction). Answering it properly would
likely require characterizing the actual native interface's shape and
charge complementarity directly (e.g. a real Sc-style shape
complementarity statistic computed ONLY over the true contact patch, or
the actual electrostatic potential correlation specifically at the native
interface rather than aggregate whole-protein statistics) rather than any
further whole-protein or whole-ligand summary statistic — a substantial
enough undertaking to be a genuine next-session task, not a quick follow-
up check.

---

## Part 19 — A sixth feature ruled out; disciplined stopping point for scalar-feature hunting

**Native buried surface area** (real BSA, computed via the already-
validated Shrake-Rupley SASA module, at the true crystallographic native
pose for each structure — not an aggregate whole-protein statistic, but
the actual measured interface size) was checked as a more targeted
follow-up to Part 18. Correlation with percentile: **-0.207, weak, with a
clear counter-example** (8FY0: BSA=148 A^2, 3.0th percentile — among the
smallest interfaces in the set, yet the single best-discriminated
structure; 8BEB: BSA=171 A^2, 73.7th percentile — larger interface, far
worse discrimination). Ruled out as an explanation.

**Six single-feature hypotheses have now been tested with equal rigor and
found insufficient**: reach distance, linker path length, ligase identity
alone, crystal resolution, target charged-residue fraction, and native
buried surface area. This is a disciplined stopping point for this
specific approach (checking one more whole-molecule scalar statistic
against percentile), not a dead end for the investigation as a whole. The
pattern across all six negative results is consistent and informative:
**no summary statistic of either partner, or even of the whole interface
as a single number, explains tractability — the answer almost certainly
lives in the DETAILED SPATIAL PATTERN of the true contact patch** (which
specific atoms touch, how tightly the specific shapes nest, whether the
electrostatic potential varies sharply or gently across that specific
patch) rather than any scalar reduction of it. This is exactly the
scope of a genuine Sc-style shape-complementarity calculation (Lawrence &
Colman 1993 — surface normal vectors and local curvature matching at the
actual contact patch), which remains the correctly-scoped next step,
flagged as substantial rather than attempted as a rushed seventh scalar
check likely to show the same pattern as the first six.

---

## Part 20 — The real Sc statistic built, validated, and tested: the strongest correlation found, in a genuinely counterintuitive direction

**Implemented Lawrence & Colman's (1993, *J. Mol. Biol.* 234:946-950) Sc
shape complementarity statistic** (`triad/scoring/shape_complementarity.py`)
— the actual field-standard measure of how well two surfaces nest at their
true contact patch, reusing the already-validated Shrake-Rupley
accessible-point logic from `triad.scoring.sasa` to generate real surface
points with outward normal vectors, rather than any further whole-molecule
scalar summary.

**A real memory bug found and fixed before this touched the full
dataset**: the first version generated full-chain surface points for
every atom before filtering to the interface, causing an out-of-memory
kill partway through the 13-structure run on real data. Fixed by
pre-filtering each body to only atoms within a generous margin of the
other body BEFORE the expensive surface-point generation step — verified
via a regression test that the optimization produces bit-identical
results to the unfiltered version on the synthetic validation case
(0.160, matched to 3 decimal places).

**Validated on synthetic geometry before touching real data**: a ball
nested in a concave cup (good fit) scored higher (Sc=0.160) than the same
ball touching the cup's rim tangentially (poor fit, Sc=-0.249) — confirms
the statistic discriminates fit quality in the physically correct
direction, though the absolute magnitude is not perfectly calibrated (a
stated limitation, not hidden).

**Result on the real 13-structure benchmark — the strongest correlation
found across all seven features tested to date:**

| Structure | Native Sc | Percentile |
|---|---|---|
| 8FY0 | -0.020 | 3.0% |
| 5T35 | 0.205 | 4.9% |
| 6BN7 | 0.363 | 6.7% |
| 6BOY | 0.304 | 8.3% |
| 8FY2 | 0.106 | 10.2% |
| 5HXB | 0.363 | 13.8% |
| 8FY1 | 0.374 | 16.2% |
| 8BDS | 0.301 | 37.6% |
| 6HAX | 0.302 | 52.5% |
| 6HR2 | 0.354 | 59.7% |
| 6HAY | 0.363 | 64.2% |
| 7KHH | 0.386 | 73.2% |
| 8BEB | 0.417 | 73.7% |

**Correlation: +0.567** (moderate-to-strong, n=13) — but in a
**counterintuitive direction**: HIGHER native shape complementarity
correlates with WORSE algorithmic discrimination, not better. The two
extremes make this vivid: 8FY0 has the lowest (even slightly negative) Sc
of the whole set yet the single best discrimination; 8BEB has the highest
Sc yet the worst.

**A candidate explanation, stated as a hypothesis, not a confirmed
mechanism**: a very smoothly-nested (high-Sc) interface may sit on a
broad, gently-curved surface region where many nearby alternative
orientations achieve comparably good geometric fit — making the specific
true orientation hard to distinguish from its neighbors by shape alone. A
lower-Sc interface may reflect a more idiosyncratic, specific contact
that few alternative orientations could replicate even approximately,
making the true pose easier to pick out (particularly via electrostatics,
already suspected as the dominant real discriminator per Part 16) even
though its raw geometric fit is less clean. This would mean shape
complementarity and dockability are not the same thing, and may even be
mildly anti-correlated for this specific problem class — a genuinely
interesting, non-obvious result if it holds up to further scrutiny.

**Honest caveats**: n=13 is still a small sample for a correlation
coefficient; the mechanism above is a hypothesis motivated by the data,
not independently confirmed; and this is the SEVENTH feature tested, so
some caution about multiple-comparisons is warranted (with enough
features tried, one moderate correlation is not automatically decisive).
That said, it is the clear standout among everything tried, is built on
a real, validated, standard structural biology algorithm rather than an
ad hoc statistic, and offers a specific, mechanistically plausible
explanation rather than an unexplained pattern — a genuinely promising
lead for continued investigation, appropriately labeled as a lead rather
than a settled conclusion.

---

## Part 21 — The honest statistical test: Sc's correlation does NOT survive multiple-comparisons correction

**Part 20 flagged the multiple-comparisons concern in words; this section
actually computes it, which is the difference between a caveat and a real
check.**

Pearson correlation: r=0.567, **p=0.0432** (n=13) — only marginally
significant at an uncorrected alpha of 0.05. Spearman rank correlation
(arguably more appropriate here, since Sc's absolute scale isn't precisely
calibrated and only the ordering should be trusted): rho=0.669,
**p=0.0125** — stronger, but still not decisive on its own.

**Applying a Bonferroni correction for the 7 features actually tested**
(reach distance, linker length, ligase identity, resolution, target
charge fraction, native BSA, native Sc) gives a required significance
threshold of **p<0.0071**. Neither the Pearson result (0.0432) nor the
Spearman result (0.0125) clears this bar.

**Honest conclusion: Sc's correlation with tractability, while the
strongest and most mechanistically interesting result of everything
tested in this investigation, does NOT meet a rigorous statistical bar
once the fact that seven different hypotheses were tried is properly
accounted for.** It remains the best available lead — a real, validated,
standard algorithm, a specific plausible mechanism, and the largest effect
size observed — but it should be described as exactly that (a lead worth
pursuing with more data) and not as a confirmed determinant of
tractability. Presenting it as confirmed would repeat, at the level of
statistics rather than chemistry or physics, the exact kind of
overclaiming this project's discipline has been built around catching
and correcting throughout (Parts 13's overstated single-structure
generalization, Part 16's uncaught chain-selection bug — both errors of
believing a result before subjecting it to the check that would have
caught it).

**What would actually resolve this**: more independent (ligase, target)
pairs than the field currently has solved structures for — this
investigation has already used essentially all suitable public PROTAC/
molecular-glue ternary structures available (the 15-structure benchmark
itself). Meaningfully more statistical power on this specific question
would require either new structures being solved by the field, or a
fundamentally different validation strategy (e.g. testing the Sc
hypothesis's mechanism directly — does a high-Sc native interface really
have more comparably-scoring nearby alternative orientations than a
low-Sc one? — rather than accumulating more single data points on
tractability alone).

**That direct mechanism test was run**, using data already computed: for
each structure, the fraction of valid candidates scoring within 5% of the
top score (a proxy for "how many near-equivalent alternatives exist"),
checked against Sc — a genuinely different dependent variable than
percentile, not a re-test of the same relationship. Result: correlation
+0.484, same direction as the mechanism predicts (higher Sc → more
comparably-scoring alternatives), providing independent, if still
modest and non-decisive, support (8FY1 is a visible counter-example: high
Sc, 0.374, but almost no near-top alternatives, 0.0001). This is
consistent with — but does not conclusively prove — the proposed
mechanism, and is exactly the right shape for an honest interim status:
a real, independently-collected piece of corroborating evidence, still
short of the statistical bar needed to call this settled.

---

## Part 22 — Full end-to-end dense rotation search retried with every fix in place: identical result, and the real reason finally diagnosed correctly

**With rotation-sampling redundancy fixed (Part 9), exhaustive occupancy-
overlap clash filtering built and validated (Part 10), and the chain-
selection bug found and fixed (Part 16), the full 3,600-rotation dense
search (~9.4 degree resolution, matching the tolerance window measured in
Part 9) was re-run on 5T35 end-to-end.**

**Result: 69.45 Å — bit-for-bit identical to Part 9's original,
pre-fix attempt.** This exact match across two meaningfully different
pipeline versions is itself informative: it means neither the rotation-
redundancy fix nor the clash-filtering fix changed the actual outcome at
all. The bottleneck lies elsewhere.

**First hypothesis tried, and found wrong on direct check**: that shape
contributes zero throughout the reach-constrained region regardless of
rotation (extrapolating from Part 16's finding that it's zero at the
NATIVE rotation specifically), making the whole multi-rotation search
electrostatics-only. Checked directly across several different rotations:
shape is NOT uniformly zero — at rotations other than native, C_shape
within the reach-sphere reaches values up to ~990, far from zero. The
hypothesis was wrong, and is recorded as such rather than quietly
replaced with the next guess.

**The actual mechanism, verified this time by direct inspection**: the
reach-sphere constraint only fixes WHERE the ligase's pivot point (its
warhead-attachment atom) sits — it says nothing about which direction the
REST of the ligase's bulk extends from that pivot. At the true native
rotation, the ligase body correctly points away from that immediate local
neighborhood, toward the real, larger-scale interface elsewhere (the
actual protein-protein contact extends well beyond a single reach-sphere
shell). At various WRONG rotations, the ligase's bulk can coincidentally
swing INTO that same local neighborhood around the pivot, and rack up
substantial (but spurious) shape "reward" from the target's real surface
shell nearby — reward that has nothing to do with correctly recreating
the actual interface, since it's driven by a geometric coincidence of one
local patch, not genuine complementary packing of the whole contact
surface.

**This is a real, structural limitation of the reach-sphere-plus-local-
shape-channel design, not a scoring-weight problem.** The shape channel,
as implemented, samples complementarity only in a thin, local, pivot-
centered shell — it has no way to "see" whether the rest of the ligase
body, extending outward from that pivot in a given rotation, is heading
toward a genuine complementary docking surface or away from it. A
correct fix would need the shape (and ideally electrostatic) channels to
meaningfully sample complementarity over the ligase's FULL bulk relative
to the target, not just near the reach-constrained pivot — a real
architectural change to the correlation channels, not a parameter tweak.

**Where this leaves the project, precisely**: the discrimination-quality
finding from Parts 8, 14, and 16-21 (native ranks well among valid
candidates AT THE CORRECT ROTATION, for a genuine majority of real
structures) remains fully valid and unaffected by this finding — that
was always tested with rotation held correct, deliberately isolating
translational discrimination. What this section adds is a precise,
verified diagnosis of why the FULL, autonomous, all-rotations-included
search still can't find that correct neighborhood on its own: not
insufficient rotational density (Part 9's original hypothesis, itself
correct as far as it went), not a remaining clash-filtering gap (Part
10's fix, real and necessary but insufficient alone), but a genuine
architectural blind spot in what the shape channel can perceive — local
pivot-shell complementarity, not whole-body correct orientation. Fixing
this is a legitimate, well-scoped Phase 4 undertaking, not a quick patch.

---

## Part 19 — A sixth hypothesis, using real interface geometry directly, also ruled out

**Took on the substantial step flagged at the end of Part 18**: computed
the actual native interface's buried surface area using the already-
validated Shrake-Rupley BSA calculation (`triad.scoring.sasa`), applied
directly to each structure's real target/ligase contact — not a whole-
protein proxy, the genuine native contact geometry itself.

**Result: correlation -0.207 — still weak, and clearly non-monotonic.**
There is a suggestive visual pattern (BCL-2/BCL-xL cases cluster at low
BSA, 100-149 A^2, and are tractable; CRBN cases and 5HXB cluster at high
BSA, 351-395 A^2, and are tractable; most RANDOM cases cluster in a
middle range, 170-240 A^2) — but 5T35, the cleanest and most-studied
TRACTABLE case all session, sits at 235.6 A^2, squarely inside the
"random" cluster's range. This is a genuine counter-example, not an
outlier to explain away, and it's enough to prevent treating the
low/high-vs-middle pattern as a real finding.

**Six hypotheses have now been tested with equal rigor and honesty across
Parts 14-19: reach distance, linker path length, ligase identity alone,
crystal resolution, target charged-residue fraction, and native interface
buried surface area. None cleanly explains tractability.** The only
predictor that has held up with 100% consistency, across all 13
structures, is the (ligase, target) PAIR identity itself (Part 17) — a
categorical, not continuous, feature.

**This is worth stating plainly rather than continuing to fish for a
seventh quick feature**: whatever determines tractability is very likely
not capturable by any single scalar summary statistic of either partner
or their aggregate contact area. The most likely remaining candidates,
each substantially more involved than anything tried so far:
1. A genuine multi-atom shape complementarity statistic (true Sc,
   Lawrence & Colman 1993 — needs local surface normal vectors and
   point-by-point matching across the interface, not a bulk area number)
2. Electrostatic potential correlation specifically restricted to the
   native contact patch (not whole-protein charge composition)
3. A property that isn't geometric at all — e.g. how much the SPECIFIC
   evolved/engineered interface deviates from what a generic rigid-body
   docking search would consider "good," which by definition requires
   comparing many decoys' properties against native's, not summarizing
   native alone (closer to what the actual FFT search already computes,
   suggesting the real signal may already be present in the full score
   distribution shape, not extractable as a single pre-computed feature
   at all)

Given six honest negative results in a row on pre-computed single
features, (3) is the most promising remaining direction: it reframes the
question from "what property predicts tractability" to "what does the
SHAPE of each structure's full score distribution (not just native's rank)
look like," which is directly computable from data already generated in
Part 14's benchmark without needing any new feature engineering.

---

## Part 20 — A strong statistical proxy found (z-score), but honestly distinguished from a mechanistic explanation

**Direction (3) from Part 19 was tested immediately, using already-
computed data.** For each structure, native's z-score within its own
valid-candidate pool (`(native_score - pool_mean) / pool_std`) was
computed and checked against percentile rank.

**Result: correlation -0.935 — by far the strongest relationship found in
this entire investigation (Parts 14-20).**

| Structure | Native z-score | Percentile |
|---|---|---|
| 8FY0 | 2.98 | 3.0% |
| 5T35 | 2.06 | 4.9% |
| 6BN7 | 1.88 | 6.7% |
| 6BOY | 1.69 | 8.3% |
| 8FY2 | 1.13 | 10.2% |
| 5HXB | 1.09 | 13.8% |
| 8FY1 | 0.69 | 16.2% |
| 8BDS | 0.31 | 37.6% |
| 6HAX | -0.19 | 52.5% |
| 6HR2 | -0.38 | 59.7% |
| 6HAY | -0.50 | 64.2% |
| 7KHH | -0.67 | 73.2% |
| 8BEB | -0.83 | 73.7% |

**Stated honestly, before this is mistaken for more than it is**: this is
NOT a mechanistic explanation. A z-score and a percentile rank are
closely related statistical descriptions of the same underlying quantity
(how far above the pool's mean does native's score sit) — a strong
correlation between them mostly confirms the score distributions are
regular and well-behaved enough for z-score to serve as a valid
continuous stand-in, not that some new causal factor has been discovered.
It does NOT explain why native's score is exceptional for some (ligase,
target) pairs and not others — that mechanistic question, the actual "why"
from Part 18-19, remains open.

**What this genuinely IS valuable for**: a cheap, continuous, single-
number proxy for tractability that doesn't require the expensive full
ranking against thousands of candidates. `native_z_score > ~1.0` reliably
predicts TRACTABLE; `< ~0.3` reliably predicts RANDOM in this dataset,
with a clean separation and no overlap at the current sample size (8BDS
at 0.31 is the closest call, and it is genuinely the most borderline
RANDOM case by full-ranking percentile too, at 37.6% — consistent, not
contradictory). This is a legitimate, useful methodological result:
future work on new structures could use this z-score as a fast screening
step before committing to a full rotational search.

**Where this leaves the investigation, honestly**: the WHAT is now
twice-confirmed (pair-identity in Part 17, z-score proxy in Part 20). The
mechanistic WHY remains open after six ruled-out single-feature hypotheses
(Parts 14-19). This is a legitimate place to pause this specific thread —
not because the question is unanswerable, but because answering it
properly needs real interface-level shape/electrostatic complementarity
machinery (Part 19's remaining options 1-2) that would be a substantial,
deliberate next build, not a quick check.
