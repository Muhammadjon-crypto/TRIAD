"""
Validates triad.correlation.preflight's classification logic against the
11 known (native_score, RMSD) pairs from manifest Part 29, before trusting
it on new structures.
"""
from triad.correlation.preflight import HIGH_CONFIDENCE_TRACTABLE, HIGH_CONFIDENCE_INTRACTABLE

# (native_score, actual_RMSD) from Part 29 -- ground truth already measured
KNOWN_RESULTS = {
    "8BDS": (333.6, 4.17), "5T35": (489.2, 4.59), "7KHH": (226.7, 26.46),
    "8BEB": (39.6, 26.92), "6BOY": (-24.6, 54.47), "6HR2": (108.8, 66.35),
    "6BN7": (-133.6, 75.65), "6HAX": (86.6, 77.49), "6HAY": (11.0, 87.72),
    "5FQD": (107.2, 135.99), "5HXB": (-404.3, 155.70),
}


def test_high_confidence_tractable_bucket_is_never_wrong_on_known_data():
    """Every structure ABOVE the high-confidence-tractable threshold in
    the known set should have a genuinely good RMSD (<30A).
    """
    for pdb_id, (score, rmsd) in KNOWN_RESULTS.items():
        if score >= HIGH_CONFIDENCE_TRACTABLE:
            assert rmsd < 30.0, f"{pdb_id}: predicted tractable but RMSD={rmsd}"


def test_high_confidence_intractable_bucket_is_never_wrong_on_known_data():
    """Every structure BELOW the high-confidence-intractable threshold in
    the known set should have a genuinely poor RMSD (>50A).
    """
    for pdb_id, (score, rmsd) in KNOWN_RESULTS.items():
        if score < HIGH_CONFIDENCE_INTRACTABLE:
            assert rmsd > 50.0, f"{pdb_id}: predicted intractable but RMSD={rmsd}"


if __name__ == "__main__":
    test_high_confidence_tractable_bucket_is_never_wrong_on_known_data()
    test_high_confidence_intractable_bucket_is_never_wrong_on_known_data()
    print("Preflight classification logic confirmed self-consistent against known data.")
