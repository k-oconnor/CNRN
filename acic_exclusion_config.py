"""
Pre-specified ACIC 2018 exclusion rule for reproducibility (reviewer request).

This criterion is fixed before viewing results. Use the same constant in
finalize_paper_artifacts and any ACIC batch scripts so that "with excluded"
vs "without excluded" reporting is consistent.

Reference: Paper revision plan (ICML); REPRODUCIBILITY.md.
"""

# Dataset bases (e.g. hash prefix or identifier) excluded due to pre-specified criterion:
# "Extreme heavy-tailed outcomes" (e.g. outcome distribution makes estimation unstable).
ACIC_PRESPECIFIED_EXCLUDED_BASES = frozenset({
    "ae021576c9b248b5942fbc7c1a0539df",
})

ACIC_EXCLUSION_REASON = (
    "Pre-specified exclusion: extreme heavy-tailed outcomes (criterion fixed before viewing results)."
)
