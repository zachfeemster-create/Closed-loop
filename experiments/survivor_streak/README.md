# Historical survivor persistence — real matched-null experiment

This is an executable historical control, not an AI-generated research proposal.
Implementation: matched_null.py. Source: all 21 original dated nightly files,
whose SHA-256 values appear in the output artifact.

## First controlled result (2026-11-18; 200 fakes; 900 swaps per fake; seed 20261118)

- Absolute longest streak = 21 days, with degree-matched null upper-tail p=1.0.
  **This is a trivial statistic**: keys present on every recorded day remain
  present in every null because we explicitly preserve their total frequency.
  It is NOT evidence of predictive pattern.
- Excluding keys appearing fewer than 2 days or more than 19 of 21 days:
  14 eligible keys. Longest intermittent run = 16 consecutive days, versus
  matched-null median 13. Upper-tail p≈0.14428. **Not a strong signal.**
- Sum of adjacent-day key overlaps = 141, versus matched-null median 134;
  upper-tail p≈0.00995 in this exploratory 200-fake test. **Potential lead
  only**, not validated discovery.

### Why this is not a certified result

The null exactly preserves:
1. Each original date's number of distinct survivor identities.
2. Each unique survivor key's total appearances.
3. The source's duplicate-deduplicated survivor identity, combining
   dataset name, lens, and EQ.

Random double-edge swaps deliberately erase temporal ordering. But they do
not reproduce original study selection, day-specific feed gaps, underlying
time-series autocorrelation, post-hoc model tuning, or all original fake tests.

Three statistics were examined; one apparently small p-value is NOT a
pre-registered result. A 21-day sample, optional data sources and repeated
survivors can create instability. Monte Carlo samples are from a Markov
mixing procedure, not guaranteed independent stationary draws.

### Immediate follow-up checks

- Re-run with larger fake counts, independent RNG seeds, and mixing diagnostics.
- Split the historical interval into early/late windows *before* interpreting
  the signal; separately re-check late periods as hold-out (no hindsight).
- Stratify or restrict to feeds present on all dates.
- Use the original momentum-matched fakes on the underlying time series,
  not only this post-selected survivor table.
- If any transfer to Quantum Arcane is considered, transfer only the
  control/test methodology. No quantum cost or equivalence claim follows
  from temporal source persistence.

The project provides the full distribution rather than just a p-value
in GitHub Actions artifacts. No original data or production code is altered.
