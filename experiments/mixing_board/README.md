# Mixing-board test (paper v1.1, 25 Oct 2026)

Splits each World Bank series into four frequency channels on the brain-wave octave ladder (delta 1-2, theta 3-4, alpha 5-7, beta 8-11 cycles per 23 years), reads each channel through the quarter turn, scores each against fakes (noise, walks, spiky walks, shuffled steps), then mixes the channels back with wet/dry levels set on half the countries and scored on the other half.

Run in order:

1. `python pull.py`  (fresh World Bank API pull, 10 indicators, 2000-2022 -> wb_raw.csv)
2. `python board.py` (per-channel separation, wet/dry mix)
3. `python check.py` (measured-only series, which theta feature drives it)

Seed 60. Results reported in The Closed Loop v1.1, section "Mixing-board test".

Needs: numpy, pandas, scikit-learn, requests.
