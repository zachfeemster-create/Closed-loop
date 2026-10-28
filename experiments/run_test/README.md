# Run a Test (the bench)

Every test from the Oct 23-27 2026 run logs, rebuilt as plain functions in [`closedloop/lab.py`](../../closedloop/lab.py) so anybody can rerun them on their own data. No AI, no account, no cost. The same tests sit behind the app's **Run a Test** page (`pages/5_Run_a_test.py`).

| Test | What it asks | Where it stood in the runs |
|---|---|---|
| `real_vs_fakes` | Can a six-measure toolkit tell your series from noise, walks, spiky walks and shuffled steps? Passes only on the hard ones (walks and shuffled). | Beat noise, near chance vs walks (World Bank) |
| `leftovers` | After the shared path is removed, which leftovers still move together? | Leftovers tie through the economy |
| `turnback` | More mid-curve reversals than each series' own momentum explains? | Survivor (countries, names yes; states, rivers no) |
| `orbit` | Do remainders cycle more regularly than momentum-matched fakes? Spiral out or in? | Survivor (both World Bank sets, per-indicator baseline) |
| `interlock` | Do orbits lock together across series? A quarter turn apart (zipper)? | Interlock survivor; zipper offset missed |
| `step_flip` | The corner Y: when a series sits in its usual corner, do others land on the flip corners next step, already parked there? | Lead on countries and calm-year states; missed on weather, names, rivers |
| `vortex` | Shorter stays in the rare corners than walks give? | Countries and states yes; names and rivers no |
| `leaderboard` | Do top-list gaps bunch or keep a rhythm beyond a smooth tail? | Donkey Kong top 50: inside the fakes |
| `three_point` | Amplitude and phase of a three-value cycle vs 500,000 random triplets (Koide / Brannen form) | Charged leptons: root two and 2/9, 0 of 500k (a rediscovery) |
| `dose_response` | Hill fit and the leftover under the ceiling, with a bootstrap range | Dopamine: ~23%, range too wide to pin |

Two tests from the logs already had homes: **who answers whom** (`closedloop/games.py`, Game theory page) and **raw vs modeled** (`demo/modeling_fingerprint`, Modeling fingerprint page).

## Command line

```
pip install -r requirements.txt
python experiments/run_test/run_test.py turnback my_data.csv
python experiments/run_test/run_test.py orbit wb.csv --groups --fakes 300
python experiments/run_test/run_test.py three_point --values 0.51099895 105.6583755 1776.86
```

Series tests want one column per series with time down the rows. `--groups` takes out each group's own shared path, where the group is the part of the column name before `|` or `:` (for example `SP.DYN.LE00.IN|Chad`). That is how the World Bank runs were done.

## Rules every test follows

- Fakes are matched to each real series (drift, step size, step memory, or its own shuffled steps).
- The pass rule is fixed in the code before any data goes in; the app also locks the plan (settings + data fingerprint) before the run button opens.
- A pass is a pattern, not a finding. Ladder: pattern → survivor → sturdy survivor → lead (fresh data) → finding (someone else reproduces it).

## Checks run before release (Oct 28 2026)

- False passes on pure random walks at the 95th-percentile bar, 20 seeds: real vs fakes 0/20, interlock 0/20, step flip 0/20, turn-back 2/20, vortex 2/20, orbit 3/20.
- Charged lepton masses reproduce A = 1.41420, delta = 0.22223, Koide Q = 0.66666, 0 of 500,000 fakes.
- Dopamine inputs reproduce Emax 77.2, ED50 10.9 mg, n 1.82, leftover 22.8.
- A fresh World Bank pull (life expectancy, infant mortality, urban share, log population, trade; 2000-2022 every 2 years) gives turn-back, interlock (no zipper offset), step flip and vortex passes, and orbit steadiness with an outward spiral under the per-indicator baseline, in line with the logs.
