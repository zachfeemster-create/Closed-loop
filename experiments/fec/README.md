# FEC outside spending, 2016–2024

Rebuilds the outside-spending tables the **Game theory** page offers as real data, and re-runs the race-level counter-spending check.

```
python experiments/fec/build_fec.py     # writes samples/fec_fronts_2016_2024.csv and samples/fec_weekly_2016_2024.csv
python experiments/fec/lead_check.py    # race-level "who answers whom" across 245 races
```

Downloads go to `.fec_cache/` (about 1 GB of FEC and FiveThirtyEight files). Pass another folder as the first argument to put them elsewhere.

## Source and why

Independent expenditures come from FEC's processed committee-to-candidate file (`pas2`, transaction types 24E support / 24A oppose). They're limited to general election, House and Senate, with memo lines dropped and the latest filing per transaction kept. Party, office and district come from the FEC candidate master. D-side means money that supports a Democrat or opposes a Republican; R-side is the reverse.

FEC's plain `independent_expenditure_YYYY.csv` is **not** used for totals. It is missing nearly all Senate Leadership Fund spending (2022: $3.2M listed vs $246M reported). It also lacks the party and candidate ID on most Congressional Leadership Fund rows, which undercounts the R side by about half. It is still used for dissemination dates in `lead_check.py`.

Race values: `closeness` is the final two-party margin from FEC compiled results (2016–2022). `fc_margin_close` and `fc_prob_close` are FiveThirtyEight's forecast margin and win-probability closeness, averaged from August 1 to Election Day (House 2018–2022, Senate 2016–2022), from their [Checking Our Work data](https://github.com/fivethirtyeight/checking-our-work-data).

## Results as of Oct 27, 2026

| Check | Result | Standing |
|---|---|---|
| Totals | D-side and R-side outside money are close to even every cycle (2024: $907M vs $917M). The old "D outspends R in most races" came from the missing R-side money. | fact about the source |
| Race-level answering (245 races, >$1M per side) | D surge → R next week +0.164, R surge → D next week +0.096, both beat 100% of within-race shuffles. Gap +0.068, 95% range +0.036 to +0.100. | lead (payment/release dates, not airing) |
| Blotto vs equilibrium | Both sides concentrate far from equilibrium every cycle: about half of races in play left empty, 45–58% of the bigger side's money past the equilibrium cap. Expected, since races aren't worth the same. | description |
| Chasing each other beyond the forecast | 2018: +0.29, past fakes. 2016 Senate (22 races): yes, small sample. 2020: +0.10, no. 2022: no. Against final margins it looked like 3 of 4 cycles; most of that was both sides following the forecasts. | survivor, one cycle |
| National weekly answering | Nothing passes in any cycle, and the detection limit shows 44 weeks is too short to catch it. | below the floor |

Limits: payment and release dates, not airing dates. Final margins are partly caused by the spending. 538 coverage stops at 2022.
