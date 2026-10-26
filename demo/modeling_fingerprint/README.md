# Demo: the modeling fingerprint

A one-command test of the Closed Loop method on public data. It's the locked Run 009, rebuilt so anyone can rerun it.

**The question:** official statistics are often not measured directly. Agencies run them through estimation and smoothing models. Can the Closed Loop's quarter-turn lenses tell an agency's **smoothed** numbers from the **raw** counts of the same thing, country by country, in a way that goes **beyond** plain smoothness (momentum)?

**The test:**
1. Train a small scorer on WHO modeled estimates (obesity, hypertension, anaemia and others) against fakes that copy each series' step size and momentum. Then freeze it.
2. Take 56 countries where both versions of infant deaths exist for 2000-2019: raw registered counts (WHO Mortality Database) and the UN IGME smoothed estimate.
3. Score both. Then subtract how much of the gap momentum-matched fakes produce on their own (50 draws per country).
4. Pass if what's left is above zero across countries (p < 0.01), against two kinds of fake, with a clean fake-vs-fake frame check.

## Run it

```
pip install -r requirements.txt
python fetch_data.py     # about 60 MB of public data, one time
python run_demo.py       # about 30 seconds
```

## Expected output (data pulled 25 Oct 2026)

```
   smoothed scores higher than raw in 48/56 countries (median gap 0.148)
   gap from momentum alone:      0.084
   EXCESS beyond momentum (A):   0.054  in 38/56 countries, p = 0.0017   (pass: p < 0.01)
   EXCESS beyond smoothing (B):  0.086  in 42/56 countries, p = 9.6e-06   (pass: p < 0.01)
   frame check (fake vs fake):   p = 0.89   (clean: p > 0.05)

VERDICT: PASS - the fingerprint sees modeling beyond momentum
```

The WHO and UNICEF sources update over time, so later downloads may give slightly different numbers. That's part of the test: if it's real, it should keep passing.

## What it means, and what it doesn't

- **It is:** a detector for modeled vs measured data. About a third of the raw-vs-smoothed gap is beyond momentum, so the fingerprint is part smoothness and part shape.
- **It isn't:** a signal about the world behind the numbers. The same lenses found nothing beyond fakes in rough measured data (CO2 and energy), and pure noise run through a fitting model shows the fingerprint too.
- **Ladder rung:** candidate finding. It becomes a finding when someone else reproduces it. If you run it, positive or negative, please share the result.

## Files

- `fetch_data.py`: downloads WHO GHO estimates, WHO Mortality Database raw files and UN IGME estimates.
- `fingerprint.py`: lens features (quarter turn: level vs step, per frequency band), momentum-matched fakes and the scorer.
- `run_demo.py`: the locked test, with the verdict printed.

The author keeps a full run history (Runs 001-009: plans locked before each run, results, misses and voids). The method is described in the paper: https://doi.org/10.5281/zenodo.23638245

Code: GPL-3.0 (see the repository LICENSE). Zach Feemster, Nightshift Solutions, LLC.
