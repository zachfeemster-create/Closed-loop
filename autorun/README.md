# Autorun: the Closed Loop on its own

Two scheduled GitHub jobs run the lens rotation with no AI and no tokens. Results land in `autorun/results/`
and show on the app's **Lens rotation → Nightly & weekly** tab.

**Nightly** (`closed-loop-nightly.yml`, about 2:17 am Mountain): pulls every feed, rotates every series
through the fixed lenses (flat EQ first as the control, then the curve in `config.json`) against its own
momentum-matched fakes, and lets Max in the back jinx anything that passes. A survivor that keeps surviving on
new data night after night builds a streak; that's the "holds on fresh data" rung.

**Weekly** (`closed-loop-weekly.yml`, Sundays): runs the daily feeds against each other as one batch (lead,
bunching, drift) over a rolling 120-day window, after measuring and removing the shared weekday rhythm (reported
separately as "lens of time"). Then the week's luck pile gets its own pass at a strict 99.9 bar.

| Feed | What it serves | Key needed |
|---|---|---|
| FRED (UNRATE, INDPRO, CPIAUCSL, PAYEMS, HOUST, DGS10, T10Y2Y, DCOILWTICO, VIXCLS) | turn-backs, bunching | no |
| World Bank GDP-per-person growth, 15 countries | drift, bunching | no |
| WSPR daily spots per band (wspr.live), last 120 days | WSPR thread | no |
| FEC independent expenditures, last 60 days, for/against each party | answer-back, issue duels | `FEC_API_KEY` repo secret (free at api.data.gov/signup) |
| arXiv JUNO watcher | the locked Δm²21 bet | no |
| WHO infant-mortality watcher | modeling fingerprint rerun alerts | no |

A feed that fails is listed as skipped; the rest still run. Edit `config.json` to change the EQ curve, the
number of fakes or the bar. Run either job by hand from the repo's **Actions** tab → **Run workflow**.

## Turning the schedule on (one-time)

The two schedule files are in `autorun/github-workflows/`. GitHub only runs them from `.github/workflows/`, and
only the repo owner can put them there. For each file: open it on GitHub, tap the pencil (edit), and in the
file-name box change the path to `.github/workflows/<same name>.yml`, then commit.
