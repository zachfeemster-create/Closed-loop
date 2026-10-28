# Closed Loop

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23654654.svg)](https://doi.org/10.5281/zenodo.23654654)

A method for telling real patterns from chance: look for patterns, then try hard to kill them with matched fakes, and keep only what survives. It borrows from surveying, where you check closure before trusting a traverse.

- **Paper:** [PAPER.md](PAPER.md) · https://doi.org/10.5281/zenodo.23638245
- **Software (this repo, all versions):** https://doi.org/10.5281/zenodo.23654654
- **Demo (start here):** [demo/modeling_fingerprint](demo/modeling_fingerprint). One command reruns a locked test on public WHO and UN data and prints the verdict.
- **App:** `app.py` and `closedloop/`
- **Lens rotation page:** fixed lenses, data rotates through them, an EQ pass with matched fakes, and Max in the back jinxing anything that passes (`pages/2_Lens_rotation.py`, `closedloop/rotation.py`)
- **Run a Test page:** every test from the run logs as a menu (real vs fakes, leftovers, turn-back, orbit, interlock, the corner Y, vortex, leaderboard gaps, three-point phase, dose-response). Pick data, lock the plan, run. Also from the command line ([experiments/run_test](experiments/run_test), `closedloop/lab.py`)
- **Autorun:** nightly and weekly jobs that pull public feeds and run the rotation on their own, no AI ([autorun/](autorun))
- **Experiments:** [experiments/](experiments)

Code: GPL-3.0 (see LICENSE). Paper: CC BY 4.0. Zach Feemster, Nightshift Solutions, LLC.
