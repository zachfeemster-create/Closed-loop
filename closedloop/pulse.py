"""Pulse sweep: plant oscillating pulses of known period and height into fakes of YOUR series, and map what each lens
can and can't hear. Then read your real series through the same lenses, so a miss can be stated as
"nothing bigger than X at period P".

Lenses (each scored against the same unplanted fakes, so 'caught' means beyond chance):
  regular   strongest cycle anywhere, each period judged against what the fakes put there (all periods scanned,
            so the look-elsewhere is paid)
  keystone  power at ONE period named in advance (the question sets the keystone; no scanning, so sharper)
  quarter   orbit steadiness on (level, step): does the remainder turn regularly
  turnback  share of points in the turn-back corners
"""
from typing import Optional

import numpy as np

from closedloop.lab import _orbit_stats, _quad_detrend, _zcols, corners_median, fakes_ar1

LENSES = ("regular", "keystone", "quarter", "turnback")


def _spectrum(x: np.ndarray):
    """Power by period for a detrended series. Returns (periods, power), longest periods first dropped below 3 rows."""
    r = _quad_detrend(x[:, None])[:, 0]
    r = r - r.mean()
    n = r.size
    f = np.fft.rfftfreq(n)[1:]
    p = np.abs(np.fft.rfft(r))[1:] ** 2
    per = 1.0 / f
    keep = (per >= 3) & (per <= n / 2)
    return per[keep], p[keep]


def _stats(x: np.ndarray, period: Optional[float], ref=None):
    """ref: the fakes' average power at each period (share of total). When given, the regular lens is the biggest
    ratio of real to expected power across periods, so slow walk-like swings don't drown every cycle."""
    per, p = _spectrum(x)
    tot = p.sum() + 1e-12
    share = p / tot
    if ref is not None and ref.size == share.size:
        ratio = share / ref
        regular = float(ratio.max())
        peak = float(per[np.argmax(ratio)])
    else:
        regular = float(share.max())
        peak = float(per[np.argmax(share)])
    if period is not None:
        band = (per >= period / 1.15) & (per <= period * 1.15)
        keystone = float(p[band].sum() / tot) if band.any() else 0.0
    else:
        keystone = np.nan
    rm = _zcols(_quad_detrend(_zcols(x[:, None])))
    quarter = _orbit_stats(rm)[0]
    turn = float(np.isin(corners_median(x[:, None]), (2, 5)).mean())
    return dict(regular=regular, keystone=keystone, quarter=quarter, turnback=turn, peak_period=peak)


def _plant(f: np.ndarray, period: float, height: float, shape: str, step_sd: float, rng):
    """Add a wave `height` typical steps tall (peak to middle) to one fake. shape: 'steady' runs the whole length;
    'burst' is a wave packet covering about a fifth of the series at a random spot."""
    n = f.size
    t = np.arange(n)
    wave = height * step_sd * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    if shape == "burst":
        mid, width = rng.uniform(0.15, 0.85) * n, max(2 * period, 0.2 * n) / 2.355
        wave = wave * np.exp(-0.5 * ((t - mid) / width) ** 2)
    return f + wave


def _ref(X: np.ndarray, rng, n: int = 400):
    """Average spectrum share of momentum-matched fakes, period by period (built from their own fresh fakes)."""
    sh = []
    for f in fakes_ar1(X, n, rng)[:, :, 0]:
        _, p = _spectrum(f)
        sh.append(p / (p.sum() + 1e-12))
    return np.mean(sh, axis=0)


def sweep(x, periods=None, heights=None, shape: str = "steady", n_null: int = 300, trials: int = 40,
          bar_pct: float = 95.0, seed: int = 20261029) -> dict:
    """Catch rate of each lens for each planted (period, height), on momentum-matched fakes of x.
    Height is in units of x's typical step (step standard deviation)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 60:
        raise ValueError(f"needs at least 60 rows (has {n}).")
    rng = np.random.default_rng(seed)
    step_sd = float(np.diff(x).std())
    if periods is None:
        periods = np.unique(np.round(np.geomspace(6, n / 4, 6)))
    if heights is None:
        heights = [0.0, 0.5, 1, 2, 4, 8]
    X = x[:, None]
    ref = _ref(X, rng)
    null = [_stats(f[:, 0], None, ref) for f in fakes_ar1(X, n_null, rng)]
    bars = {k: float(np.percentile([s[k] for s in null], bar_pct)) for k in ("regular", "quarter", "turnback")}
    key_bar = {}
    nullF = fakes_ar1(X, n_null, rng)[:, :, 0]
    for P in periods:
        key_bar[P] = float(np.percentile([_stats(f, P)["keystone"] for f in nullF], bar_pct))
    grid = {k: np.zeros((len(periods), len(heights))) for k in LENSES}
    for i, P in enumerate(periods):
        fakes = fakes_ar1(X, trials * len(heights), rng)[:, :, 0]
        for j, h in enumerate(heights):
            for k in range(trials):
                f = _plant(fakes[j * trials + k], P, h, shape, step_sd, rng)
                s = _stats(f, P, ref)
                grid["regular"][i, j] += s["regular"] > bars["regular"]
                grid["keystone"][i, j] += s["keystone"] > key_bar[P]
                grid["quarter"][i, j] += s["quarter"] > bars["quarter"]
                grid["turnback"][i, j] += s["turnback"] > bars["turnback"]
    for k in grid:
        grid[k] /= trials
    floor = {}
    for k in LENSES:
        floor[k] = {}
        for i, P in enumerate(periods):
            hit = [heights[j] for j in range(len(heights)) if heights[j] > 0 and grid[k][i, j] >= 0.8]
            floor[k][float(P)] = min(hit) if hit else None
    return dict(periods=[float(p) for p in periods], heights=list(map(float, heights)), shape=shape,
                catch=grid, floor=floor, false_pass={k: float(grid[k][:, 0].mean()) for k in LENSES},
                bars=bars, key_bars={float(p): v for p, v in key_bar.items()}, step_sd=step_sd, rows=n)


def read_real(x, named_period: Optional[float] = None, n_null: int = 1000, bar_pct: float = 95.0,
              seed: int = 20261029) -> dict:
    """Your real series through the same lenses, against the same kind of fakes."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    ref = _ref(x[:, None], rng)
    real = _stats(x, named_period, ref)
    null = [_stats(f[:, 0], named_period, ref) for f in fakes_ar1(x[:, None], n_null, rng)]
    out = {}
    for k in LENSES:
        if k == "keystone" and named_period is None:
            continue
        fs = np.array([s[k] for s in null])
        out[k] = dict(real=round(real[k], 4), bar=round(float(np.percentile(fs, bar_pct)), 4),
                      percentile=round(float((fs < real[k]).mean() * 100), 1),
                      passed=bool(real[k] > np.percentile(fs, bar_pct)))
    return dict(lenses=out, peak_period=real["peak_period"], named_period=named_period)


def pulse_test(x, named_period: Optional[float] = None, shape: str = "steady", periods=None, heights=None,
               bar_pct: float = 95.0, seed: int = 20261029, name: str = "series") -> dict:
    """One run in the bench's format: read the real series through the four lenses, then sweep planted pulses into
    its fakes to give each lens's floor (smallest wave caught 80% of the time) at each period."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    note = []
    if x.size > 5000:
        k = int(np.ceil(x.size / 5000))
        x = x[::k]
        note.append(f"Long series: every {k}th row kept ({x.size:,} rows) so it runs quickly.")
    n = x.size
    if periods is None:
        auto = [int(round(p)) for p in np.geomspace(6, n / 4, 5)]
        if named_period:
            auto = [p for p in auto if abs(p / named_period - 1) > 0.25] + [int(round(named_period))]
        periods = sorted(set(auto))
    periods = [p for p in periods if 3 <= p <= n / 2]
    rd = read_real(x, named_period, bar_pct=bar_pct, seed=seed)
    sw = sweep(x, periods=periods, heights=heights or [0, 1, 2, 4, 8, 16, 32], shape=shape, bar_pct=bar_pct,
               seed=seed + 1)
    labels = dict(regular="strongest cycle (scans every period)", keystone=f"named period ({named_period:g} rows)"
                  if named_period else "named period", quarter="quarter turn (orbit steadiness)",
                  turnback="turn-back share")
    rows = []
    for k in LENSES:
        if k not in rd["lenses"]:
            continue
        r = rd["lenses"][k]
        rows.append(dict(stat=labels[k], real=r["real"], bar=r["bar"], percentile=r["percentile"], passed=r["passed"],
                         note=(f"peak at {rd['peak_period']:.1f} rows" if k == "regular" else "")))
    table = []
    for k in LENSES:
        row = dict(lens=labels[k] if (k != "keystone" or named_period) else "named period (each column's period, if named)")
        for P in sw["periods"]:
            h = sw["floor"][k][P]
            row[f"{P:g} rows"] = "deaf" if h is None else f"{h:g}"
        table.append(row)
    # One lens decides, fixed before the run: the named period when one is given, otherwise the scanner.
    decider = rows[1] if named_period else rows[0]
    cyc = decider["passed"]
    if cyc:
        verdict = ("PASS: a cycle beyond the fakes at the named " + f"{named_period:g} rows" if named_period
                   else f"PASS: a cycle beyond the fakes (peak at {rd['peak_period']:.1f} rows)")
    else:
        verdict = "FAIL: no cycle beyond the fakes"
        if named_period and sw["floor"]["keystone"].get(float(named_period)) is not None:
            verdict += (f"; nothing taller than about {sw['floor']['keystone'][float(named_period)]:g} typical steps "
                        f"at {named_period:g} rows")
    return dict(test="pulse", question=f"Does {name} carry a cycle, and how small a wave could we have heard?",
                rows=rows, passed=bool(cyc), verdict=verdict,
                notes=note + [
                    ("Decided by the named period alone (set before the run); the scanner is shown but doesn't count."
                     if named_period else "Decided by the scanner alone (no period named)."),
                    "Floor table: the smallest planted wave each lens caught 80% of the time, in units of the series' "
                    f"typical step ({sw['step_sd']:.4g}). 'deaf' = missed even at 32 steps. Shape: {shape}.",
                    "False alarms with nothing planted: " + ", ".join(f"{k} {v:.0%}" for k, v in sw["false_pass"].items())
                    + " (should sit near 5%).",
                    "Naming a period before you look (the keystone) is sharper than scanning, because scanning pays "
                    "for every period it checks. Name it from a reason (seasons, orbits, elections), not from a peek.",
                    "The quarter-turn and turn-back lenses read curve shape, not clean cycles; they're expected to be deaf here."],
                details=dict(rows=n, step_sd=sw["step_sd"], shape=shape, periods=sw["periods"]),
                floor_table=table)
