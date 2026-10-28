"""Panic before drops: is a signal (news panic, fear gauge, anything) already unusual BEFORE big market drops?

An event study with Closed Loop fakes: the same drop dates, with the signal slid to random points in time.
Used by the Run a Test page and the Ask page.
"""
import numpy as np

from closedloop.lab import _r, _row


def _drop_events(price: np.ndarray, drop: float, within: int, gap: int):
    """Start rows of big drops: the price falls at least `drop` (e.g. 0.10) within the next `within` rows, measured
    from that row's level. Events are kept at least `gap` rows apart so windows don't overlap."""
    lp = np.log(price)
    n = lp.size
    ev = []
    t = 0
    while t < n - 1:
        end = min(n, t + within + 1)
        if lp[t + 1:end].size and lp[t + 1:end].min() - lp[t] <= np.log(1 - drop):
            ev.append(t)
            t += gap
        else:
            t += 1
    return np.array(ev, dtype=int)


def _rolling_z(x: np.ndarray, base: int):
    """Each day's reading against its own previous `base` days (past only): how unusual it is for that time."""
    import pandas as pd
    s = pd.Series(np.log(np.where(x > 0, x, np.nan)) if np.all(x[np.isfinite(x)] > 0) else x)
    m = s.shift(1).rolling(base, min_periods=base // 2).mean()
    sd = s.shift(1).rolling(base, min_periods=base // 2).std()
    return ((s - m) / sd.replace(0, np.nan)).to_numpy()


def _window_means(z: np.ndarray, ev: np.ndarray, a: int, b: int):
    """Average of z over rows ev+a .. ev+b (inclusive) for each event, ignoring gaps; NaN when out of range."""
    n = z.size
    zz = np.where(np.isfinite(z), z, 0.0)
    cnt = np.isfinite(z).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(zz)])
    cc = np.concatenate([[0.0], np.cumsum(cnt)])
    lo, hi = ev + a, ev + b + 1
    ok = (lo >= 0) & (hi <= n)
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    tot, num = cs[hi] - cs[lo], cc[hi] - cc[lo]
    out = np.where(ok & (num > 0), tot / np.maximum(num, 1), np.nan)
    return out


def lead_up(price, signal, dates=None, names=("market", "signal"), drop_pct: float = 10.0, within: int = 60,
            before: int = 63, during: int = 21, after: int = 63, base: int = 250, n_fakes: int = 2000,
            seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """Is a signal (news panic, fear gauge, anything) already unusual BEFORE big drops, beyond chance?
    Drops: the market falls drop_pct within `within` rows. The signal is read against its own previous year
    (past only), then averaged over the `before` rows ahead of each drop, the `during` rows of it and the `after`
    rows that follow. Fakes: the same drop dates with the signal slid to random points in time (keeps both
    series' own rhythm, breaks only their timing together). Rows are trading days for daily market data."""
    p = np.asarray(price, float)
    x = np.asarray(signal, float)
    ok = np.isfinite(p) & (p > 0)
    p, x = p[ok], x[ok]
    dts = None if dates is None else np.asarray(dates)[ok]
    if p.size < base + before + after + 50:
        raise ValueError(f"needs a longer history (has {p.size} rows; wants {base + before + after + 50}+).")
    rng = np.random.default_rng(seed)
    z = _rolling_z(x, base)
    ev = _drop_events(p, drop_pct / 100.0, within, gap=max(within, before) + during)
    ev = ev[(ev - before >= base) & (ev + during + after < p.size)]
    if ev.size < 3:
        raise ValueError(f"only {ev.size} drops of {drop_pct:g}% in this stretch; lower the drop size or use a longer history.")
    wins = {"before": (-before, -1), "during": (0, during - 1), "after": (during, during + after - 1)}
    real = {k: float(np.nanmean(_window_means(z, ev, a, b))) for k, (a, b) in wins.items()}
    rise = float(np.nanmean(_window_means(z, ev, -10, -1) - _window_means(z, ev, -before, -before + 9)))
    n = z.size
    fs = {k: np.empty(n_fakes) for k in list(wins) + ["rise"]}
    lo_shift = before + during + after
    for i in range(n_fakes):
        zs = np.roll(z, int(rng.integers(lo_shift, n - lo_shift)))
        for k, (a, b) in wins.items():
            fs[k][i] = np.nanmean(_window_means(zs, ev, a, b))
        fs["rise"][i] = np.nanmean(_window_means(zs, ev, -10, -1) - _window_means(zs, ev, -before, -before + 9))
    rows = [_row(f"{names[1]} in the {before} rows BEFORE drops (vs its own past year)", real["before"], fs["before"], "both", bar_pct),
            _row(f"{names[1]} climbing into the drop (last 10 rows minus first 10 of the lead-up)", rise, fs["rise"], "both", bar_pct),
            _row(f"{names[1]} DURING drops (first {during} rows)", real["during"], fs["during"], "both", bar_pct,
                 note="expected to react; not a lead"),
            _row(f"{names[1]} AFTER drops (next {after} rows)", real["after"], fs["after"], "both", bar_pct)]
    lead = rows[0]["passed"] or rows[1]["passed"]
    each = _window_means(z, ev, -before, -1)
    hit = float(np.nanmean(each > 0.5))
    events = []
    for k, e in enumerate(ev[:40]):
        fall = float(np.exp(np.log(p[e + 1:min(p.size, e + within + 1)]).min() - np.log(p[e])) - 1)
        events.append(dict(start=str(dts[e])[:10] if dts is not None else int(e), fall=f"{fall:.0%}",
                           before_z=_r(each[k], 2), during_z=_r(_window_means(z, ev[k:k + 1], 0, during - 1)[0], 2)))
    direction = "higher" if real["before"] > 0 else "lower"
    return dict(test="lead_up", question=f"Is {names[1]} already unusual before {names[0]} drops of {drop_pct:g}%+?",
                rows=rows, passed=lead,
                verdict=(f"PASS: {names[1]} runs {direction} than chance in the lead-up to drops" if rows[0]["passed"]
                         else ("PASS: it climbs into the drops more than chance" if rows[1]["passed"]
                               else f"FAIL: {names[1]} in the lead-up is inside the random-timing fakes"))
                + f" ({ev.size} drops)",
                notes=[f"{ev.size} drops of {drop_pct:g}% or more within {within} rows. {hit:.0%} of them had the signal "
                       "more than half a standard deviation above its own past year in the lead-up.",
                       "During and after rows show reaction (news and fear spike once prices fall); only the BEFORE "
                       "and climbing rows can say anything about warning.",
                       "Drops cluster in a few crises, so a handful of events carry a lot of weight; a pass is a pattern. "
                       "The next check is fresh data: a different market, or a stretch held out before you looked."],
                details=dict(rows=int(p.size), drops=int(ev.size), drop_pct=drop_pct, within=within, before=before,
                             during=during, after=after, base=base, fakes=n_fakes),
                events=events)
