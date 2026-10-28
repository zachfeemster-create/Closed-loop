"""The test bench: every Closed Loop test from the run logs, rebuilt as plain functions anyone can run.

No AI anywhere. Each test takes data and settings, builds matched fakes, scores the real data against them,
and returns a verdict under a rule fixed before it runs. Rebuilt from the locked plans and run logs of
Oct 23-27 2026 (World Bank, US states, weather, baby names, rivers, particle masses, Donkey Kong, dopamine).

Panel tests take a 2-D array X with one column per series (rows = time, no gaps).
Every function returns a dict with: test, question, passed, rows (the scored numbers), notes, details.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- shared helpers


def _pct(fakes: np.ndarray, real: float) -> float:
    fakes = np.asarray(fakes, float)
    fakes = fakes[np.isfinite(fakes)]
    if fakes.size == 0 or not np.isfinite(real):
        return float("nan")
    return float(100.0 * (np.sum(fakes < real) + 0.5 * np.sum(fakes == real)) / fakes.size)


def _row(stat: str, real: float, fakes: np.ndarray, side: str, bar_pct: float, note: str = "") -> dict:
    """One scored line. side: 'high' (real must beat the top bar), 'low' (real must sit under the bottom bar),
    'both' (either tail, split the bar)."""
    f = np.asarray(fakes, float)
    f = f[np.isfinite(f)]
    if f.size == 0 or not np.isfinite(real):
        return dict(stat=stat, real=None, fakes_mid=None, bar=None, percentile=None, passed=False, note=note or "not scorable")
    if side == "high":
        bar = float(np.percentile(f, bar_pct))
        ok = real > bar
    elif side == "low":
        bar = float(np.percentile(f, 100 - bar_pct))
        ok = real < bar
    else:
        lo, hi = np.percentile(f, [(100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2])
        bar = [float(lo), float(hi)]
        ok = real < lo or real > hi
    sd = float(np.std(f))
    z = (real - float(np.mean(f))) / sd if sd > 0 else float("nan")
    return dict(stat=stat, real=_r(real), fakes_mid=_r(float(np.median(f))),
                bar=[_r(b) for b in bar] if isinstance(bar, list) else _r(bar),
                percentile=round(_pct(f, real), 1), z=_r(z, 2), passed=bool(ok), note=note)


def _r(v, d=4):
    try:
        v = float(v)
    except Exception:
        return v
    if not np.isfinite(v):
        return None
    return round(v, d) if abs(v) >= 1e-3 or v == 0 else float(f"{v:.3e}")


def _clean(X, groups=None):
    """Drop rows with gaps and flat columns. With groups, returns (X, groups) kept in step."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    keep = np.isfinite(X).all(axis=1)
    X = X[keep]
    ok = X.std(axis=0) > 0
    if groups is not None:
        return X[:, ok], [g for g, k in zip(groups, ok) if k]
    return X[:, ok]


def _zcols(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)


def _quad_detrend(X: np.ndarray) -> np.ndarray:
    """Remove each column's own straight-plus-bend (quadratic) trend."""
    T = X.shape[0]
    t = np.linspace(-1, 1, T)
    B = np.column_stack([np.ones(T), t, t ** 2])
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return X - B @ coef


def _baseline(Z: np.ndarray, groups=None) -> tuple[np.ndarray, float]:
    """Subtract the shared path: across all series, or within each group (e.g. each indicator) when groups
    is given. Groups with fewer than 3 members are left alone. Returns (leftover, share of variation removed)."""
    if groups is None:
        if Z.shape[1] < 3:
            return Z, 0.0
        out = Z - Z.mean(axis=1, keepdims=True)
    else:
        g = np.asarray(groups)
        out = Z.copy()
        for lab in np.unique(g):
            m = g == lab
            if m.sum() >= 3:
                out[:, m] = Z[:, m] - Z[:, m].mean(axis=1, keepdims=True)
    share = float(1 - np.var(out) / max(np.var(Z), 1e-12))
    return out, share


def remainder(X: np.ndarray, remove_shared: bool = True, groups=None) -> tuple[np.ndarray, float]:
    """Standardize, take out the shared baseline (the average path, across all series or within each group),
    then each series' own straight-plus-bend trend. Returns (remainder, share of variation that was baseline)."""
    Z = _zcols(X)
    share = 0.0
    if remove_shared:
        Z, share = _baseline(Z, groups)
    Rm = _quad_detrend(Z)
    return _zcols(Rm), share


def group_labels(names, seps="|:") -> list | None:
    """Group = the part of each series name before the first '|' or ':' (e.g. 'SP.DYN.LE00.IN|Chad').
    None when names carry no groups."""
    out = []
    for n in map(str, names):
        cut = min((n.find(c) for c in seps if c in n), default=-1)
        out.append(n[:cut] if cut > 0 else None)
    return out if all(o is not None for o in out) and len(set(out)) > 1 else None


def _parts(X: np.ndarray):
    """Level (detrended), step, bend — lined up on the same rows (first two rows lost)."""
    L = _quad_detrend(X)
    S = np.diff(X, axis=0)
    Bd = np.diff(S, axis=0)
    return L[2:], S[1:], Bd


def corners_median(X: np.ndarray) -> np.ndarray:
    """v2 corner code per point: each axis split at the series' own median. code = 4*level + 2*step + bend."""
    L, S, Bd = _parts(X)
    f = lambda A: (A > np.median(A, axis=0)).astype(int)
    return 4 * f(L) + 2 * f(S) + f(Bd)


def corners_sign(X: np.ndarray, funnel: bool = True, radius: float = 1.0) -> np.ndarray:
    """v1 corner code: signs of standardized (level, step, bend). Points inside the sphere get -1 when funnel is on."""
    L, S, Bd = _parts(X)
    zL, zS, zB = _zcols(L), _zcols(S), _zcols(Bd)
    C = 4 * (zL > 0) + 2 * (zS > 0) + (zB > 0)
    C = C.astype(int)
    if funnel:
        inside = np.sqrt(zL ** 2 + zS ** 2 + zB ** 2) < radius
        C[inside] = -1
    return C


# ---------------------------------------------------------------- fake makers (matched to each real series)


def fakes_ar1(X: np.ndarray, nsim: int, rng) -> np.ndarray:
    """Momentum-matched fakes: each series' steps refit as AR(1) (same drift, size and step memory),
    rebuilt from the same start. Returns (nsim, T, N)."""
    from scipy.signal import lfilter
    T, N = X.shape
    D = np.diff(X, axis=0)
    mu = D.mean(0)
    Dc = D - mu
    phi = np.array([np.corrcoef(Dc[1:, j], Dc[:-1, j])[0, 1] if Dc[:, j].std() > 0 else 0.0 for j in range(N)])
    phi = np.clip(np.nan_to_num(phi), -0.95, 0.95)
    sd = Dc.std(0) * np.sqrt(1 - phi ** 2)
    e = rng.normal(size=(nsim, T - 1, N)) * sd
    e[:, 0] = e[:, 0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))      # start at the stationary spread
    steps = np.empty_like(e)
    for j in range(N):
        steps[:, :, j] = lfilter([1.0], [1.0, -phi[j]], e[:, :, j], axis=1)
    steps += mu
    return np.concatenate([np.broadcast_to(X[0], (nsim, 1, N)), X[0] + np.cumsum(steps, axis=1)], axis=1)


def _batches(maker, X: np.ndarray, n: int, rng, budget: float = 4e6):
    """Yield fakes one at a time, built in small batches so memory stays flat however long the data is."""
    T, N = X.shape
    size = int(max(1, min(n, budget // max(T * N, 1))))
    done = 0
    while done < n:
        k = min(size, n - done)
        F = maker(X, k, rng)
        for i in range(k):
            yield F[i]
        done += k
        del F


def fakes_walk(X: np.ndarray, nsim: int, rng, spiky: bool = False) -> np.ndarray:
    """Plain random walks (or spiky walks with heavy-tailed t steps, df 2.5) with each series' drift and step size."""
    T, N = X.shape
    D = np.diff(X, axis=0)
    mu, sd = D.mean(0), D.std(0)
    if spiky:
        e = rng.standard_t(2.5, size=(nsim, T - 1, N)) / np.sqrt(2.5 / 0.5)
    else:
        e = rng.normal(size=(nsim, T - 1, N))
    steps = mu + sd * e
    return np.concatenate([np.broadcast_to(X[0], (nsim, 1, N)), X[0] + np.cumsum(steps, axis=1)], axis=1)


def fakes_noise(X: np.ndarray, nsim: int, rng) -> np.ndarray:
    T, N = X.shape
    return X.mean(0) + X.std(0) * rng.normal(size=(nsim, T, N))


def fakes_shuffled(X: np.ndarray, nsim: int, rng) -> np.ndarray:
    """Each series' own real steps in a shuffled order."""
    T, N = X.shape
    D = np.diff(X, axis=0)
    idx = np.argsort(rng.random((nsim, T - 1, N)), axis=1)
    steps = np.take_along_axis(np.broadcast_to(D, (nsim, T - 1, N)), idx, axis=1)
    return np.concatenate([np.broadcast_to(X[0], (nsim, 1, N)), X[0] + np.cumsum(steps, axis=1)], axis=1)


FAKE_KINDS = {"noise": fakes_noise, "walk": fakes_walk,
              "spiky walk": lambda X, n, r: fakes_walk(X, n, r, spiky=True), "shuffled steps": fakes_shuffled}


def _roll_cols(A: np.ndarray, rng, min_shift: int = 1) -> np.ndarray:
    """Slide each column by its own random circular offset (keeps each series' runs, breaks timing across series)."""
    T, N = A.shape
    k = rng.integers(min_shift, max(min_shift + 1, T - min_shift + 1), size=N)
    rows = (np.arange(T)[:, None] - k[None, :]) % T
    return np.take_along_axis(A, rows, axis=0)


# ================================================================ 1. real vs fakes


def _features(X: np.ndarray) -> np.ndarray:
    """Six-number standard toolkit per series: step memory, step tails, step skew, bend ratio, biggest shock, orbit steadiness."""
    D = np.diff(X, axis=0)
    Dc = D - D.mean(0)
    sd = np.where(Dc.std(0) > 0, Dc.std(0), 1.0)
    lag1 = (Dc[1:] * Dc[:-1]).mean(0) / sd ** 2
    kurt = (Dc ** 4).mean(0) / sd ** 4 - 3
    skew = (Dc ** 3).mean(0) / sd ** 3
    B = np.diff(D, axis=0)
    bend = B.std(0) / sd
    shock = np.abs(Dc).max(0) / sd
    Rm = _zcols(_quad_detrend(_zcols(X)))
    th = np.arctan2(np.diff(Rm, axis=0), Rm[1:])
    dth = np.angle(np.exp(1j * np.diff(th, axis=0)))
    steady = np.abs(np.exp(1j * dth).mean(0))
    return np.column_stack([lag1, kurt, skew, bend, shock, steady])


def _cv_auc(F_real, F_fake, rng, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    n = F_real.shape[0]
    grp = rng.permutation(n) % folds
    scores = np.zeros(2 * n)
    y = np.r_[np.ones(n), np.zeros(n)]
    F = np.nan_to_num(np.vstack([F_real, F_fake]))
    g2 = np.r_[grp, grp]
    for k in range(folds):
        tr, te = g2 != k, g2 == k
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(max_iter=500).fit(sc.transform(F[tr]), y[tr])
        scores[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
    return float(roc_auc_score(y, scores))


def real_vs_fakes(X, n_reps: int = 5, n_chance: int = 40, seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """Can a standard six-measure toolkit tell your series from matched fakes? One fake per real series,
    5-fold cross-validated, grouped so a series and its own fake never split. Separation = |AUC - 0.5| x 2
    (0 = can't tell at all, 1 = perfect). Chance band = the same with labels shuffled inside each pair."""
    X = _clean(X)
    T, N = X.shape
    split_note = []
    if N < 10:
        # Few series but long ones: cut each into windows so there are enough pieces to score.
        k = min(math.ceil(30 / max(N, 1)), T // 20)
        if N * k < 10:
            raise ValueError(f"needs 10+ series, or fewer series long enough to cut into 10+ windows of 20 rows "
                             f"(has {N} series, {T} rows).")
        w = T // k
        X = np.column_stack([X[i * w:(i + 1) * w, j] for j in range(N) for i in range(k)])
        X = _clean(X)
        split_note.append(f"Only {N} series, so each was cut into {k} windows of {w} rows ({X.shape[1]} pieces scored). "
                          "Pieces of the same series share its character, so read a pass a notch more cautiously.")
        T, N = X.shape
    if T < 20:
        raise ValueError(f"needs at least 20 rows (has {T}).")
    rng = np.random.default_rng(seed)
    F_real = _features(X)
    rows, details = [], {}
    for kind, maker in FAKE_KINDS.items():
        seps, chance = [], []
        for _ in range(n_reps):
            Fk = _features(maker(X, 1, rng)[0])
            seps.append(abs(_cv_auc(F_real, Fk, rng) - 0.5) * 2)
        for _ in range(n_chance):
            swap = rng.random(N) < 0.5
            A = np.where(swap[:, None], Fk, F_real)
            Bm = np.where(swap[:, None], F_real, Fk)
            chance.append(abs(_cv_auc(A, Bm, rng) - 0.5) * 2)
        real = float(np.mean(seps))
        rows.append(_row(f"vs {kind}", real, np.array(chance), "high", bar_pct,
                         note="separation (0 = can't tell, 1 = perfect)"))
        details[kind] = dict(each_rep=[round(s, 3) for s in seps])
    n_pass = sum(r["passed"] for r in rows)
    hard = rows[1]["passed"] and rows[3]["passed"]
    return dict(test="real_vs_fakes", question="Is there anything in these series that matched fakes can't fake?",
                rows=rows, passed=hard,
                verdict=f"{n_pass} of {len(rows)} fake kinds told apart; "
                + ("PASS: beats both hard ones (walks and shuffled steps)" if hard else
                   "FAIL: doesn't beat both hard ones (walks and shuffled steps)"),
                notes=split_note + ["Rule: pass only if it beats both hard fakes (walks and shuffled steps). Noise and spiky walks are "
                       "reported but easy: plain random walks themselves get told apart from those.",
                       "In the World Bank runs the toolkit beat noise (~0.27) but sat near chance against walks."],
                details=dict(series=N, rows=T, reps=n_reps, **details))


# ================================================================ 2. shared baseline and leftovers


def _step_link(Rm: np.ndarray, lags: int):
    """Strongest |correlation| between two leftovers' steps at any lag from -lags to +lags, plus the signed value
    and the lag where it peaks (positive lag = the second series follows the first)."""
    D = np.diff(Rm, axis=0)
    D = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1.0)
    T = D.shape[0]
    best = np.zeros((D.shape[1], D.shape[1]))
    at = np.zeros_like(best)
    for L in range(-lags, lags + 1):
        A, B = (D[:T - L], D[L:]) if L >= 0 else (D[-L:], D[:T + L])
        Cm = (A.T @ B) / A.shape[0]
        better = np.abs(Cm) > np.abs(best)
        best = np.where(better, Cm, best)
        at = np.where(better, L, at)
    return np.abs(best), best, at


def leftovers(X, names=None, n_slides: int = 300, seed: int = 20261028, bar_pct: float = 95.0, top: int = 12,
              groups=None, max_series: int = 300, lags: int = 1, remove_shared: bool = False) -> dict:
    """Which series' leftovers move together beyond chance? Optionally take out the shared baseline first
    (off by default: on series of different shapes it manufactures links).
    Fakes: independent momentum-matched series run through the exact same steps (baseline removal included),
    so the drifting look-alike correlations that long trending series always have are in the fakes too.
    (n_slides is the number of fake sets; the name is kept from the earlier slide control.)"""
    X0 = np.asarray(X, float)
    names = list(names) if names is not None else [f"S{i + 1}" for i in range(X0.shape[1])]
    X, kept = _clean(X0, list(zip(names, groups if groups is not None else [None] * len(names))))
    names = [k[0] for k in kept]
    groups = [k[1] for k in kept] if groups is not None else None
    T, N = X.shape
    if N < 3 or T < 10:
        raise ValueError(f"needs at least 3 series and 10 rows (has {N}, {T}).")
    rng = np.random.default_rng(seed)
    capped = []
    pick = np.arange(N)
    if N > max_series:
        pick = np.sort(rng.choice(N, max_series, replace=False))
        capped.append(f"Random {max_series} of {N} series scored to keep it quick"
                      + (" (shared path taken from all of them)." if remove_shared else "."))
    Rm_all, share = remainder(X, remove_shared, groups)
    Rm = Rm_all[:, pick]
    names = [names[i] for i in pick]
    n = len(pick)
    absC, C, at = _step_link(Rm, lags)
    iu = np.where(~np.eye(n, dtype=bool)) if lags else np.triu_indices(n, 1)     # with lags, A->B and B->A differ
    real = absC[iu]
    bins = np.linspace(0, 1.0000001, 20001)
    hist = np.zeros(bins.size - 1)
    for f in _batches(fakes_ar1, X, n_slides, rng):
        Af, _, _ = _step_link(remainder(f, remove_shared, groups)[0][:, pick], lags)
        hist += np.histogram(np.clip(Af[iu], 0, 1), bins=bins)[0]
    tail = np.cumsum(hist[::-1])[::-1]                       # fakes at or above each bin, pooled over every pair
    total = hist.sum()
    idx = np.clip(np.searchsorted(bins, real, side="right") - 1, 0, hist.size - 1)
    p = (tail[idx] + 1) / (total + 1)
    m = p.size
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    alpha = (100 - bar_pct) / 100
    surv = q <= alpha
    show = np.argsort(p)[:top]
    def _lbl(k):
        a, b, L = names[iu[0][k]], names[iu[1][k]], int(at[iu][k])
        return f"{a} → {b} (next step)" if L > 0 else (f"{b} → {a} (next step)" if L < 0 else f"{a} ~ {b} (same step)")
    rows = [dict(stat=_lbl(k), real=_r(C[iu][k], 3), percentile=None,
                 p=_r(p[k], 4), q=_r(q[k], 4), passed=bool(surv[k]), note="link between the leftovers' steps") for k in show]
    return dict(test="leftovers", question="Which series move together step by step (or one a step ahead) beyond chance?",
                rows=rows, passed=bool(surv.any()),
                verdict=f"{int(surv.sum())} of {m} pairs survive (false-discovery control at {alpha:.0%})",
                notes=capped + [(f"Shared baseline = {share:.0%} of the variation (taken out before scoring)." if remove_shared
                                 else "Shared baseline left in, so links that run through everything count too.")
                                + f" Links are scored on the leftovers' steps, same step or up to {lags} step(s) apart, because "
                                "drifting levels fake strong-looking correlations on their own.",
                                *(["Taking out a shared path from series of different shapes ties every leftover to the "
                                   "busiest series (it fooled the check on unrelated series every time), so use it only on "
                                   "many series of the same kind, like one indicator across countries."] if remove_shared else []),
                                f"Fakes from every pair are pooled into one yardstick ({n_slides} fake sets × {m:,} pairs), "
                                "so a strong link can clear the many-pairs correction.",
                                "A surviving pair is a pattern, not a cause: shared world events tie a lot of series together."],
                details=dict(series=n, rows=T, fakes=n_slides, shared_share=round(share, 3), pairs=int(m)))


# ================================================================ 3. turn-back beyond momentum


def turnback(X, n_fakes: int = 300, seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """Do the series turn back mid-curve (step against both level and bend: corners 010 and 101) more often
    than their own momentum explains? Fakes: AR(1) steps matched to each series' drift, size and step memory."""
    X = _clean(X)
    T, N = X.shape
    if T < 10:
        raise ValueError(f"needs at least 10 rows (has {T}).")
    rng = np.random.default_rng(seed)
    rare = lambda C: float(np.isin(C, (2, 5)).mean())
    real = rare(corners_median(X))
    fs = np.array([rare(corners_median(f)) for f in _batches(fakes_ar1, X, n_fakes, rng)])
    walk = float(np.mean([rare(corners_median(f)) for f in _batches(fakes_walk, X, min(n_fakes, 100), rng)]))
    row = _row("share in turn-back corners (010 + 101)", real, fs, "high", bar_pct)
    return dict(test="turnback", question="Do these series turn back mid-curve more than their own momentum explains?",
                rows=[row], passed=row["passed"],
                verdict="PASS: more turn-backs than momentum explains" if row["passed"] else "FAIL: inside the momentum fakes",
                notes=[f"Even split would be 25%; plain random walks land about {walk:.1%} here, because the coding "
                       "itself makes these corners rare. That's why the fakes, not 25%, are the yardstick.",
                       "In the runs: countries and baby names passed, US states and rivers didn't (survivor level)."],
                details=dict(series=N, rows=T, fakes=n_fakes, walk_share=round(walk, 4)))


# ================================================================ 4. orbit steadiness and spiral


def _orbit_stats(Rm: np.ndarray):
    u, v = Rm[1:], np.diff(Rm, axis=0)
    th = np.arctan2(v, u)
    dth = np.angle(np.exp(1j * np.diff(th, axis=0)))
    steady = float(np.abs(np.exp(1j * dth).mean(0)).mean())
    oneway = float(np.maximum((dth > 0).mean(0), (dth < 0).mean(0)).mean())
    r = np.log(np.sqrt(u ** 2 + v ** 2) + 1e-9)
    t = np.arange(r.shape[0]) - (r.shape[0] - 1) / 2
    slope = float(((t[:, None] * (r - r.mean(0))).sum(0) / (t ** 2).sum()).mean())
    return steady, oneway, slope, th


def orbit(X, n_fakes: int = 300, seed: int = 20261028, bar_pct: float = 95.0, remove_shared: bool = True,
          groups=None) -> dict:
    """Look at each remainder as a point going around (level, step). Steadiness = how regular the turning is
    (1 = same turn every step). Spiral = whether the orbit drifts outward or inward over time.
    Fakes: AR(1) steps matched to each series, run through the same remainder steps."""
    X, groups = _clean(X, groups) if groups is not None else (_clean(X), None)
    T, N = X.shape
    if T < 10:
        raise ValueError(f"needs at least 10 rows (has {T}).")
    rng = np.random.default_rng(seed)
    Rm, share = remainder(X, remove_shared, groups)
    steady, oneway, slope, _ = _orbit_stats(Rm)
    fs = np.array([_orbit_stats(remainder(f, remove_shared, groups)[0])[:3]
                   for f in _batches(fakes_ar1, X, n_fakes, rng)])
    r1 = _row("orbit steadiness", steady, fs[:, 0], "high", bar_pct)
    r2 = _row("spiral (log-radius slope)", slope, fs[:, 2], "both", bar_pct,
              note="outward" if slope > 0 else "inward")
    r3 = _row("share of turns going one way", oneway, fs[:, 1], "high", bar_pct, note="mostly built into the view")
    return dict(test="orbit", question="Do the leftovers cycle more regularly than momentum explains, and do they spiral?",
                rows=[r1, r2, r3], passed=r1["passed"],
                verdict=("PASS: steadier orbits than the fakes" if r1["passed"] else "FAIL: orbit steadiness inside the fakes")
                + ("; spiral also outside the fakes" if r2["passed"] else "; no spiral beyond the fakes"),
                notes=["Primary rule: steadiness. Spiral reported alongside; the one-way share comes mostly from the view itself.",
                       "In the runs: steadiness passed on both World Bank sets with each indicator's own path taken out "
                       "(the 'within groups' baseline); the spiral passed on one of two."],
                details=dict(series=N, rows=T, fakes=n_fakes, shared_share=round(share, 3)))


# ================================================================ 5. interlock (and the zipper offset)


def _plv(th: np.ndarray):
    Z = np.exp(1j * th)
    M = (Z.conj().T @ Z) / th.shape[0]
    iu = np.triu_indices(th.shape[1], 1)
    lock = np.abs(M[iu])
    off = np.abs(np.sin(np.angle(M[iu])))
    return float(lock.mean()), float((lock * off).sum() / max(lock.sum(), 1e-12)), float((np.cos(np.angle(M[iu])) > 0.7).mean())


def interlock(X, n_fakes: int = 200, seed: int = 20261028, bar_pct: float = 95.0, max_series: int = 300,
              groups=None) -> dict:
    """Do the series' orbits lock together (keep a steady angle gap), and is the gap a quarter turn (the zipper)?
    Fakes: independent momentum-matched series run through the exact same steps (shared-baseline removal
    included, since taking out a shared path ties the leftovers together a little on its own)."""
    X, groups = _clean(X, groups) if groups is not None else (_clean(X), None)
    T, N = X.shape
    if N < 2 or T < 10:
        raise ValueError(f"needs at least 2 series and 10 rows (has {N}, {T}).")
    rng = np.random.default_rng(seed)
    note = []
    if N > max_series:
        pick = np.sort(rng.choice(N, max_series, replace=False))
        X = X[:, pick]
        groups = [groups[i] for i in pick] if groups is not None else None
        note.append(f"Random {max_series} of {N} series used to keep it quick.")
    shared = X.shape[1] >= 3
    th_of = lambda A: _orbit_stats(remainder(A, shared, groups)[0])[3]
    lock, off, instep = _plv(th_of(X))
    fs = np.array([_plv(th_of(f)) for f in _batches(fakes_ar1, X, n_fakes, rng)])
    r1 = _row("interlock (average lock strength)", lock, fs[:, 0], "high", bar_pct)
    r2 = _row("zipper offset (1 = quarter turn apart)", off, fs[:, 1], "high", bar_pct)
    r3 = _row("share of pairs locked in step", instep, fs[:, 2], "high", bar_pct)
    return dict(test="interlock", question="Do the orbits lock together, and are they offset a quarter turn like a zipper?",
                rows=[r1, r2, r3], passed=r1["passed"],
                verdict=("PASS: orbits interlock" if r1["passed"] else "FAIL: no interlock beyond the fakes")
                + ("; zipper offset passes too" if r2["passed"] else "; not a quarter turn apart"),
                notes=note + ["In the runs: interlock passed on both World Bank sets, but they lined up in step or opposite, "
                              "not a quarter turn apart (zipper offset missed). Those runs used time-slid angles as the "
                              "control; this version uses independent fakes, which is stricter."],
                details=dict(series=int(X.shape[1]), rows=T, fakes=n_fakes))


# ================================================================ 6. step flip, level+bend flip, parked receivers (the Y)


def _flip_stats(C: np.ndarray, blue: np.ndarray):
    T, N = C.shape
    K = np.zeros((T, 8))
    P = np.zeros((T - 1, 8))
    for k in range(8):
        I = C == k
        K[:, k] = I.sum(1)
        P[:, k] = (I[:-1] & I[1:]).sum(1)
    E = (C[:-1] == blue[None, :])
    out = []
    for mask in (0b010, 0b101):
        L = blue ^ mask
        nxt = K[1:, L] - (C[1:] == L[None, :])
        parked = P[:, L] - ((C[:-1] == L[None, :]) & (C[1:] == L[None, :]))
        ev = E.sum()
        hits = float((E * nxt).sum())
        rate = hits / max(ev * (N - 1), 1)
        park = float((E * parked).sum()) / max(hits, 1)
        out += [rate, park]
    return out


def step_flip(X, n_slides: int = 300, seed: int = 20261028, bar_pct: float = 95.0, funnel: bool = True,
              remove_shared: bool = True, groups=None) -> dict:
    """The corner Y. Each series' 'blue' = the corner it visits most. When a series sits in its blue, do OTHER
    series land one step later on (a) the step-flip corner (same level and bend, step reversed) and
    (b) the level+bend-flip corner, more than chance? And were the ones that land already parked there?
    Control: run-keeping slides (each series' own corner sequence slid by a random offset)."""
    X, groups = _clean(X, groups) if groups is not None else (_clean(X), None)
    T, N = X.shape
    if N < 3 or T < 10:
        raise ValueError(f"needs at least 3 series and 10 rows (has {N}, {T}).")
    rng = np.random.default_rng(seed)
    Z = _zcols(X)
    share = 0.0
    if remove_shared:
        Z, share = _baseline(Z, groups)
    C = corners_sign(Z, funnel=funnel)
    blue = np.array([np.bincount(C[:, j][C[:, j] >= 0], minlength=8).argmax() if (C[:, j] >= 0).any() else 0
                     for j in range(N)])
    real = _flip_stats(C, blue)
    fs = np.array([_flip_stats(_roll_cols(C, rng, 1), blue) for _ in range(n_slides)])
    labels = ["step flip lands (rate)", "step flip: receivers already parked",
              "level+bend flip lands (rate)", "level+bend flip: receivers already parked"]
    rows = [_row(labels[i], real[i], fs[:, i], "high", bar_pct) for i in range(4)]
    both = rows[0]["passed"] and rows[2]["passed"]
    return dict(test="step_flip", question="When a series sits in its usual corner, do others answer with the flip one step later?",
                rows=rows, passed=both,
                verdict="PASS: both arms of the Y beat the slides" if both else
                ("PARTIAL: one arm beats the slides" if rows[0]["passed"] or rows[2]["passed"] else "FAIL: neither arm beats the slides"),
                notes=[f"Shared baseline taken out first ({share:.0%} of the variation)." if remove_shared else "Shared baseline left in.",
                       "Caution from the runs: the two landing corners are rare by construction, so passes rest on few events. "
                       "It held on countries and calm-year US states, missed on weather, names and rivers."],
                details=dict(series=N, rows=T, slides=n_slides, funnel=funnel,
                             blues={str(k): int(v) for k, v in zip(*np.unique(blue, return_counts=True))}))


# ================================================================ 7. vortex short stays


def _stays(C: np.ndarray):
    rare_len, other_len = [], []
    for j in range(C.shape[1]):
        c = C[:, j]
        cut = np.flatnonzero(np.diff(c) != 0) + 1
        starts = np.r_[0, cut]
        lens = np.diff(np.r_[starts, c.size])
        isr = np.isin(c[starts], (2, 5))
        rare_len += list(lens[isr])
        other_len += list(lens[~isr])
    a, b = np.mean(rare_len) if rare_len else np.nan, np.mean(other_len) if other_len else np.nan
    return float(a / b) if b else float("nan")


def vortex(X, n_fakes: int = 200, seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """The empty-corner passage idea: are visits to the rare corners (010, 101) shorter than visits elsewhere,
    more so than random walks with each series' steps would give?"""
    X = _clean(X)
    T, N = X.shape
    if T < 10:
        raise ValueError(f"needs at least 10 rows (has {T}).")
    rng = np.random.default_rng(seed)
    real = _stays(corners_median(X))
    fs = np.array([_stays(corners_median(f)) for f in _batches(fakes_walk, X, n_fakes, rng)])
    row = _row("rare-corner stay length / other stays", real, fs, "low", bar_pct)
    return dict(test="vortex", question="Do series pass through the rare corners faster than walks would?",
                rows=[row], passed=row["passed"],
                verdict="PASS: shorter stays than the walks" if row["passed"] else "FAIL: inside the walk fakes",
                notes=["In the runs: countries and states passed, names and rivers didn't; the rarity of these corners "
                       "turned out to be built into the coding, so read a pass as 'turns back quickly', not a passage."],
                details=dict(series=N, rows=T, fakes=n_fakes))


# ================================================================ 10. leaderboard gaps


def leaderboard(scores, n_fakes: int = 20000, seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """Top-N list: do the gaps between neighbor ranks bunch, or keep a rhythm, beyond chance?
    Gaps are scaled by rank (Renyi spacings), which makes them even-sized for any smooth tail."""
    s = np.sort(np.asarray([v for v in scores if np.isfinite(v)], float))[::-1]
    n = s.size
    if n < 10:
        raise ValueError(f"needs at least 10 scores (has {n}).")
    rng = np.random.default_rng(seed)
    g = (s[:-1] - s[1:]) * np.arange(1, n)

    def stats(G):
        G = np.atleast_2d(G)
        m = G.mean(1, keepdims=True)
        cv = G.std(1) / np.where(m[:, 0] > 0, m[:, 0], 1)
        Gc = G - m
        lag = (Gc[:, 1:] * Gc[:, :-1]).mean(1) / np.maximum(Gc.var(1), 1e-12)
        return cv, lag

    cv, lag = stats(g)
    fe = rng.exponential(size=(n_fakes, n - 1))
    M = max(10 * n, 500)
    top = -np.sort(-rng.normal(size=(min(n_fakes, 4000), M)), axis=1)[:, :n]
    fn = (top[:, :-1] - top[:, 1:]) * np.arange(1, n)
    cve, lage = stats(fe)
    cvn, lagn = stats(fn)
    rows = [_row("bunching (gap CV) vs exponential tail", cv[0], cve, "high", bar_pct),
            _row("bunching (gap CV) vs normal top-N", cv[0], cvn, "high", bar_pct),
            _row("rhythm (gap lag-1) vs exponential tail", lag[0], lage, "both", bar_pct),
            _row("rhythm (gap lag-1) vs normal top-N", lag[0], lagn, "both", bar_pct)]
    bunch = rows[0]["passed"] and rows[1]["passed"]
    rhythm = rows[2]["passed"] and rows[3]["passed"]
    big = np.argsort(-g)[:3]
    return dict(test="leaderboard", question="Do the gaps in this top list bunch or keep a rhythm beyond chance?",
                rows=rows, passed=bunch or rhythm,
                verdict=("bunching survives; " if bunch else "no bunching beyond chance; ")
                + ("rhythm survives" if rhythm else "no rhythm beyond chance"),
                notes=["Biggest scaled jumps between ranks " + ", ".join(f"{i + 1}-{i + 2}" for i in sorted(big)) + ".",
                       "Rule: a finding has to clear both kinds of fakes."],
                details=dict(scores=n, fakes=n_fakes))


# ================================================================ 11. three-point phase (Koide / Brannen form)


def menu_fractions(qmax: int = 9, limit: float = math.pi / 3):
    out = set()
    for q in range(1, qmax + 1):
        for p in range(1, q * 2):
            v = p / q
            if 0 < v < limit:
                out.add(round(v, 12))
    return np.array(sorted(out))


def _brannen(m: np.ndarray):
    """sqrt(m_n) = mu (1 + A cos(delta + 2 pi n / 3)), solved exactly. m: (..., 3). Returns A, delta in [0, pi/3], Koide Q."""
    r = np.sqrt(m)
    mu = r.mean(-1)
    w = np.exp(-2j * np.pi * np.arange(3) / 3)
    c = (2 / 3) * (r * w).sum(-1)
    A = np.abs(c) / mu
    d = np.mod(np.angle(c), 2 * np.pi / 3)
    d = np.where(d > np.pi / 3, 2 * np.pi / 3 - d, d)
    Q = m.sum(-1) / (r.sum(-1) ** 2)
    return A, d, Q


def three_point(values, target_A: float = math.sqrt(2), target_delta: float | None = None, lo: float = 0.5,
                hi: float = 200_000.0, n_fakes: int = 500_000, seed: int = 20261028, bar_pct: float = 95.0) -> dict:
    """Three positive values (like three generations of masses): solve the amplitude and phase of the
    three-point cycle, then ask how many random triplets land as close to the target amplitude AND to a simple
    phase (any fraction p/q under pi/3 with q up to 9, or one phase you name). Fakes: log-uniform triplets lo..hi."""
    m = np.asarray(values, float)
    if m.size != 3 or (m <= 0).any():
        raise ValueError("needs exactly three positive values.")
    A, d, Q = (float(np.ravel(v)[0]) for v in _brannen(m[None, :]))
    rng = np.random.default_rng(seed)
    F = np.exp(rng.uniform(math.log(lo), math.log(hi), size=(n_fakes, 3)))
    Af, df, _ = _brannen(F)
    tolA = abs(A - target_A)
    menu = np.array([target_delta]) if target_delta is not None else menu_fractions()
    dist = lambda x: np.min(np.abs(np.asarray(x)[..., None] - menu), axis=-1)
    tolD = float(dist(d))
    nearest = float(menu[np.argmin(np.abs(menu - d))])
    hitA = np.abs(Af - target_A) <= tolA
    hit = hitA & (dist(df) <= tolD)
    frac = float(hit.mean())
    passed = frac < (100 - bar_pct) / 100
    rows = [dict(stat="amplitude A", real=_r(A, 6), note=f"target {target_A:.6g}, off by {tolA:.2e}"),
            dict(stat="phase delta (rad)", real=_r(d, 6), note=f"nearest {'target' if target_delta is not None else 'simple fraction'} {nearest:.6g}, off by {tolD:.2e}"),
            dict(stat="Koide Q", real=_r(Q, 6), note="2/3 when A = root two"),
            dict(stat="fakes as close on A", real=_r(float(hitA.mean()), 6), note=f"of {n_fakes:,}"),
            dict(stat="fakes as close on both", real=_r(frac, 6), passed=passed,
                 note=f"{int(hit.sum()):,} of {n_fakes:,}; bar {(100 - bar_pct) / 100:.2%}")]
    return dict(test="three_point", question="Does this triplet sit on a clean three-point cycle that random triplets rarely hit?",
                rows=rows, passed=passed,
                verdict=(f"PASS: {int(hit.sum()):,} of {n_fakes:,} random triplets as close" if passed else
                         f"FAIL: {frac:.2%} of random triplets land as close"),
                notes=["The phase menu counts the look-elsewhere effect: a phase near ANY simple fraction is allowed to fakes too.",
                       "In the runs: charged leptons give A = root two and delta = 2/9 (0 of 500k fakes); a rediscovery of Koide/Brannen. "
                       "Quarks and neutrinos miss."],
                details=dict(A=A, delta=d, Q=Q, menu_size=int(menu.size), fakes=n_fakes, range=[lo, hi]))


# ================================================================ 12. dose-response leftover


def _hill(d, emax, ed50, n):
    return emax * d ** n / (ed50 ** n + d ** n)


def dose_response(doses, responses, ceiling: float = 100.0, n_boot: int = 2000, seed: int = 20261028,
                  targets=(), bar_pct: float = 95.0) -> dict:
    """Fit a Hill curve (Emax, ED50, steepness n) and read what's left under the ceiling (ceiling - Emax),
    with a bootstrap range. Optional targets: values you suspect the leftover is; the test says which the
    range can rule out."""
    from scipy.optimize import curve_fit
    d = np.asarray(doses, float)
    y = np.asarray(responses, float)
    ok = np.isfinite(d) & np.isfinite(y) & (d > 0)
    d, y = d[ok], y[ok]
    if d.size < 4:
        raise ValueError("needs at least 4 dose/response pairs with doses above zero.")
    p0 = [max(y.max(), 1e-6), float(np.median(d)), 1.0]
    bounds = ([0, 1e-9, 0.1], [ceiling * 3 if ceiling > 0 else np.inf, d.max() * 100, 8])
    p, _ = curve_fit(_hill, d, y, p0=p0, bounds=bounds, maxfev=20000)
    fit = _hill(d, *p)
    res = y - fit
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        yb = fit + rng.choice(res, size=res.size, replace=True)
        try:
            pb, _ = curve_fit(_hill, d, yb, p0=p, bounds=bounds, maxfev=5000)
            boots.append(pb)
        except Exception:
            pass
    boots = np.array(boots)
    lo_q, hi_q = (100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2
    left = ceiling - p[0]
    lb = ceiling - boots[:, 0]
    rng_left = np.percentile(lb, [lo_q, hi_q])
    rows = [dict(stat="Emax (top of the curve)", real=_r(p[0], 3), note=f"range {np.percentile(boots[:, 0], lo_q):.3g} to {np.percentile(boots[:, 0], hi_q):.3g}"),
            dict(stat="ED50 (half-way dose)", real=_r(p[1], 4), note=f"range {np.percentile(boots[:, 1], lo_q):.3g} to {np.percentile(boots[:, 1], hi_q):.3g}"),
            dict(stat="steepness n", real=_r(p[2], 3), note=f"range {np.percentile(boots[:, 2], lo_q):.3g} to {np.percentile(boots[:, 2], hi_q):.3g}"),
            dict(stat="leftover (ceiling - Emax)", real=_r(left, 3), passed=bool(rng_left[0] > 0),
                 note=f"range {rng_left[0]:.3g} to {rng_left[1]:.3g}")]
    for t in targets:
        inside = rng_left[0] <= t <= rng_left[1]
        rows.append(dict(stat=f"is the leftover {t:g}?", real=None, passed=None,
                         note="can't rule it out" if inside else "ruled out by the range"))
    return dict(test="dose_response", question="How much of the ceiling does the curve never reach, and can we pin it down?",
                rows=rows, passed=bool(rng_left[0] > 0),
                verdict=(f"Leftover {left:.3g} (range {rng_left[0]:.3g} to {rng_left[1]:.3g})"
                         + ("; real gap under the ceiling" if rng_left[0] > 0 else "; can't tell it from zero")),
                notes=["In the dopamine run the leftover was about 23% with a range of 3-32%: too wide to tell 23 from 17 or 19."],
                details=dict(points=int(d.size), boots=int(len(boots)), ceiling=ceiling,
                             curve=dict(emax=float(p[0]), ed50=float(p[1]), n=float(p[2]))))


# ================================================================ menu

PANEL_TESTS = {
    "real_vs_fakes": real_vs_fakes,
    "leftovers": leftovers,
    "turnback": turnback,
    "orbit": orbit,
    "interlock": interlock,
    "step_flip": step_flip,
    "vortex": vortex,
}
