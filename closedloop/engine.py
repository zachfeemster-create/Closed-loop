"""Closed Loop test engine — plain Python, no AI needed.

Method by Zach Feemster ("Vex"), CC BY 4.0.

Every test returns a dict with the same keys:
    test, question, stat_name, real, bar, percentile, passed,
    fake_arm {real, passed}, details {...}, detection {...} (on a miss), notes [...]
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- helpers


def _pct(fakes: np.ndarray, real: float) -> float:
    return float(np.mean(fakes < real) * 100.0)


def lagmat(x: np.ndarray, L: int) -> np.ndarray:
    """Columns x(t), x(t-1), ..., x(t-L+1). Rows before L are junk (wrapped)."""
    return np.column_stack([np.roll(x, k) for k in range(L)])


def ar_fit(x: np.ndarray, pmax: int = 4):
    """Fit AR(p), p chosen by AIC. Returns (coefs, noise_sd)."""
    x = np.asarray(x, float) - np.mean(x)
    best = None
    for p in range(1, pmax + 1):
        if len(x) <= 3 * p:
            break
        X = np.column_stack([x[p - k - 1: len(x) - k - 1] for k in range(p)])
        y = x[p:]
        a, *_ = np.linalg.lstsq(X, y, rcond=None)
        s2 = float(np.var(y - X @ a)) + 1e-18
        aic = len(y) * np.log(s2) + 2 * p
        if best is None or aic < best[0]:
            best = (aic, a, np.sqrt(s2))
    if best is None:
        return np.array([0.0]), float(np.std(x)) + 1e-12
    return best[1], best[2]


def ar_sim(a: np.ndarray, s: float, n: int, nsim: int, rng, burn: int = 300) -> np.ndarray:
    """Simulate nsim AR series of length n. Returns (n, nsim)."""
    p = len(a)
    x = np.zeros((n + burn, nsim))
    e = rng.normal(0.0, s, (n + burn, nsim))
    for t in range(p, n + burn):
        x[t] = (a[:, None] * x[t - p:t][::-1]).sum(0) + e[t]
    return x[burn:]


def _limit(curve: dict, rule: float = 0.8):
    for k, v in curve.items():
        if v >= rule:
            return k
    return None


# ---------------------------------------------------------------- 1. cycle (one series)


def cycle_test(y, drivers=None, season: int = 0, band=(2.0, None), n_fakes: int = 500,
               seed: int = 20261025, mask=None, bar_pct: float = 99.0, driver_lags: int = 12,
               detection: bool = True, _fake_arm: bool = True):
    """Remove drivers (lagged regression) and a fixed season, then look for the strongest
    repeating cycle in what's left, against momentum-matched (AR) fakes.

    band: (min_period, max_period) in rows; max None = n/3.
    mask: boolean array, True = flagged event row (remainder set to zero there).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    n = len(y)
    L = max(1, min(driver_lags, n // 20))
    start = L if drivers is not None and np.size(drivers) else 0
    if drivers is not None and np.size(drivers):
        D = np.atleast_2d(np.asarray(drivers, float))
        if D.shape[0] != n:
            D = D.T
        X = np.column_stack([np.ones(n)] + [lagmat(D[:, j], L) for j in range(D.shape[1])])[start:]
        b, *_ = np.linalg.lstsq(X, y[start:], rcond=None)
        fitted = X @ b
        r = y[start:] - fitted
    else:
        fitted = np.full(n - start, y.mean())
        r = y[start:] - fitted
    m = len(r)
    if season and season > 1:
        pos = (np.arange(m) + start) % season
        smean = np.array([r[pos == k].mean() if np.any(pos == k) else 0.0 for k in range(season)])
        seas = smean[pos]
    else:
        seas = np.zeros(m)
    rem = r - seas
    msk = None if mask is None else np.asarray(mask, bool)[start:]
    pmin, pmax = band[0], (band[1] if band[1] else m / 3)

    def score(x):
        x = np.asarray(x, float)
        if msk is not None:
            x = np.where(msk, 0.0, x)
        x = x - x.mean()
        P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
        k = np.arange(len(P))
        per = np.where(k > 0, len(x) / np.maximum(k, 1), np.inf)
        ok = (per >= pmin) & (per <= pmax)
        if season and season > 1:
            for h in (1, 2, 3, 4):
                hp = season / h
                ok &= ~((per > hp * 0.95) & (per < hp * 1.05))
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return 0.0, []
        top = idx[np.argsort(P[idx])[::-1][:3]]
        return float(P[idx].max()), [(round(float(per[t]), 2), round(float(P[t]), 6)) for t in top]

    real, top = score(rem)
    a, s = ar_fit(rem)
    F = ar_sim(a, s, m, n_fakes, rng)
    fs = np.array([score(F[:, i])[0] for i in range(n_fakes)])
    bar = float(np.percentile(fs, bar_pct))
    # fake arm: drivers' fit + season + AR noise, through the whole pipeline
    fake_y = np.concatenate([y[:start], fitted + seas + ar_sim(a, s, m, 1, rng)[:, 0]])
    fa = cycle_test(fake_y, drivers, season, band, n_fakes=min(200, n_fakes), seed=seed + 1,
                    mask=mask, bar_pct=bar_pct, driver_lags=driver_lags, detection=False,
                    _fake_arm=False) if _fake_arm else None
    out = dict(test="cycle", question="Is there a repeating cycle left after the obvious drivers and season?",
               stat_name="strongest cycle power", real=round(real, 6), bar=round(bar, 6),
               percentile=_pct(fs, real), passed=bool(real > bar),
               fake_arm=None if fa is None else dict(real=fa["real"], passed=bool(fa["real"] > bar)),
               details=dict(rows_used=int(m), top3_periods_rows=top, ar_order=int(len(a)),
                            share_left=round(float(np.var(rem) / (np.var(y) + 1e-18)), 4)),
               notes=[])
    if detection and not out["passed"]:
        sd = float(np.std(rem))
        grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
        curve = {}
        t = np.arange(m)
        for g in grid:
            Fs = ar_sim(a, s, m, 100, rng)
            hits = 0
            for i in range(100):
                per = np.exp(rng.uniform(np.log(pmin), np.log(pmax)))
                x = Fs[:, i] + g * sd * 2 * np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
                hits += score(x)[0] > bar
            curve[g] = hits / 100
        lim = _limit(curve)
        out["detection"] = dict(unit="cycle amplitude as a multiple of the remainder's spread x2",
                                catch_rate=curve, limit=lim,
                                limit_in_data_units=None if lim is None else round(lim * sd * 2, 6),
                                real_peak_amplitude=round(float(np.sqrt(4 * real / m)), 6),
                                pile=("below the floor" if lim is None or np.sqrt(4 * real / m) < lim * sd * 2
                                      else "nothing there"))
    return out


# ---------------------------------------------------------------- 2. asymmetry (driver -> response)


class _Asym:
    def __init__(self, dy, dx, L, P, early, rowmask):
        self.dy, self.n, self.L, self.P, self.early = dy, len(dy), L, P, early
        self.t0 = max(L, P)
        up, dn = np.clip(dx, 0, None), np.clip(dx, None, 0)
        self.EX = np.column_stack([lagmat(up, L), lagmat(dn, L)])[self.t0:]
        self.keep = np.ones(self.n - self.t0, bool) if rowmask is None else ~np.asarray(rowmask, bool)[self.t0:]
        self.b, X = self.fit(dy)
        self.e = (dy[self.t0:] - X @ self.b)[self.keep]
        self.bs = self.b.copy()
        a = (self.b[1:1 + L] + self.b[1 + L:1 + 2 * L]) / 2
        self.bs[1:1 + L] = a
        self.bs[1 + L:1 + 2 * L] = a

    def fit(self, y):
        Xo = np.column_stack([y[self.t0 - k:self.n - k] for k in range(1, self.P + 1)])
        X = np.column_stack([np.ones(self.n - self.t0), self.EX, Xo])
        b, *_ = np.linalg.lstsq(X[self.keep], y[self.t0:][self.keep], rcond=None)
        return b, X

    def D(self, b):
        L, e = self.L, self.early
        return float(b[1:1 + e].sum() - b[1 + L:1 + L + e].sum())

    def sims(self, nsim, rng, delta=0.0, block=8):
        L, e = self.L, self.early
        bb = self.bs.copy()
        bb[1:1 + e] += delta / e
        base = bb[0] + self.EX @ bb[1:1 + 2 * L]
        ar = bb[1 + 2 * L:]
        m = self.n - self.t0
        res = self.e
        nb = int(np.ceil(m / block))
        E = np.stack([np.concatenate([res[k:k + block] for k in rng.integers(0, max(1, len(res) - block), nb)])[:m]
                      for _ in range(nsim)], 1)
        Y = np.tile(self.dy[:, None], (1, nsim)).astype(float)
        for t in range(self.t0, self.n):
            Y[t] = base[t - self.t0] + (ar[:, None] * Y[t - self.P:t][::-1]).sum(0) + E[t - self.t0]
        return np.array([self.D(self.fit(Y[:, k])[0]) for k in range(nsim)])


def asymmetry_test(response, driver, lags: int = 12, early: int = 4, own_lags: int = 4,
                   direction: str = "up", n_fakes: int = 500, seed: int = 20261025, mask=None,
                   bar_pct: float = 99.0, already_changes: bool = False, use_log: bool = True,
                   detection: bool = True):
    """Does the response move faster when the driver rises than when it falls (or the reverse)?
    D = (response to driver increases, first `early` lags) - (response to decreases, same lags)."""
    rng = np.random.default_rng(seed)
    y, x = np.asarray(response, float), np.asarray(driver, float)
    if not already_changes:
        if use_log:
            y, x = np.log(y), np.log(x)
        dy, dx = np.diff(y), np.diff(x)
        msk = None if mask is None else np.asarray(mask, bool)[1:]
    else:
        dy, dx, msk = y, x, (None if mask is None else np.asarray(mask, bool))
    T = _Asym(dy, dx, lags, own_lags, early, msk)
    real = T.D(T.b)
    fs = T.sims(n_fakes, rng)
    if direction == "up":
        bar = float(np.percentile(fs, bar_pct)); ok = lambda v: v > bar
    elif direction == "down":
        bar = float(np.percentile(fs, 100 - bar_pct)); ok = lambda v: v < bar
    else:
        lo, hi = np.percentile(fs, [(100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2])
        bar = (float(lo), float(hi)); ok = lambda v: v < lo or v > hi
    fa = float(T.sims(1, rng)[0])
    L = lags
    out = dict(test="asymmetry", question=f"Does the response move {'faster up' if direction=='up' else 'faster down' if direction=='down' else 'unevenly'} with the driver?",
               stat_name=f"gap, first {early} steps (up minus down)", real=round(real, 5),
               bar=(round(bar, 5) if not isinstance(bar, tuple) else [round(bar[0], 5), round(bar[1], 5)]),
               percentile=_pct(fs, real), passed=bool(ok(real)),
               fake_arm=dict(real=round(fa, 5), passed=bool(ok(fa))),
               details=dict(rows_used=int(T.keep.sum()),
                            up_early=round(float(T.b[1:1 + early].sum()), 4),
                            down_early=round(float(T.b[1 + L:1 + L + early].sum()), 4),
                            up_total=round(float(T.b[1:1 + L].sum()), 4),
                            down_total=round(float(T.b[1 + L:1 + 2 * L].sum()), 4)),
               notes=[])
    if detection and not out["passed"]:
        sign = -1 if (direction == "down" or (direction == "either" and real < 0)) else 1
        curve = {}
        for g in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]:
            curve[g] = float(np.mean([ok(v) for v in T.sims(100, rng, sign * g)]))
        lim = _limit(curve)
        out["detection"] = dict(unit="gap (share of a driver move)", catch_rate=curve, limit=lim,
                                pile=("below the floor" if lim is None or abs(real) < lim else "nothing there")
                                if abs(real) > 0 else "nothing there")
        if lim is not None and abs(real) < lim / 3:
            out["detection"]["pile"] = "nothing there"
    return out


# ---------------------------------------------------------------- 3. lead (many series)


def _leads(Z):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    a, b = Zs[:-1], Zs[1:]
    C = (a.T @ b) / len(a)
    k = Z.shape[1]
    return np.array([np.mean([C[i, j] - C[j, i] for j in range(k) if j != i]) for i in range(k)])


def lead_test(panel, names=None, n_fakes: int = 500, seed: int = 20261025, mask=None, bar_pct: float = 99.0,
              already_changes: bool = False, use_log: bool = True, detection: bool = True):
    """Does one series move first? Lead = avg over others of corr(its change now, theirs next step)
    minus the reverse. Fakes: no-lead world (own 2 lags, same-step correlated shocks)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(panel, float)
    if not already_changes:
        Z = np.diff(np.log(X) if use_log else X, axis=0)
        msk = None if mask is None else np.asarray(mask, bool)[1:]
    else:
        Z, msk = X, (None if mask is None else np.asarray(mask, bool))
    if msk is not None:
        Z = Z[~msk]
    T, k = Z.shape
    names = names or [f"s{i}" for i in range(k)]
    Lr = _leads(Z)
    stat = float(np.abs(Lr).max())
    A, E = [], []
    for i in range(k):
        Xi = np.column_stack([np.ones(T - 2), Z[1:-1, i], Z[:-2, i]])
        bi, *_ = np.linalg.lstsq(Xi, Z[2:, i], rcond=None)
        A.append(bi); E.append(Z[2:, i] - Xi @ bi)
    A = np.array(A)
    Lc = np.linalg.cholesky(np.cov(np.column_stack(E).T) + 1e-12 * np.eye(k))

    def sim(plant=0.0, src=0):
        x = np.zeros((T + 200, k))
        sh = rng.normal(size=(T + 200, k)) @ Lc.T
        sd = np.sqrt(np.diag(np.cov(np.column_stack(E).T)))
        for t in range(2, T + 200):
            x[t] = A[:, 0] + A[:, 1] * x[t - 1] + A[:, 2] * x[t - 2] + sh[t]
            if plant:
                x[t] += plant * (x[t - 1, src] / (sd[src] + 1e-12)) * sd * (np.arange(k) != src)
        return x[200:]

    fs = np.array([np.abs(_leads(sim())).max() for _ in range(n_fakes)])
    bar = float(np.percentile(fs, bar_pct))
    fa = float(np.abs(_leads(sim())).max())
    top = int(np.argmax(np.abs(Lr)))
    out = dict(test="lead", question="Does one series move first?", stat_name="largest |lead|",
               real=round(stat, 5), bar=round(bar, 5), percentile=_pct(fs, stat), passed=bool(stat > bar),
               fake_arm=dict(real=round(fa, 5), passed=bool(fa > bar)),
               details=dict(rows_used=int(T), strongest=names[top], strongest_direction="leads" if Lr[top] > 0 else "lags",
                            lead_by_series={n: round(float(v), 4) for n, v in zip(names, Lr)}),
               notes=[])
    if detection and not out["passed"]:
        curve = {}
        for g in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]:
            curve[g] = float(np.mean([np.abs(_leads(sim(g))).max() > bar for _ in range(60)]))
        lim = _limit(curve)
        out["detection"] = dict(unit="planted next-step pull from one series on the others (in spreads)",
                                catch_rate=curve, limit=lim, pile="below the floor" if lim is None else "nothing there")
    return out


# ---------------------------------------------------------------- 4. drift (many series)


def _hp_matrix(n, lam=100.0):
    D = np.diff(np.eye(n), 2, axis=0)
    return np.linalg.inv(np.eye(n) + lam * D.T @ D)


def _thetas(left, H):
    lev = np.cumsum(left, 0)
    C = lev - H @ lev
    dC = np.diff(C, axis=0)
    z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)
    return np.arctan2(z(dC), z(C[1:]))


def _locks(th):
    N = th.shape[1]
    iu = np.triu_indices(N, 1)
    E = np.exp(1j * th)
    return np.abs((E.conj().T @ E) / th.shape[0])[iu]


def _r1r2(th):
    T = th.shape[0]; h = T // 2
    R1 = np.corrcoef(_locks(th[:h]), _locks(th[h:]))[0, 1]
    w = np.array_split(np.arange(T), 4)
    Lw = [_locks(th[i]) for i in w]
    c = lambda a, b: np.corrcoef(Lw[a], Lw[b])[0, 1]
    return float(R1), float(np.mean([c(0, 1), c(1, 2), c(2, 3)]) - c(0, 3))


def drift_test(panel, n_fakes: int = 200, seed: int = 20261025, use_log: bool = True, bar_pct: float = 99.0,
               _fake_arm: bool = True):
    """Do the pairs that move together keep changing over time (drift), beyond shared-clock fakes?
    R1 low = pair links don't recur between halves; R2 high = links fade with time gap."""
    rng = np.random.default_rng(seed)
    X = np.asarray(panel, float)
    ch = np.diff(np.log(X) if use_log else X, axis=0)
    left = ch - ch.mean(1, keepdims=True)
    T = left.shape[0]
    H = _hp_matrix(T)
    R1, R2 = _r1r2(_thetas(left, H))

    def fakeB(lf):
        mu = lf.mean(0); x = lf - mu
        F = np.fft.rfft(x, axis=0); K = F.shape[0]
        ph = rng.uniform(0, 2 * np.pi, K); ph[0] = 0
        if T % 2 == 0:
            ph[-1] = 0
        return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0) + mu

    def fakeC(lf):
        b = fakeB(lf); o = np.empty_like(b)
        for j in range(lf.shape[1]):
            o[np.argsort(b[:, j]), j] = np.sort(lf[:, j])
        return o

    fb = np.array([_r1r2(_thetas(fakeB(left), H)) for _ in range(n_fakes)])
    fc = np.array([_r1r2(_thetas(fakeC(left), H)) for _ in range(n_fakes)])
    lo = (100 - bar_pct)
    r1bar = float(min(np.percentile(fb[:, 0], lo), np.percentile(fc[:, 0], lo)))
    r2bar = float(max(np.percentile(fb[:, 1], bar_pct), np.percentile(fc[:, 1], bar_pct)))
    passed = bool(R1 < r1bar)
    fa_out = None
    if _fake_arm:
        # fake arm: independent momentum-matched series (no links at all), run through the whole test
        mu = ch.mean(0); y = ch - mu
        phi = np.clip((y[1:] * y[:-1]).sum(0) / ((y[:-1] ** 2).sum(0) + 1e-18), -0.99, 0.99)
        sig = np.sqrt(((y[1:] - phi * y[:-1]) ** 2).mean(0))
        f = np.empty_like(ch); f[0] = rng.normal(0, 1, ch.shape[1]) * sig / np.sqrt(1 - phi ** 2)
        e = rng.normal(0, 1, ch.shape) * sig
        for t in range(1, len(ch)):
            f[t] = phi * f[t - 1] + e[t]
        base = np.log(X[:1]) if use_log else X[:1]
        lvl = np.vstack([base, base + np.cumsum(f + mu, 0)])
        fake_panel = np.exp(lvl) if use_log else lvl
        fr = drift_test(fake_panel, n_fakes=min(100, n_fakes), seed=seed + 7, use_log=use_log, bar_pct=bar_pct,
                        _fake_arm=False)
        fa_out = dict(real=fr["real"], passed=bool(fr["passed"]))
    return dict(test="drift", question="Do the links between series keep shifting over time?",
                stat_name="R1 recurrence (low = drift)", real=round(R1, 5), bar=round(r1bar, 5),
                percentile=_pct(np.concatenate([fb[:, 0], fc[:, 0]]), R1), passed=passed,
                fake_arm=fa_out,
                details=dict(R2_decay=round(R2, 5), R2_bar=round(r2bar, 5), R2_passed=bool(R2 > r2bar),
                             series=int(X.shape[1]), rows=int(X.shape[0])),
                notes=["Detection limit for drift isn't built yet — a miss here means 'not seen', not 'not there'.",
                       "Very smooth series (modelled or interpolated data) can make the fake arm pass; then the result doesn't count.",
                       "A pass means the links recur less than a steady world would allow — that can be drifting links or no stable links at all."])


# ---------------------------------------------------------------- 5. remainder bunching (many series)


def bunching_test(panel, n_fakes: int = 200, seed: int = 20261025, use_log: bool = True, bar_pct: float = 99.0):
    """After each series' own shape and the shared traffic are removed, do the leftover turns
    bunch at the same moments? Fakes: momentum-matched (AR(1)) series, whole pipeline."""
    rng = np.random.default_rng(seed)
    X = np.asarray(panel, float)
    ch = np.diff(np.log(X) if use_log else X, axis=0)
    T, N = ch.shape
    H = _hp_matrix(T)

    def cycles(c):
        c = c - c.mean(1, keepdims=True)
        lev = np.cumsum(c, 0); C = lev - H @ lev
        return (C - C.mean(0)) / (C.std(0) + 1e-12)

    def shares(M):
        s = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2
        return s / s.sum()

    def remainder(M, k):
        U, s, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        return M - M.mean(0) - (U[:, :k] * s[:k]) @ Vt[:k]

    def slide(M):
        return np.column_stack([np.roll(M[:, j], rng.integers(M.shape[0])) for j in range(M.shape[1])])

    def bunch(R):
        Z = R / (R.std(0) + 1e-12); E = np.zeros_like(Z, bool); mid = Z[1:-1]
        E[1:-1] = ((mid > Z[:-2]) & (mid > Z[2:]) | (mid < Z[:-2]) & (mid < Z[2:])) & (np.abs(mid) > 1)
        cnt = E.sum(1)[1:-1]
        return float(cnt.var() / (cnt.mean() + 1e-12)), E.sum(1)

    Xc = cycles(ch)
    p95 = np.percentile(np.array([shares(slide(Xc)) for _ in range(100)]), 95, 0)

    def kof(s):
        k = 0
        while k < len(s) and s[k] > p95[k]:
            k += 1
        return k

    k = kof(shares(Xc))
    real, counts = bunch(remainder(Xc, k))
    fits = []
    for j in range(N):
        x = ch[:, j] - ch[:, j].mean()
        a = float(np.clip(np.corrcoef(x[1:], x[:-1])[0, 1], -0.95, 0.95)) if np.std(x) > 0 else 0.0
        fits.append((a, float(np.std(x) * np.sqrt(1 - a * a))))
    av = np.array([f[0] for f in fits]); sv = np.array([f[1] for f in fits])

    def fake():
        x = np.zeros((T, N)); e = rng.normal(size=(T, N)) * sv
        for t in range(T):
            x[t] = (av * x[t - 1] if t else 0) + e[t]
        return x

    fs = []
    for _ in range(n_fakes):
        M = cycles(fake()); fs.append(bunch(remainder(M, kof(shares(M))))[0])
    fs = np.array(fs)
    bar = float(np.percentile(fs, bar_pct))
    M = cycles(fake()); fa = bunch(remainder(M, kof(shares(M))))[0]
    busiest = np.argsort(counts[1:-1])[::-1][:3] + 1
    return dict(test="bunching", question="Do the leftover turns bunch at the same moments?",
                stat_name="bunching (variance / mean of turns per step)", real=round(real, 4), bar=round(bar, 4),
                percentile=_pct(fs, real), passed=bool(real > bar),
                fake_arm=dict(real=round(fa, 4), passed=bool(fa > bar)),
                details=dict(traffic_components_removed=int(k), busiest_rows=[int(b + 1) for b in busiest],
                             busiest_counts=[int(counts[b]) for b in busiest]),
                notes=["Busiest moments that match known world events are a real 'shared clock' (time link), not a direct link.",
                       "Blind spot: big shared moves are removed as traffic before turns are counted, so a clean event that hits "
                       "everything (or a large group) at once gets removed rather than counted. This test registers loosely "
                       "synchronized turns, not clean shared shocks.",
                       "Detection limit for bunching isn't built yet."])


# ---------------------------------------------------------------- data checks


import re

AGG_PATTERNS = [r"\bworld\b", r"\bincome\b", r"\boecd\b", r"\btotal\b", r"\bregion", r"\baverage\b",
                r"\bunion\b", r"\barea\b", r"\baggregate", r"sub-saharan", r"latin america", r"east asia",
                r"europe &", r"middle east", r"\bida\b", r"\bibrd\b", r"dividend", r"\bfragile\b",
                r"small states", r"\ball\b", r"members\b", r"\bexcluding\b"]


def data_warnings(df, cols):
    """Plain-language checks from the traps list."""
    w = []
    agg = [c for c in cols if any(re.search(p, str(c).lower()) for p in AGG_PATTERNS)]
    if agg:
        w.append(f"These look like group averages, not single members — they lock together by construction: {', '.join(map(str, agg[:8]))}{'…' if len(agg) > 8 else ''}")
    nonpos = [c for c in cols if (df[c] <= 0).any()]
    if nonpos:
        w.append(f"Zero or negative values in {', '.join(map(str, nonpos[:6]))} — logs are off for these. Check whether the column is a rate of change rather than a level.")
    short = len(df) < 30
    if short:
        w.append("Fewer than 30 rows — most tests will have very little power.")
    return w
