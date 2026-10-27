"""Game theory lenses for the Closed Loop — plain Python, no AI needed.

Method by Zach Feemster ("Vex"), CC BY 4.0.

Two lenses for contests, where two sides choose moves against each other:

1. response_test  — who answers whom over time, and with what strategy
                    (match move for move, escalate, back off, or ignore).
                    Fakes keep each side's own rhythm but break the pairing between them
                    (the side being answered is swapped for a fake with its own rhythm and
                    shocks), so a pass means the answering is real, not two series that wiggle alike.

2. blotto_test    — Colonel Blotto: two sides split a budget across many fronts and the
                    bigger spend on a front takes it. Real play is checked against the known
                    equilibrium (Roberson 2006): fakes are drawn from equilibrium play with the
                    same budgets, so a pass means the side does NOT play like equilibrium, and
                    the stat says how.

Both return dicts in the engine's shape: question, stat rows with real / bar / percentile /
passed, a fake arm where it applies, notes, and details.
"""
from __future__ import annotations

import numpy as np

from closedloop.engine import _limit, _pct, ar_fit

# ---------------------------------------------------------------- helpers


def _changes(x, use_log: bool):
    x = np.asarray(x, float)
    logged = bool(use_log and np.all(x > 0))
    return np.diff(np.log(x) if logged else x), logged


def _two_sided(fs: np.ndarray, real: float, bar_pct: float):
    lo, hi = np.percentile(fs, [(100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2])
    return (float(lo), float(hi)), bool(real < lo or real > hi)


def _ranks(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def _spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- 1. response (two sides over time)


def _answer(dy: np.ndarray, dx: np.ndarray, L: int):
    """How the answering side's change (dy) follows the other side's last L changes (dx),
    with its own last change held fixed. Returns (total answer, answer to rises, answer to drops)."""
    n = len(dy)
    y = dy[L:]
    own = dy[L - 1:n - 1]
    lagged = [dx[L - k:n - k] for k in range(1, L + 1)]
    one = np.ones(n - L)
    X = np.column_stack([one] + lagged + [own])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    total = float(b[1:1 + L].sum())
    ups = [np.maximum(v, 0.0) for v in lagged]
    dns = [np.minimum(v, 0.0) for v in lagged]
    X2 = np.column_stack([one] + ups + dns + [own])
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    return total, float(b2[1:1 + L].sum()), float(b2[1 + L:1 + 2 * L].sum())


def _ar_boot(x: np.ndarray, n: int, nsim: int, rng, pmax: int = 4, burn: int = 200) -> np.ndarray:
    """Fakes with the same rhythm as x: its own AR fit, driven by its own shocks resampled
    (so bursty, fat-tailed series keep their bursts). Returns (n, nsim)."""
    x = np.asarray(x, float)
    mu = float(np.mean(x))
    a, _ = ar_fit(x, pmax)
    p = len(a)
    xc = x - mu
    pred = sum(a[k] * xc[p - k - 1:len(xc) - k - 1] for k in range(p))
    resid = xc[p:] - pred
    resid = resid - resid.mean()
    e = rng.choice(resid, size=(n + burn, nsim))
    out = np.zeros((n + burn, nsim))
    for t in range(p, n + burn):
        out[t] = (a[:, None] * out[t - p:t][::-1]).sum(0) + e[t]
    return out[burn:] + mu


def _null_stats(dA, dB, L, n_fakes, rng):
    """Answer stats when the pairing is broken: the answering side stays real, the side it would be
    answering is swapped for a rhythm-matched fake. Columns match STAT_ROWS."""
    n = len(dA)
    FA = _ar_boot(dA, n, n_fakes, rng)
    FB = _ar_boot(dB, n, n_fakes, rng)
    out = np.empty((n_fakes, 5))
    for i in range(n_fakes):
        ab, ab_up, ab_dn = _answer(dB, FA[:, i], L)
        ba, ba_up, ba_dn = _answer(dA, FB[:, i], L)
        gap = _answer(FB[:, i], FA[:, i], L)[0] - _answer(FA[:, i], FB[:, i], L)[0]
        out[i] = (ab, ba, gap, ab_up - ab_dn, ba_up - ba_dn)
    return out


def _strategy(total: float, passed: bool, logged: bool) -> str:
    unit = "in percent terms" if logged else "in raw units"
    if not passed:
        return "ignores (no answer beyond chance)"
    if total < 0:
        return f"backs off / counters: moves the other way, about {abs(total):.2f} of each move ({unit})"
    if total >= 1.25:
        return f"escalates: answers bigger than the move, about {total:.2f}x ({unit})"
    if total >= 0.75:
        return f"matches move for move (tit-for-tat), about {total:.2f}x ({unit})"
    return f"answers in kind, partly: about {total:.2f} of each move ({unit})"


STAT_ROWS = ["B answers A", "A answers B", "Who answers more (B minus A)",
             "B: rises vs drops", "A: rises vs drops"]


def response_test(a, b, names=("A", "B"), lags: int = 1, n_fakes: int = 500, seed: int = 20261027,
                  use_log: bool = True, bar_pct: float = 99.0, detection: bool = True) -> dict:
    """Who answers whom, and how. a, b: the two sides' levels over the same time steps."""
    rng = np.random.default_rng(seed)
    A_name, B_name = names
    dA, logA = _changes(a, use_log)
    dB, logB = _changes(b, use_log)
    logged = logA and logB
    if not logged and (logA or logB):          # keep both on the same footing
        dA, _ = _changes(a, False)
        dB, _ = _changes(b, False)
    n = len(dA)
    L = max(1, int(lags))
    if n < 24 + L:
        raise ValueError(f"needs at least {25 + L} time steps; this has {n + 1}.")
    k = len(STAT_ROWS)
    corr_bar = 100 - (100 - bar_pct) / k          # several looks at one pair → stricter bar each
    ab, ab_up, ab_dn = _answer(dB, dA, L)
    ba, ba_up, ba_dn = _answer(dA, dB, L)
    real = np.array([ab, ba, ab - ba, ab_up - ab_dn, ba_up - ba_dn])
    fs = _null_stats(dA, dB, L, n_fakes, rng)
    labels = [f"{B_name} answers {A_name}", f"{A_name} answers {B_name}",
              f"Who answers more ({B_name} minus {A_name})", f"{B_name}: rises vs drops", f"{A_name}: rises vs drops"]
    rows = []
    for j, name in enumerate(labels):
        bar, ok = _two_sided(fs[:, j], real[j], corr_bar)
        rows.append(dict(stat=name, real=round(float(real[j]), 4),
                         bar=[round(bar[0], 4), round(bar[1], 4)], percentile=round(_pct(fs[:, j], real[j]), 1),
                         passed=ok))
    passed_ab, passed_ba = rows[0]["passed"], rows[1]["passed"]

    # fake arm: two independent sides with the same rhythms — the lens should stay quiet
    fa_A = _ar_boot(dA, n, 1, rng)[:, 0]
    fa_B = _ar_boot(dB, n, 1, rng)[:, 0]
    fa_fs = _null_stats(fa_A, fa_B, L, min(n_fakes, 300), rng)
    fa_real = _answer(fa_B, fa_A, L)[0]
    _, fa_pass = _two_sided(fa_fs[:, 0], fa_real, corr_bar)

    reads = {
        f"{B_name} toward {A_name}": _strategy(ab, passed_ab, logged),
        f"{A_name} toward {B_name}": _strategy(ba, passed_ba, logged),
    }
    if rows[2]["passed"]:
        lead = B_name if ab > ba else A_name
        reads["Lopsided"] = f"{lead} does more of the answering"
    for j, who in ((3, B_name), (4, A_name)):
        if rows[j]["passed"]:
            reads[f"{who} ratchet"] = (f"{who} answers rises more than drops" if real[j] > 0
                                       else f"{who} answers drops more than rises")

    out = dict(test="response",
               question=f"Does one side answer the other's last move, and with what strategy? ({A_name} vs {B_name})",
               names=dict(A=A_name, B=B_name), rows=rows, reads=reads,
               passed=bool(any(r["passed"] for r in rows)),
               fake_arm=dict(real=round(float(fa_real), 4), passed=bool(fa_pass)),
               details=dict(steps=int(n + 1), lags=L, logged=logged, fakes=int(n_fakes),
                            corrected_bar_pct=round(corr_bar, 3),
                            answer_rises_B=round(ab_up, 4), answer_drops_B=round(ab_dn, 4),
                            answer_rises_A=round(ba_up, 4), answer_drops_A=round(ba_dn, 4)),
               notes=[])
    if fa_pass:
        out["notes"].append("The fake arm passed: two unrelated sides with these rhythms fooled the lens once. "
                            "Treat a single pass here with extra suspicion.")

    if detection and not passed_ab:
        # plant B answering A at growing sizes into fake pairs, see what the lens catches
        scale = float(np.std(dB) / (np.std(dA) + 1e-12))
        curve = {}
        for g in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            hits = 0
            trials = 30
            for _ in range(trials):
                xa = _ar_boot(dA, n, 1, rng)[:, 0]
                xb = _ar_boot(dB, n, 1, rng)[:, 0]
                xb[1:] += g * scale * xa[:-1]
                ffs = _null_stats(xa, xb, L, 200, rng)
                _, ok = _two_sided(ffs[:, 0], _answer(xb, xa, L)[0], corr_bar)
                hits += ok
            curve[round(g * scale, 4)] = hits / trials
        lim = _limit(curve)
        out["detection"] = dict(unit=f"answer size ({B_name} per {A_name} move)", catch_rate=curve, limit=lim,
                                pile="below the floor" if lim is None or abs(ab) >= lim / 3 else "nothing there")
    return out


# ---------------------------------------------------------------- 2. Colonel Blotto (budget across fronts)


def _eq_draw(budget_s, budget_w, n, nsim, rng):
    """Equilibrium-style allocations (Roberson 2006, homogeneous fronts, 2/n <= W/S <= 1):
    strong side ~ Uniform[0, 2S/n] per front; weak side puts nothing on a share 1 - W/S of fronts,
    Uniform[0, 2S/n] elsewhere. Each draw is rescaled to spend its budget exactly."""
    cap = 2.0 * budget_s / n
    r = budget_w / budget_s
    S = rng.uniform(0, cap, (nsim, n))
    W = rng.uniform(0, cap, (nsim, n)) * (rng.random((nsim, n)) < r)
    S *= budget_s / np.maximum(S.sum(1, keepdims=True), 1e-12)
    W *= budget_w / np.maximum(W.sum(1, keepdims=True), 1e-12)
    return S, W


def _ks_uniform(x01: np.ndarray) -> float:
    x = np.sort(np.clip(x01, 0, None))
    if len(x) == 0:
        return 0.0
    m = len(x)
    cdf = np.clip(x, 0, 1)
    return float(max(np.max(np.arange(1, m + 1) / m - cdf), np.max(cdf - np.arange(m) / m)))


def _blotto_stats(S, W, cap, tol):
    """Rows: S and W are (nsim, n) or (n,). Returns dict of arrays."""
    S, W = np.atleast_2d(S), np.atleast_2d(W)
    budget_s = S.sum(1)
    over = np.where(S > cap, S - cap, 0).sum(1) / np.maximum(budget_s, 1e-12)
    abandon_w = (W <= tol * cap).mean(1)
    abandon_s = (S <= tol * cap).mean(1)
    upset = (W > S).mean(1)
    ks_s = np.array([_ks_uniform(row / cap) for row in S])
    chase = np.array([_spearman(s, w) for s, w in zip(S, W)])
    return dict(over_cap=over, abandon_weak=abandon_w, abandon_strong=abandon_s,
                upsets=upset, shape_strong=ks_s, chase=chase)


BLOTTO_ROWS = {
    "abandon_weak": "Weak side's fronts left empty",
    "abandon_strong": "Strong side's fronts left empty",
    "over_cap": "Strong side's money piled past the equilibrium cap",
    "shape_strong": "Strong side's spread vs equilibrium spread",
    "upsets": "Fronts where the weak side outspends the strong",
    "chase": "Both sides on the same fronts (rank match)",
}


def blotto_test(a, b, names=("A", "B"), value=None, n_fakes: int = 1000, seed: int = 20261027,
                bar_pct: float = 99.0, empty_tol: float = 0.02) -> dict:
    """Do the two sides split their budgets across fronts the way Blotto equilibrium predicts?
    a, b: each side's spend per front (one row per front in play, zeros included)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("spending can't be negative.")
    keep = (a + b) > 0
    a, b = a[keep], b[keep]
    if value is not None:
        value = np.asarray(value, float)[keep]
    n = len(a)
    if n < 10:
        raise ValueError(f"needs at least 10 fronts with spending; this has {n}.")
    strong_is_a = a.sum() >= b.sum()
    S, W = (a, b) if strong_is_a else (b, a)
    s_name, w_name = (names[0], names[1]) if strong_is_a else (names[1], names[0])
    BS, BW = float(S.sum()), float(W.sum())
    r = BW / BS
    cap = 2.0 * BS / n
    k = len(BLOTTO_ROWS)
    corr_bar = 100 - (100 - bar_pct) / k
    real = _blotto_stats(S, W, cap, empty_tol)
    fS, fW = _eq_draw(BS, BW, n, n_fakes, rng)
    fake = _blotto_stats(fS, fW, cap, empty_tol)
    pred = dict(abandon_weak=(1 - r) + r * empty_tol, abandon_strong=empty_tol, over_cap=0.0, upsets=r / 2,
                chase=0.0, shape_strong=None)
    rows = []
    for key, label in BLOTTO_ROWS.items():
        rv = float(real[key][0])
        bar, ok = _two_sided(fake[key], rv, corr_bar)
        rows.append(dict(key=key, stat=label, real=round(rv, 4),
                         equilibrium=(None if pred[key] is None else round(float(pred[key]), 4)),
                         bar=[round(bar[0], 4), round(bar[1], 4)],
                         percentile=round(_pct(fake[key], rv), 1), passed=ok))

    # fake arm: one equilibrium draw run through the same check should NOT stand out
    fa = _blotto_stats(fS[:1], fW[:1], cap, empty_tol)
    rest = {kk: v[1:] for kk, v in fake.items()}
    fa_pass = any(_two_sided(rest[kk], float(fa[kk][0]), corr_bar)[1] for kk in BLOTTO_ROWS)

    reads = []
    by = {rw["key"]: rw for rw in rows}
    if by["over_cap"]["passed"]:
        reads.append(f"{s_name} piles money onto a few fronts: {by['over_cap']['real']:.0%} of its budget sits past "
                     "the most any front gets in equilibrium.")
    if by["abandon_weak"]["passed"]:
        more = by["abandon_weak"]["real"] > (1 - r)
        reads.append(f"{w_name} leaves {'more' if more else 'fewer'} fronts empty than equilibrium "
                     f"({by['abandon_weak']['real']:.0%} vs about {1 - r:.0%}).")
    if by["abandon_strong"]["passed"]:
        reads.append(f"{s_name} skips {by['abandon_strong']['real']:.0%} of fronts; in equilibrium the strong side "
                     "contests nearly all of them.")
    if by["upsets"]["passed"]:
        more = by["upsets"]["real"] > r / 2
        reads.append(f"{w_name} outspends {s_name} on {by['upsets']['real']:.0%} of fronts, "
                     f"{'more' if more else 'less'} than the {r / 2:.0%} equilibrium predicts.")
    if by["chase"]["passed"]:
        reads.append(f"The two sides {'pile onto the same' if by['chase']['real'] > 0 else 'avoid each other’s'} "
                     f"fronts (rank match {by['chase']['real']:+.2f}); equilibrium play is independent (0).")
    if not reads:
        reads.append("Nothing stands out from equilibrium play at this bar.")

    out = dict(test="blotto",
               question=f"Do {names[0]} and {names[1]} split their budgets across fronts like Blotto equilibrium?",
               strong=s_name, weak=w_name, rows=rows, reads=reads,
               passed=bool(any(rw["passed"] for rw in rows)),
               fake_arm=dict(passed=bool(fa_pass)),
               details=dict(fronts=int(n), budget_strong=round(BS, 2), budget_weak=round(BW, 2),
                            budget_ratio=round(r, 4), equilibrium_cap_per_front=round(cap, 2),
                            corrected_bar_pct=round(corr_bar, 3), fakes=int(n_fakes),
                            empty_means=f"under {empty_tol:.0%} of the cap"),
               notes=["Equilibrium here assumes every front is worth the same and both sides move once, at the "
                      "same time. Real fronts differ in value, so some distance from equilibrium is expected; "
                      "what matters is how each side departs and whether it holds across cycles.",
                      "List every front in play, including ones a side skipped (spend 0). Leaving out skipped "
                      "fronts hides abandonment."])
    if r < 2.0 / n:
        out["notes"].append(f"Budgets are very lopsided (weak/strong {r:.3f} < 2/{n}); the uniform equilibrium "
                            "used here is only approximate in that range.")
    if fa_pass:
        out["notes"].append("The fake arm stood out once, so a single borderline pass here is weak evidence.")

    # chasing beyond value: do they meet on the same fronts more than the fronts' value explains?
    if value is not None and np.isfinite(value).all() and np.std(value) > 0:
        # strip each side's smooth trend against value (on ranks), then ask whether what's left lines up
        order = np.argsort(value, kind="stable")
        w = max(5, n // 10)
        kern = np.ones(w) / w

        def _resid(x):
            rx = _ranks(x)[order]
            pad = np.r_[np.full(w // 2, rx[0]), rx, np.full(w - 1 - w // 2, rx[-1])]
            sm = np.convolve(pad, kern, mode="valid")
            out_ = np.empty(n)
            out_[order] = rx - sm
            return out_

        ea, eb = _resid(a), _resid(b)
        rv = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else 0.0
        fv = np.array([np.corrcoef(ea, rng.permutation(eb))[0, 1] for _ in range(n_fakes)])
        bar, ok = _two_sided(fv, rv, bar_pct)
        out["value_check"] = dict(stat="Rank match beyond the fronts' value", real=round(rv, 4),
                                  bar=[round(bar[0], 4), round(bar[1], 4)], percentile=round(_pct(fv, rv), 1),
                                  passed=ok,
                                  value_track=dict(zip(names, (round(_spearman(a, value), 3),
                                                               round(_spearman(b, value), 3)))))
        out["reads"].append(
            "Beyond the fronts' value, the two sides "
            + ("still meet on the same fronts: they chase each other, not just the prize." if ok and rv > 0 else
               "avoid each other more than value explains." if ok else
               "meet only as much as value explains: both chase the prize, not each other."))
    return out
