"""Number-menu test: does a target number show up among a set of constants (and their ratios)
more often than chance? Method by Zach Feemster ("Vex"), CC BY 4.0.

Honesty rules built in (from the 30 Oct 2026 contest run, where a naive test gave a false pass):
  * identical values are counted once (one number reached six ways is still one number)
  * fakes copy the boring parts of the menu: whole numbers stay whole, powers of two stay powers of two,
    and constants that are equal in the real menu stay equal in every fake
"""
from __future__ import annotations
import math
import numpy as np


def _kind(x: float) -> str:
    if float(x).is_integer() and x > 0:
        n = int(x)
        return 'pow2' if n & (n - 1) == 0 else 'int'
    return 'real'


def _fake_value(x: float, kind: str, rng) -> float:
    if kind == 'pow2':
        e = int(round(math.log2(x))) + int(rng.integers(-3, 4))
        return float(2 ** max(e, 0))
    lo, hi = math.log10(abs(x) / 10), math.log10(abs(x) * 10)
    v = 10 ** rng.uniform(lo, hi)
    if kind == 'int':
        v = max(1.0, float(round(v)))
    return math.copysign(v, x)


def _candidates(vals: np.ndarray, names: list[str], ratios: bool):
    c, lab = list(vals), list(names)
    if ratios:
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j and vals[j] != 0:
                    c.append(vals[i] / vals[j]); lab.append(f'{a}/{b}')
    return np.array(c, float), lab


def _distinct_hits(c: np.ndarray, target: float, tol: float) -> list[int]:
    idx = np.flatnonzero(np.abs(c / target - 1) < tol)
    seen, keep = [], []
    for i in idx:
        if not any(abs(c[i] / s - 1) < 1e-9 for s in seen):
            seen.append(c[i]); keep.append(int(i))
    return keep


def menu_test(constants: dict, target: float, tol: float = 0.01, ratios: bool = True,
              n_fakes: int = 2000, bar_pct: float = 99.0, seed: int = 78) -> dict:
    names = list(constants)
    vals = np.array([float(constants[k]) for k in names])
    c, lab = _candidates(vals, names, ratios)
    raw = int((np.abs(c / target - 1) < tol).sum())
    keep = _distinct_hits(c, target, tol)
    real = len(keep)
    # equal constants share one fake draw; each keeps its kind
    groups = {}
    for i, v in enumerate(vals):
        groups.setdefault(v, []).append(i)
    rng = np.random.default_rng(seed)
    fk = np.empty(n_fakes, int)
    for t in range(n_fakes):
        fv = vals.copy()
        for v, members in groups.items():
            fv[members] = _fake_value(v, _kind(v), rng)
        fk[t] = len(_distinct_hits(_candidates(fv, names, ratios)[0], target, tol))
    bar = float(np.percentile(fk, bar_pct))
    pct = float((fk < real).mean() * 100 + 0.5 * (fk == real).mean() * 100)
    return dict(target=target, candidates=len(c), raw_hits=raw, distinct_hits=real,
                hits=[(lab[i], float(c[i])) for i in keep], fake_mean=float(fk.mean()),
                fake_bar=bar, percentile=round(pct, 2), passed=bool(real > bar),
                settings=dict(tol=tol, ratios=ratios, n_fakes=n_fakes, bar_pct=bar_pct, seed=seed,
                              fakes='lattice-matched (int/pow2 kept, equal constants tied)'))
