"""Zipper finder: shape an EQ, slide a window through time, and catch the moments two series lock together
inside the part of the EQ you boosted. Each lock is checked against fakes and saves a snapshot of what the
EQ looked like at that moment.

Settings were locked on planted data (Oct 28 2026) before any real run:
- bells use time-slide fakes (the other series keeps its real shape but is slid out of step), window = 4 cycles;
- shelves use phase-scrambled fakes (same EQ curve, timing scrambled), window = 5x the shelf's cycle;
- one global check per pair: the real strongest moment must beat the strongest moment of every fake (5%),
  so scanning many windows can't manufacture a lock.
"""
from __future__ import annotations

import numpy as np

from . import rotation as R

KINDS = ["bell", "low shelf", "high shelf"]


def bands(vals: list) -> list[dict]:
    """[[gain_db, width_oct, position, kind], ...] from the drag curve -> band dicts with a center in reference Hz."""
    out = []
    for i, v in enumerate(vals):
        g = float(v[0]); w = float(v[1]) if len(v) > 1 else R.EQ_WIDTH_OCT
        c = float(v[2]) if len(v) > 2 else float(i); k = int(v[3]) if len(v) > 3 else 0
        out.append(dict(gain=g, width=max(0.08, w), hz=float(R.EQ_BANDS[0]) * 2 ** c, kind=k))
    return out


def cycle_of(hz: float) -> float:
    """Cycle length in samples (rows) for a reference-Hz position (bands are read as audio at 44.1 kHz)."""
    return 2 * R.REF_NYQUIST / hz


def response_db(fr_hz: np.ndarray, bl: list[dict]) -> np.ndarray:
    db = np.zeros_like(fr_hz, dtype=float)
    with np.errstate(divide="ignore"):
        for b in bl:
            if not b["gain"]:
                continue
            o = np.log2(np.maximum(fr_hz, 1e-9) / b["hz"])
            if b["kind"] == 1:
                db += b["gain"] / (1 + np.exp(o / (0.6 * b["width"])))
            elif b["kind"] == 2:
                db += b["gain"] / (1 + np.exp(-o / (0.6 * b["width"])))
            else:
                db += b["gain"] * np.exp(-0.5 * (o / b["width"]) ** 2)
    return db


def listen_weight(n: int, bl: list[dict]) -> np.ndarray:
    """What the zipper finder listens to: only the boosted part of the curve (cuts just stay out)."""
    fr = np.fft.rfftfreq(n) * 2 * R.REF_NYQUIST
    w = np.clip(10 ** (response_db(fr, bl) / 20) - 1, 0, None)
    w[0] = 0
    return w / w.max() if w.max() > 0 else w


def boosted(bl: list[dict]) -> list[dict]:
    return [b for b in bl if b["gain"] > 0.05]


def uses_shelf(bl: list[dict]) -> bool:
    return any(b["kind"] in (1, 2) for b in boosted(bl))


def auto_window(bl: list[dict], n: int) -> int:
    """4 cycles of the slowest boosted bell, or 5x a low shelf's cycle; at least 12 rows, at most n - 5."""
    longest = 0.0
    for b in boosted(bl):
        cyc = cycle_of(b["hz"])
        longest = max(longest, 4 * cyc if b["kind"] == 0 else 5 * cyc if b["kind"] == 1 else 4 * cyc)
    return int(min(max(12, round(longest or 12)), n - 5))


def _filt(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    t = np.arange(len(x)); A = np.column_stack([np.ones(len(x)), t])
    c = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return np.fft.irfft(np.fft.rfft(c) * w, n=len(x))


def _slide_corr(a: np.ndarray, b: np.ndarray, W: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as sw
    A = sw(a, W); B = sw(b, W)
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-12
    return np.abs((A * B).sum(1) / den)


def _scramble(x: np.ndarray, rng) -> np.ndarray:
    F = np.fft.rfft(x); ph = np.exp(2j * np.pi * rng.random(len(F))); ph[0] = 1
    if len(x) % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(F * ph, n=len(x))


def pair(x, y, bl: list[dict], W: int | None = None, n_fakes: int = 300, seed: int = 28) -> dict:
    """Slide through time and find where x and y lock inside the boosted part of the EQ."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    W = W or auto_window(bl, n)
    w = listen_weight(n, bl)
    ax, ay = _filt(x, w), _filt(y, w)
    r = _slide_corr(ax, ay, W)
    rng = np.random.default_rng(seed)
    if uses_shelf(bl):
        fakes = np.array([_slide_corr(ax, _filt(_scramble(y - y.mean(), rng), w), W) for _ in range(n_fakes)])
        fake_kind = "phase-scrambled"
    else:
        shifts = list(range(W, n - W + 1)) or [n // 2]
        fakes = np.array([_slide_corr(ax, np.roll(ay, s), W) for s in shifts])
        fake_kind = "time-slid"
    mx = fakes.max(1)
    thr = float(np.quantile(mx, 0.95))
    return dict(r=r, W=W, threshold=thr, p=float((mx >= r.max()).mean()), zip_idx=np.where(r > thr)[0],
                fake_kind=fake_kind, n_fakes=len(fakes))


def snapshot(x, y, i: int, W: int) -> dict:
    """The EQ picture at a moment: share of each series' movement in each of the ten bands, inside that window."""
    out = {}
    for name, s in (("a", x), ("b", y)):
        seg = np.asarray(s, float)[i:i + W]
        seg = seg - np.polyval(np.polyfit(np.arange(W), seg, 1), np.arange(W))
        P = np.abs(np.fft.rfft(seg)) ** 2; fr = np.fft.rfftfreq(W) * 2 * R.REF_NYQUIST
        edges = [0.0] + [np.sqrt(a * b) for a, b in zip(R.EQ_BANDS[:-1], R.EQ_BANDS[1:])] + [R.REF_NYQUIST + 1]
        tot = P[1:].sum() + 1e-12
        out[name] = [float(P[(fr >= edges[k]) & (fr < edges[k + 1]) & (fr > 0)].sum() / tot) for k in range(len(R.EQ_BANDS))]
    return out
