"""Lens rotation with Max in the back — plain Python, no AI, no tokens.

Method by Zach Feemster ("Vex"), CC BY 4.0.

The lenses stay fixed; every dataset rotates through every lens, under every EQ state, and comes back
carrying all of its results (the spiral staircase). Datasets are blinded (D01, D02, ...) while measured.
Each step is written to a step log as it happens. Max in the back reads the log as the tunnels get built;
when something stumbles or passes he calls a jinx and the jinx is actually run.

Honesty rules built in:
  * every lens is scored against momentum-matched fakes (same trend, same wobble, no pattern)
  * the fakes go through the exact same EQ as the real data, so the knobs can't invent a pattern
  * a fake arm runs through the whole pipeline as if it were real (frame check)
  * Max only reads finished steps; his jinxes become new runs, never edits to old ones
"""
from __future__ import annotations

import numpy as np

from .engine import ar_fit, ar_sim

# ---------------------------------------------------------------- EQ

REF_NYQUIST = 22050.0                     # bands are read as if the series were audio at 44.1 kHz
EQ_BANDS = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]     # the ten drag dots (octave ladder)
FLAT = {f: 0.0 for f in EQ_BANDS}
# Zach's original handles (kept exactly, so the nightly/weekly runs stay comparable with earlier logs)
ZACH_CURVE = {125: 0.0, 500: 0.0, 1000: 0.0, 2000: -1.5, 4000: 0.5, 7000: 1.0, 8000: 0.15, 16000: 0.2}
# Zach's phone EQ, Oct 27 2026 screenshot
PHONE_CURVE = dict(zip(EQ_BANDS, [-0.9, -0.2, -0.8, 0.7, 1.0, -1.4, -0.4, -0.2, 1.6, 0.1]))
EQ_WIDTH_OCT = 0.7                        # bell width of each band, in octaves
# one color per band, low to high (Parametric EQ 2 style); hues match the recolored sigil handles
BAND_HUES = [15, 35, 55, 95, 140, 185, 220, 255, 285, 315]
BAND_COLORS = [f"hsl({h},90%,60%)" for h in BAND_HUES]


def bells(curve: dict) -> list[tuple[float, float, float]]:
    """(center Hz, gain dB, width in octaves) for every band in a curve. A band's value is either a plain gain
    (fixed width) or [gain, width] once it has been shaped; keys are the band centers, moved or not."""
    out = []
    for f0, v in curve.items():
        g, w = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), EQ_WIDTH_OCT)
        out.append((float(f0), g, max(0.08, w)))
    return out


def curve_json(curve: dict) -> dict:
    """The curve as plain rounded numbers, for logs and for counting looks."""
    return {f"{round(f, 1):g}": (round(g, 2) if abs(w - EQ_WIDTH_OCT) < 1e-9 else [round(g, 2), round(w, 3)])
            for f, g, w in sorted(bells(curve))}


def curve_at(curve: dict, freqs=EQ_BANDS) -> list[float]:
    """The dB a curve actually gives at each frequency (so any curve can be drawn on the ten dots)."""
    return [float(sum(g * np.exp(-0.5 * (np.log2(f / f0) / w) ** 2) for f0, g, w in bells(curve))) for f in freqs]


def eq_gain(n: int, curve: dict) -> np.ndarray:
    """Amplitude gain for each FFT bin of an n-point series."""
    fr = np.fft.rfftfreq(n) * 2 * REF_NYQUIST          # bin position in 'reference Hz'
    db = np.zeros_like(fr)
    for f0, g, w in bells(curve):
        if g:
            with np.errstate(divide="ignore"):
                oct_ = np.log2(np.maximum(fr, 1e-9) / f0)
            db += g * np.exp(-0.5 * (oct_ / w) ** 2)
    db[0] = 0.0
    return 10 ** (db / 20)


def is_flat(curve: dict) -> bool:
    return not any(abs(g) > 1e-12 for _, g, _ in bells(curve))


def apply_eq(X: np.ndarray, curve: dict) -> np.ndarray:
    """EQ along axis 0 (works on one series or a stack of fakes). Trend is kept outside the EQ."""
    if is_flat(curve):
        return X
    n = X.shape[0]
    t = np.arange(n)
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    trend = A @ coef
    g = eq_gain(n, curve)
    g = g if X.ndim == 1 else g[:, None]
    return np.fft.irfft(np.fft.rfft(X - trend, axis=0) * g, n=n, axis=0) + trend


# ---------------------------------------------------------------- band view (the spiral across the dB plane)

def band_split(x: np.ndarray) -> dict:
    """Split a series into the ten EQ bands (octave slices, read as audio at 44.1 kHz).
    Returns {band_hz: dict(env_db, phase, share)}; bands too low for the series' length come back empty."""
    x = np.asarray(x, float)
    n = len(x)
    c = _detrend(x[:, None])[:, 0]
    F = np.fft.fft(c)
    fr = np.abs(np.fft.fftfreq(n)) * 2 * REF_NYQUIST
    edges = [0.0] + [np.sqrt(a * b) for a, b in zip(EQ_BANDS[:-1], EQ_BANDS[1:])] + [REF_NYQUIST + 1]
    pos = np.fft.fftfreq(n) > 0
    total = float((np.abs(F[pos]) ** 2).sum()) + 1e-12
    out = {}
    for i, f0 in enumerate(EQ_BANDS):
        m = pos & (fr >= edges[i]) & (fr < edges[i + 1])
        if not m.any():
            out[f0] = None
            continue
        a = np.fft.ifft(np.where(m, 2 * F, 0))        # analytic signal of this band only
        env = np.abs(a)
        out[f0] = dict(env_db=20 * np.log10(env + 1e-9 * env.max() + 1e-300),
                       phase=np.unwrap(np.angle(a)), share=float((np.abs(F[m]) ** 2).sum() / total),
                       bins=int(m.sum()))
    return out


def octave_lock(bands: dict) -> dict:
    """How tightly each band locks to the next one up (one turn of the low band = two of the high band).
    1 = locked like teeth in a zipper, 0 = no relation."""
    out = {}
    for lo, hi in zip(EQ_BANDS[:-1], EQ_BANDS[1:]):
        a, b = bands.get(lo), bands.get(hi)
        if a and b and a["bins"] >= 3 and b["bins"] >= 3:      # a one- or two-slice band locks by itself
            out[f"{lo}->{hi}"] = float(np.abs(np.mean(np.exp(1j * (2 * a["phase"] - b["phase"])))))
    return out


def spectrum_db(x: np.ndarray, points: int = 160):
    """Smoothed spectrum of a series on the EQ's frequency axis (for the Parametric-style view)."""
    x = np.asarray(x, float)
    n = len(x)
    c = _detrend(x[:, None])[:, 0] * np.hanning(n)
    P = np.abs(np.fft.rfft(c, n=4 * n)) ** 2
    fr = np.fft.rfftfreq(4 * n) * 2 * REF_NYQUIST
    grid = np.geomspace(20, 20000, points)
    keep = fr > 0
    db = 10 * np.log10(np.interp(np.log(grid), np.log(fr[keep]), P[keep]) + 1e-12)
    return grid, db - db.max()


# ---------------------------------------------------------------- lenses

def _detrend(X):
    n = X.shape[0]
    t = np.arange(n)
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def _orbit(X):
    """Quarter turn: each point -> (level, step). Returns the unwrapped angle path, per column."""
    c = _detrend(X)
    u, v = c[1:], np.diff(c, axis=0)
    s = c.std(0) + 1e-12
    return np.unwrap(np.arctan2(v / s, u / s), axis=0)


def lens_regular(X):
    """q1 regular: strongest repeating cycle, as a share of the series' wobble."""
    c = _detrend(X)
    n = c.shape[0]
    w = np.hanning(n)[:, None] if c.ndim == 2 else np.hanning(n)
    m = 4 * n                                    # zero-padded so a cycle between two slices isn't missed
    P = np.abs(np.fft.rfft(c * w, n=m, axis=0)) ** 2
    k = np.arange(P.shape[0])
    per = np.where(k > 0, m / np.maximum(k, 1), np.inf)
    ok = (per >= 2) & (per <= n / 3)
    return P[ok].max(0) / (P[1:].sum(0) + 1e-12)


def lens_quarter(X):
    """q2 quarter turn: how often the orbit turns back on itself mid-curve."""
    return (np.diff(_orbit(X), axis=0) < 0).mean(0)


def lens_superposed(X):
    """q3 |theta|: how hard the orbit spins, direction ignored."""
    return np.abs(np.diff(_orbit(X), axis=0)).mean(0)


def lens_inverse(X):
    """q4 -theta: arrow of time. How lopsided the steps are run forward versus run backward
    (slow climbs and sharp drops score one way, the reverse scores the other).
    The old version compared turn-backs forward and backward, which are always equal, so it was stuck at 0."""
    d = np.diff(_detrend(X), axis=0)
    d = d - d.mean(0)
    sd = d.std(0) + 1e-12
    fwd = (d ** 3).mean(0) / sd ** 3
    return fwd - (-fwd)                      # running time backward flips every step's sign


LENSES = {
    "q1 Regular": dict(fn=lens_regular, sided="up", what="strongest cycle share"),
    "q2 Quarter turn": dict(fn=lens_quarter, sided="both", what="turn-back rate"),
    "q3 Superposed |θ|": dict(fn=lens_superposed, sided="both", what="spin"),
    "q4 Inverse −θ": dict(fn=lens_inverse, sided="both", what="forward vs backward step lopsidedness"),
}


# ---------------------------------------------------------------- fakes

def fakes_for(x: np.ndarray, n_fakes: int, rng) -> np.ndarray:
    """Momentum-matched fakes: same straight-line trend + simulated wobble with the same momentum."""
    n = len(x)
    t = np.arange(n)
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    trend = A @ coef
    a, s = ar_fit(x - trend)
    return trend[:, None] + ar_sim(a, s, n, n_fakes, rng)


def _verdict(real, fs, sided, bar_pct):
    if sided == "up":
        bar = float(np.percentile(fs, bar_pct))
        return bool(real > bar), bar, float(np.mean(fs < real) * 100)
    lo, hi = np.percentile(fs, [(100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2])
    pct = float(np.mean(fs < real) * 100)
    return bool(real < lo or real > hi), (float(lo), float(hi)), pct


def measure(x, curve, lens, n_fakes, rng, bar_pct=99.0):
    """One tunnel step: one series, one EQ state, one lens, scored against EQ'd fakes. Plus a fake arm."""
    L = LENSES[lens]
    F = fakes_for(x, n_fakes + 1, rng)
    Xe = apply_eq(np.asarray(x, float), curve)
    Fe = apply_eq(F, curve)
    real = float(L["fn"](Xe[:, None])[0])
    fs = L["fn"](Fe[:, :n_fakes])
    passed, bar, pct = _verdict(real, fs, L["sided"], bar_pct)
    fake_arm = float(L["fn"](Fe[:, n_fakes:])[0])
    fa_passed = _verdict(fake_arm, fs, L["sided"], bar_pct)[0]
    return dict(stat=round(real, 5), bar=bar, percentile=round(pct, 2), passed=passed,
                fake_arm_passed=bool(fa_passed), sided=L["sided"])


# ---------------------------------------------------------------- Max in the back

MAX_LINES = {
    "power outage": "HEY, it's not like the power's gonna go out or anything.",
    "static": "Good thing nobody's running a blender next to the antenna.",
    "one bad shot": "Glad we never take a bad shot. That'd be crazy.",
    "short day": "We'll definitely have time to finish the whole run.",
    "eq nudge": "Nobody's gonna bump the knobs. Nobody ever bumps the knobs.",
    "frame trip": "Huh. The fake one passed too. That's fine. That's totally fine.",
    "near miss": "So close. Not like that's how every luck-pile story starts.",
    "error": "It's not like a step can just fall over— oh.",
    "flip": "Funny how the EQ moved that one. Wonder which band it was hiding in.",
}


def _jinxes(x, rng, curve):
    n = len(x)
    out = {}
    k = max(2, int(0.15 * n))
    s0 = int(rng.integers(0, n - k))
    out["power outage"] = (np.r_[x[:s0], x[s0 + k:]], curve)
    resid = _detrend(np.asarray(x, float)[:, None])[:, 0]
    out["static"] = (x + rng.normal(0, 0.10 * resid.std() + 1e-12, n), curve)
    st = np.abs(np.diff(x))
    j = int(np.argmax(st)) + 1
    y = np.array(x, float)
    if 0 < j < n - 1:
        y[j] = (y[j - 1] + y[j + 1]) / 2
    out["one bad shot"] = (y, curve)
    out["short day"] = (x[: int(0.75 * n)], curve)
    if not is_flat(curve):
        out["eq nudge"] = (x, {f: [g + (rng.uniform(-0.5, 0.5) if g else 0.0), w] for f, g, w in bells(curve)})
    return out


class Max:
    """Reads the step log as it grows. Keeps his own notes so he never re-reads old steps."""

    def __init__(self, jinx_fakes: int = 200, bar_pct: float = 99.0, seed: int = 4401):
        self.rng = np.random.default_rng(seed)
        self.jinx_fakes, self.bar_pct = jinx_fakes, bar_pct
        self.notes: list[dict] = []
        self.seen = 0

    def note(self, step, kind, text, **extra):
        self.notes.append(dict(step=step, kind=kind, line=MAX_LINES.get(kind, ""), note=text, **extra))

    def watch(self, log: list[dict], series: dict):
        """Look at every step finished since last time."""
        new = log[self.seen:]
        self.seen = len(log)
        for e in new:
            sid = e["step"]
            if e.get("error"):
                self.note(sid, "error", f"{e['dataset']} / {e['lens']} / {e['eq']} fell over: {e['error']}")
                continue
            if e["fake_arm_passed"]:
                self.note(sid, "frame trip", f"{e['dataset']} / {e['lens']} / {e['eq']}: the fake arm passed too, "
                                              "so this tunnel's frame is suspect; a pass here doesn't count.")
            p = e["percentile"]
            near = (e["sided"] == "up" and 97 <= p < 99) or (e["sided"] == "both" and (0.5 < p <= 1.5 or 98.5 <= p < 99.5))
            if near and not e["passed"]:
                self.note(sid, "near miss", f"{e['dataset']} / {e['lens']} / {e['eq']} sat at the {p:.1f}th percentile, just "
                                             "under the bar. Watching it, not counting it.")
            if e["passed"] and not e["fake_arm_passed"]:
                e["max"] = self.jinx(e, series[e["dataset"]])
        return self.notes

    def jinx(self, e, x):
        res = {}
        for name, (y, crv) in _jinxes(np.asarray(x, float), self.rng, e["curve"]).items():
            try:
                r = measure(y, crv, e["lens"], self.jinx_fakes, self.rng, self.bar_pct)
                res[name] = bool(r["passed"])
                self.note(e["step"], name, f"{e['dataset']} / {e['lens']} / {e['eq']}: "
                                           f"{'still passes' if r['passed'] else 'BROKE'} after {name} "
                                           f"(percentile {r['percentile']:.1f}).", survived=bool(r["passed"]))
            except Exception as ex:          # a jinx that can't run is logged, not hidden
                res[name] = None
                self.note(e["step"], "error", f"jinx {name} couldn't run: {ex}")
        broke = [k for k, v in res.items() if v is False]
        return dict(jinxes=res, broke=broke, standing="sturdy (Max couldn't break it)" if not broke else
                    f"luck pile (broke on {', '.join(broke)})")


# ---------------------------------------------------------------- the rotation

def blind(names: list[str], seed: int):
    order = np.random.default_rng(seed).permutation(len(names))
    key = {f"D{i + 1:02d}": names[j] for i, j in enumerate(order)}
    return key


def rotate(series: dict, curve: dict | None = None, n_fakes: int = 500, bar_pct: float = 99.0,
           seed: int = 20261026, max_on: bool = True, on_step=None):
    """series: {name: 1-D array}. curve: EQ dict (None = flat only). Flat always runs as the control.

    Returns dict(key, log, max_notes, survivors, luck_pile, containment, settings)."""
    rng = np.random.default_rng(seed)
    key = blind(list(series), seed)
    blinded = {lab: np.asarray(series[name], float) for lab, name in key.items()}
    eq_states = [("flat", FLAT)] + ([("curve", dict(curve))] if curve and not is_flat(curve) else [])
    log: list[dict] = []
    mx = Max(bar_pct=bar_pct, seed=seed + 1) if max_on else None
    total = len(blinded) * len(eq_states) * len(LENSES)
    for lab, x in blinded.items():
        for eqn, crv in eq_states:
            for lens in LENSES:
                e = dict(step=len(log) + 1, dataset=lab, eq=eqn, lens=lens, curve=crv)
                try:
                    if len(x) < 24 or not np.all(np.isfinite(x)):
                        raise ValueError("needs at least 24 clean points")
                    e.update(measure(x, crv, lens, n_fakes, rng, bar_pct))
                except Exception as ex:
                    e.update(error=str(ex), passed=False, fake_arm_passed=False, percentile=float("nan"), sided="up")
                log.append(e)
                if mx:
                    mx.watch(log, blinded)
                if on_step:
                    on_step(len(log), total, e)
        # flips between flat and curve, per lens: where did the EQ move things?
        if mx and len(eq_states) == 2:
            for lens in LENSES:
                a = [g for g in log if g["dataset"] == lab and g["lens"] == lens]
                if len(a) == 2 and a[0]["passed"] != a[1]["passed"]:
                    mx.note(a[1]["step"], "flip", f"{lab} / {lens}: {'passed flat, lost it' if a[0]['passed'] else 'failed flat, passed'} "
                                                  "with the EQ curve. The pattern lives in (or is hidden by) the bands you moved.")
    tests = len(blinded) * len(eq_states) * len(LENSES)
    strict = 100 - (100 - bar_pct) / (len(eq_states) * len(LENSES))
    survivors, luck = [], []
    for e in log:
        if e["passed"] and not e["fake_arm_passed"]:
            row = dict(dataset=e["dataset"], name=key[e["dataset"]], lens=e["lens"], eq=e["eq"], percentile=e["percentile"])
            m = e.get("max")
            if m and m["broke"]:
                luck.append({**row, "why": m["standing"]})
            else:
                survivors.append({**row, "standing": (m or {}).get("standing", "not jinxed")})
        elif e["passed"] and e["fake_arm_passed"]:
            luck.append(dict(dataset=e["dataset"], name=key[e["dataset"]], lens=e["lens"], eq=e["eq"],
                             percentile=e["percentile"], why="frame trip (fake arm passed)"))
    containment = {}
    for lab in blinded:
        hits = [e for e in log if e["dataset"] == lab and e["passed"] and not e["fake_arm_passed"]]
        strong = [e for e in hits if (e["sided"] == "up" and e["percentile"] >= strict) or
                  (e["sided"] == "both" and (e["percentile"] <= 100 - strict or e["percentile"] >= strict))]
        containment[lab] = dict(name=key[lab], passes=len(hits), passes_after_correction=len(strong))
    for e in log:
        e.pop("curve", None)
    return dict(key=key, log=log, max_notes=mx.notes if mx else [], survivors=survivors, luck_pile=luck,
                containment=containment,
                settings=dict(n_fakes=n_fakes, bar_pct=bar_pct, seed=seed, eq_states=[n for n, _ in eq_states],
                              curve=curve_json(curve or FLAT), lenses=list(LENSES), tests=tests,
                              corrected_bar_pct=round(strict, 4), eq_reference="bands read as audio at 44.1 kHz"))
