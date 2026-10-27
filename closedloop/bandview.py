"""Band view charts: Parametric-EQ-style colored bands, and each band spiraling across the dB plane (glow = wide faint line under a thin bright one)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from . import rotation as R

BG = "#141519"


def _label(f: int) -> str:
    return f"{f // 1000}k" if f >= 1000 else str(f)


def _dark(ch):
    return (ch.configure(background=BG).configure_view(strokeWidth=0)
            .configure_axis(gridColor="#2b2e35", domainColor="#3a3e46", labelColor="#c9ccd3", titleColor="#c9ccd3")
            .configure_legend(labelColor="#c9ccd3", titleColor="#c9ccd3", symbolOpacity=1, symbolStrokeWidth=3))


def parametric_chart(x, curve: dict):
    """The series' spectrum behind, each band's bell in its own color, the summed curve glowing on top."""
    grid, spec = R.spectrum_db(x)
    freqs = np.geomspace(20, 20000, 200)
    total = np.zeros_like(freqs)
    rows = []
    for f0, g, w in R.bells(curve):
        bell = g * np.exp(-0.5 * (np.log2(freqs / f0) / w) ** 2)
        total += bell
        if abs(g) > 0.05:
            i = int(np.argmin([abs(np.log2(f0 / b)) for b in R.EQ_BANDS]))
            rows += [dict(hz=f, db=v, band=_label(R.EQ_BANDS[i]), order=i) for f, v in zip(freqs, bell)]
    xs = alt.X("hz:Q", scale=alt.Scale(type="log", domain=[20, 20000]), title="band (Hz, as audio)")
    labels = [_label(f) for f in R.EQ_BANDS]
    color = alt.Color("band:N", scale=alt.Scale(domain=labels, range=R.BAND_COLORS), legend=None)
    sp = alt.Chart(pd.DataFrame(dict(hz=grid, s=spec))).mark_area(color="#6b7280", opacity=0.35).encode(
        xs, alt.Y("s:Q", title="data spectrum (dB)", scale=alt.Scale(domain=[-60, 0])))
    layers = []
    if rows:
        b = pd.DataFrame(rows)
        layers.append(alt.Chart(b).mark_area(opacity=0.22).encode(xs, alt.Y("db:Q", title="EQ (dB)", scale=alt.Scale(domain=[-6, 6])),
                                                                  color, detail="band:N"))
        layers.append(alt.Chart(b).mark_line(strokeWidth=1.2, opacity=0.8).encode(xs, alt.Y("db:Q"), color, detail="band:N"))
    t = pd.DataFrame(dict(hz=freqs, db=total))
    ty = alt.Y("db:Q", title="EQ (dB)", scale=alt.Scale(domain=[-6, 6]))
    layers.append(alt.Chart(t).mark_line(color="#7CFFB2", strokeWidth=9, opacity=0.18).encode(xs, ty))
    layers.append(alt.Chart(t).mark_line(color="#7CFFB2", strokeWidth=2.4).encode(xs, ty))
    eq = alt.layer(*layers)
    return _dark(alt.layer(sp, eq).resolve_scale(y="independent").properties(height=230))


def spiral_chart(bands: dict, by_time: bool = True, max_pts: int = 500):
    """Angle = the band's phase; radius = time (staircase out) or the band's dB level (orbit)."""
    live = {f: b for f, b in bands.items() if b}
    if not live:
        return _dark(alt.Chart(pd.DataFrame(dict(x=[0], y=[0]))).mark_point(opacity=0))
    lo = min(float(np.percentile(b["env_db"], 5)) for b in live.values())
    hi = max(float(b["env_db"].max()) for b in live.values())
    rows = []
    top = max(b["share"] for b in live.values()) + 1e-12
    for f, b in live.items():
        n = len(b["phase"])
        idx = np.unique(np.linspace(0, n - 1, min(n, max_pts)).astype(int))
        db = np.clip((b["env_db"][idx] - lo) / (hi - lo + 1e-12), 0, 1)
        r = 0.12 + idx / max(n - 1, 1) if by_time else 0.12 + db
        # take out each band's own carrier spin, so what's left is how its phase drifts (the spiral)
        fc = f / (2 * R.REF_NYQUIST)                     # band center in cycles per point
        ph = b["phase"][idx] - 2 * np.pi * fc * idx
        w = 0.2 + 0.8 * b["share"] / top                  # bands carrying more of the energy glow brighter
        rows += [dict(x=rr * np.cos(p), y=rr * np.sin(p), t=int(i), band=_label(f), level=float(d), w=w)
                 for rr, p, i, d in zip(r, ph, idx, db)]
    df = pd.DataFrame(rows)
    labels = [_label(f) for f in R.EQ_BANDS]
    color = alt.Color("band:N", scale=alt.Scale(domain=labels, range=R.BAND_COLORS), title="band")
    ax = dict(axis=None, scale=alt.Scale(domain=[-1.15, 1.15] if by_time else [-1.15, 1.15]))
    base = alt.Chart(df).encode(alt.X("x:Q", **ax), alt.Y("y:Q", **ax), color, order="t:Q", detail="band:N")
    glow = base.mark_line(strokeWidth=8).encode(opacity=alt.Opacity("w:Q", scale=alt.Scale(domain=[0, 1], range=[0, 0.3]), legend=None))
    core = base.mark_line(strokeWidth=1.5).encode(opacity=alt.Opacity("w:Q", scale=alt.Scale(domain=[0, 1], range=[0.08, 1]), legend=None))
    return _dark(alt.layer(glow, core).resolve_scale(opacity="independent").properties(height=360, width=360))
