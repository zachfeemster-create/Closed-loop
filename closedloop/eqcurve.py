"""Drag-the-sigil EQ: ten band handles on one glowing curve, each handle the amalgamation mark in its band color.
Drag up or down for gain; twist two fingers (or tap the quarter-turn buttons) to shape the band: every quarter turn
doubles or halves its width."""
from __future__ import annotations

import os

import streamlit.components.v1 as components

from . import rotation as R
from . import thumbs as T

_eq = components.declare_component(
    "eq_curve", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "eq_curve"))


def _label(f: int) -> str:
    return f"{f // 1000}k" if f >= 1000 else str(f)


def _pair(v) -> list[float]:
    return [float(v[0]), float(v[1])] if isinstance(v, (list, tuple)) else [float(v), R.EQ_WIDTH_OCT]


def _full(v, i: int) -> list[float]:
    """[gain, width, position (band steps), kind (0 bell, 1 low shelf, 2 high shelf)]."""
    if isinstance(v, (list, tuple)):
        return [float(v[0]), float(v[1]) if len(v) > 1 else R.EQ_WIDTH_OCT,
                float(v[2]) if len(v) > 2 else float(i), int(v[3]) if len(v) > 3 else 0]
    return [float(v), R.EQ_WIDTH_OCT, float(i), 0]


def eq_curve(values: list, key: str, disabled: bool = False, lo: float = -6.0, hi: float = 6.0,
             size: int = 34, slide: bool = False, shelves: bool = False, unit: str = "") -> list[list[float]]:
    """Shows the curve; values are gains or [gain, width] per band. Returns [[gain, width], ...] for the ten bands
    (unchanged until a handle is let go)."""
    sigils = [T.thumb_uri(h, size * 2) for h in R.BAND_HUES]
    if slide or shelves:      # moved / shelf bands: [gain, width, position, kind]
        vals = [[round(a, 2), round(b, 3), round(c, 3), k] for a, b, c, k in (_full(v, i) for i, v in enumerate(values))]
    else:
        vals = [[round(g, 2), round(w, 3)] for g, w in map(_pair, values)]
    out = _eq(values=vals, labels=[_label(f) for f in R.EQ_BANDS], colors=R.BAND_COLORS, sigils=sigils,
              min=lo, max=hi, size=size, disabled=disabled, key=key, default=None,
              slide=slide, shelves=shelves, unit=unit, base=31.25)
    if slide or shelves:
        return [_full(v, i) for i, v in enumerate(out)] if out else vals
    return [_pair(v) for v in out] if out else vals


def pos_hz(c: float, i: int | None = None) -> float:
    """Where a band sits, in reference Hz, from its position in band steps (0 = 31, 9 = 16k, one step = one octave).
    A band that hasn't been moved keeps its exact ladder value."""
    if i is not None and abs(c - i) < 1e-6:
        return float(R.EQ_BANDS[i])
    return 31.25 * 2 ** float(c)


def as_curve(values: list) -> dict:
    """[[gain, width], ...] (or [gain, width, position, kind] once bands can move) -> a curve dict the rotation
    understands, keyed by each band's center; a shelf carries its kind as a third number."""
    out = {}
    for i, (f, v) in enumerate(zip(R.EQ_BANDS, values)):
        if isinstance(v, (list, tuple)) and len(v) > 2:
            k = int(v[3]) if len(v) > 3 else 0
            out[pos_hz(float(v[2]), i)] = [float(v[0]), float(v[1])] + ([k] if k else [])
        else:
            out[f] = _pair(v)
    return out
