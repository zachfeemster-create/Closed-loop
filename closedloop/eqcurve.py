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


def eq_curve(values: list, key: str, disabled: bool = False, lo: float = -6.0, hi: float = 6.0,
             size: int = 34) -> list[list[float]]:
    """Shows the curve; values are gains or [gain, width] per band. Returns [[gain, width], ...] for the ten bands
    (unchanged until a handle is let go)."""
    sigils = [T.thumb_uri(h, size * 2) for h in R.BAND_HUES]
    vals = [[round(g, 2), round(w, 3)] for g, w in map(_pair, values)]
    out = _eq(values=vals, labels=[_label(f) for f in R.EQ_BANDS], colors=R.BAND_COLORS, sigils=sigils,
              min=lo, max=hi, size=size, disabled=disabled, key=key, default=None)
    return [_pair(v) for v in out] if out else vals


def as_curve(values: list) -> dict:
    """[[gain, width], ...] for the ten bands -> a curve dict the rotation understands."""
    return {f: _pair(v) for f, v in zip(R.EQ_BANDS, values)}
