"""Drag-the-sigil EQ: ten band handles on one glowing curve, each handle the amalgamation mark in its band color."""
from __future__ import annotations

import os

import streamlit.components.v1 as components

from . import rotation as R
from . import thumbs as T

_eq = components.declare_component(
    "eq_curve", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "eq_curve"))


def _label(f: int) -> str:
    return f"{f // 1000}k" if f >= 1000 else str(f)


def eq_curve(values: list[float], key: str, disabled: bool = False, lo: float = -6.0, hi: float = 6.0,
             size: int = 34) -> list[float]:
    """Shows the curve and returns the ten band values in dB (unchanged until a handle is let go)."""
    sigils = [T.thumb_uri(h, size * 2) for h in R.BAND_HUES]
    out = _eq(values=[round(float(v), 2) for v in values], labels=[_label(f) for f in R.EQ_BANDS],
              colors=R.BAND_COLORS, sigils=sigils, min=lo, max=hi, size=size, disabled=disabled,
              key=key, default=None)
    return [float(v) for v in out] if out else [float(v) for v in values]
