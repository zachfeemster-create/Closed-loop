"""Slider handles made from the amalgamation mark: the metal parts recolored per band, the red slash kept red."""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image

MARK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "amalgamation_mark.png")

# one metal color per EQ band (low to high), hue in degrees
BAND_HUES = [15, 35, 55, 95, 140, 185, 220, 255, 285, 315]     # same as rotation.BAND_HUES


def _rgb_to_hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m] = np.stack([r[m], g[m], b[m]], -1)
    return out


@lru_cache(maxsize=32)
def thumb_png(hue_deg: float | None, size: int = 44) -> bytes:
    """The mark at `size` px. hue_deg None = original copper."""
    if os.path.exists(MARK):
        src = Image.open(MARK)
    else:
        from .mark import MARK_B64
        src = Image.open(io.BytesIO(base64.b64decode(MARK_B64)))
    im = src.convert("RGBA").resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(float) / 255
    rgb, al = a[..., :3], a[..., 3]
    if hue_deg is not None:
        h, s, v = _rgb_to_hsv(rgb)
        red_slash = (s > 0.62) & ((h < 0.035) | (h > 0.94))
        metal = ~red_slash & (s > 0.12) & (h < 0.16)
        nh = np.where(metal, hue_deg / 360.0, h)
        ns = np.where(metal, np.clip(s * 1.35 + 0.15, 0, 1), s)
        rgb = _hsv_to_rgb(nh, ns, v)
    out = np.dstack([rgb, al]) * 255
    buf = io.BytesIO()
    Image.fromarray(out.astype(np.uint8)).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def thumb_uri(hue_deg: float | None, size: int = 44) -> str:
    return "data:image/png;base64," + base64.b64encode(thumb_png(hue_deg, size)).decode()


def slider_css(keys: list[str], size: int = 40) -> str:
    """CSS that turns the slider handle inside each keyed container into a recolored mark."""
    rules = []
    for i, k in enumerate(keys):
        hue = BAND_HUES[i % len(BAND_HUES)]
        img = f"url('{thumb_uri(hue, size * 2)}') center/contain no-repeat"
        look = (f"width:{size}px!important;height:{size}px!important;background:{img}!important;"
                "border:none!important;box-shadow:none!important;border-radius:0!important;")
        # newer Streamlit: the handle is the div that wraps the hidden range input
        rules.append(f".st-key-{k} [data-testid='stSlider'] div:has(> div > input[type='range'])"
                     f"{{{look}}}")
        # older Streamlit (BaseWeb slider)
        rules.append(f".st-key-{k} div[role='slider']{{{look}}}")
    return "<style>" + "".join(rules) + "</style>"
