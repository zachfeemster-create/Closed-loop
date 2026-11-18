import os

# The app expects a logo file. If the assets folder wasn't uploaded,
# rebuild the real Closed Loop coin from the copy stored as text in logo.py.
_logo = os.path.join("assets", "closed_loop_coin_256.png")
if not os.path.exists(_logo):
    try:
        import base64
        import io
        from PIL import Image
        from .logo import LOGO_WEBP_B64
        os.makedirs("assets", exist_ok=True)
        _img = Image.open(io.BytesIO(base64.b64decode(LOGO_WEBP_B64))).convert("RGBA")
        _img.resize((256, 256), Image.LANCZOS).save(_logo)
    except Exception:
        pass

# Every page uses the coin as its tab icon. Pages import closedloop before they call
# st.set_page_config, so swap their emoji icon for the coin here (the emoji stays as the fallback).
try:
    import base64 as _b64
    import io as _io

    import streamlit as _st
    from PIL import Image as _Image

    from .logo import LOGO_WEBP_B64 as _B64
    _COIN = _Image.open(_io.BytesIO(_b64.b64decode(_B64)))
    if not getattr(_st.set_page_config, "_coin", False):
        _orig = _st.set_page_config

        def _with_coin(*a, **k):
            k["page_icon"] = _COIN
            return _orig(*a, **k)

        _with_coin._coin = True
        _st.set_page_config = _with_coin
except Exception:
    pass
