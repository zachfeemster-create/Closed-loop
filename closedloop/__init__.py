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
