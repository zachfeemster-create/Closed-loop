import os

# The app expects a logo file. If the assets folder wasn't uploaded,
# make a plain stand-in so the app starts instead of crashing.
_logo = os.path.join("assets", "closed_loop_coin_256.png")
if not os.path.exists(_logo):
    try:
        from PIL import Image, ImageDraw
        os.makedirs("assets", exist_ok=True)
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((8, 8, 248, 248), fill=(201, 162, 39, 255), outline=(120, 90, 20, 255), width=8)
        d.ellipse((70, 70, 186, 186), outline=(60, 45, 10, 255), width=14)
        img.save(_logo)
    except Exception:
        pass
