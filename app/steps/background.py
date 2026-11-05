import io

import numpy as np
from PIL import Image, ImageOps


def hex_to_rgb(color):
    color = color.lstrip("#")

    if len(color) != 6:
        raise ValueError("Ungültige Hintergrundfarbe")

    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def load_background_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def make_background(canvas_size, mode, color="#D3D3D3", image=None, fallback=None):
    """Erzeugt die Leinwand (PIL RGB).

    mode: "color" | "image" | "none" (fallback = Randfarbe des Originals)
    """
    if mode == "image":
        if image is None:
            raise ValueError("Kein Hintergrundbild gewählt")

        return ImageOps.fit(image, canvas_size, Image.Resampling.LANCZOS)

    if mode == "none":
        return Image.new("RGB", canvas_size, fallback or (128, 128, 128))

    return Image.new("RGB", canvas_size, hex_to_rgb(color))


def composite(canvas, rgb, alpha, position):
    person = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    mask = Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8))

    canvas.paste(person, position, mask)

    return canvas
