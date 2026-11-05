import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf",
    ".rw2", ".raf", ".pef", ".srw", ".raw",
}

MAX_SIDE = 3000


def is_raw(filename):
    return Path(filename).suffix.lower() in RAW_EXTENSIONS


def _load_raw(image_bytes):
    import rawpy

    with rawpy.imread(io.BytesIO(image_bytes)) as raw:
        # Feste Belichtung (keine Auto-Helligkeit), Kamera-Weißabgleich, 16 Bit
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
        )

    return rgb16.astype(np.float32) / 65535.0


def _load_standard(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_float(array, size):
    """Skaliert ein float-Bild (H, W[, C]) auf size=(Breite, Höhe)."""
    if array.ndim == 2:
        channel = Image.fromarray(array.astype(np.float32), mode="F")
        resized = channel.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32)

    channels = [
        resize_float(array[..., i], size)
        for i in range(array.shape[2])
    ]

    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def load_image(image_bytes, filename=""):
    """Lädt JPG/PNG/WebP oder RAW als float32-sRGB-Array (H, W, 3) in 0..1."""
    if is_raw(filename):
        array = _load_raw(image_bytes)
    else:
        array = _load_standard(image_bytes)

    height, width = array.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError("Ungültige Bildgröße")

    longest = max(width, height)

    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        array = resize_float(
            array,
            (round(width * scale), round(height * scale)),
        )

    return array
