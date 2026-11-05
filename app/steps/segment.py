import numpy as np
from PIL import Image
from rembg import remove

SEGMENT_SIDE = 1024


def segment(image, session):
    """Liefert eine weiche Personenmaske (float 0..1) auf verkleinerter Bildgröße."""
    height, width = image.shape[:2]

    pil = Image.fromarray(
        (np.clip(image, 0, 1) * 255).astype(np.uint8)
    )

    scale = min(1.0, SEGMENT_SIDE / max(width, height))

    if scale < 1.0:
        pil = pil.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )

    mask = remove(pil, session=session, only_mask=True)

    return np.asarray(mask, dtype=np.float32) / 255.0
