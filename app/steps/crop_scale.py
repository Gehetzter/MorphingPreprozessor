import numpy as np

from .load_image import resize_float


def find_bbox(alpha, threshold=0.5):
    ys, xs = np.where(alpha > threshold)

    if len(xs) == 0:
        raise ValueError("Keine Person erkannt")

    return (
        int(xs.min()), int(ys.min()),
        int(xs.max()) + 1, int(ys.max()) + 1,
    )


def crop_and_scale(image, alpha, bbox, canvas_size, margin_percent):
    """Schneidet die Person aus und skaliert sie proportional in die Leinwand.

    Gibt (rgb, alpha, position, scale) zurück.
    """
    left, top, right, bottom = bbox
    canvas_width, canvas_height = canvas_size

    person_width = right - left
    person_height = bottom - top

    margin_x = round(canvas_width * margin_percent / 100)
    margin_y = round(canvas_height * margin_percent / 100)

    available_width = canvas_width - 2 * margin_x
    available_height = canvas_height - 2 * margin_y

    if available_width <= 0 or available_height <= 0:
        raise ValueError("Randabstand zu groß")

    scale = min(
        available_width / person_width,
        available_height / person_height,
    )

    new_size = (
        max(1, round(person_width * scale)),
        max(1, round(person_height * scale)),
    )

    rgb = resize_float(image[top:bottom, left:right], new_size)
    alpha = resize_float(alpha[top:bottom, left:right], new_size)

    position = (
        (canvas_width - new_size[0]) // 2,
        (canvas_height - new_size[1]) // 2,
    )

    return rgb, np.clip(alpha, 0, 1), position, scale
