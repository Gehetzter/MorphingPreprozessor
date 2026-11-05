
import io
from PIL import Image, ImageOps
from rembg import remove


def hex_to_rgb(color):
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError("Ungültige Hintergrundfarbe")

    return tuple(
        int(color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def process_image(
    image_bytes,
    target_height=1200,
    margin_percent=5,
    background="#D3D3D3",
    session=None,
):
    # Bild laden und EXIF-Orientierung korrigieren
    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Ungültige Bildgröße")

    # Person segmentieren
    segmented = remove(
        image,
        session=session
    ).convert("RGBA")

    # Alpha-Maske der Person
    alpha = segmented.getchannel("A")

    # Kleine Restpixel ignorieren
    mask = alpha.point(
        lambda p: 255 if p > 10 else 0
    )

    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("Keine Person erkannt")

    left, top, right, bottom = bbox

    person_width = right - left
    person_height = bottom - top

    if person_width <= 0 or person_height <= 0:
        raise ValueError("Ungültige Personen-Bounding-Box")

    # Person ausschneiden
    person = segmented.crop(bbox)

    # Zielbildhöhe und ursprüngliches Seitenverhältnis
    canvas_height = int(target_height)

    canvas_width = round(
        original_width / original_height * canvas_height
    )

    # Sicherheitsrand
    margin_x = round(
        canvas_width * margin_percent / 100
    )

    margin_y = round(
        canvas_height * margin_percent / 100
    )

    available_width = canvas_width - 2 * margin_x
    available_height = canvas_height - 2 * margin_y

    if available_width <= 0 or available_height <= 0:
        raise ValueError("Randabstand zu groß")

    # Proportionale Skalierung
    scale = min(
        available_width / person_width,
        available_height / person_height
    )

    new_width = max(1, round(person_width * scale))
    new_height = max(1, round(person_height * scale))

    person = person.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Bounding Box zentrieren
    x = (canvas_width - new_width) // 2
    y = (canvas_height - new_height) // 2

    # Neutralgraue Leinwand
    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        hex_to_rgb(background)
    )

    canvas.paste(
        person,
        (x, y),
        person.getchannel("A")
    )

    metadata = {
        "original_size": (original_width, original_height),
        "bounding_box": bbox,
        "person_size": (person_width, person_height),
        "output_size": (canvas_width, canvas_height),
        "scaled_size": (new_width, new_height),
        "scale": scale,
        "position": (x, y),
    }

    return canvas, metadata
