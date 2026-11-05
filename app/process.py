"""Prozessablauf: verbindet die einzelnen Schritte aus dem Paket `steps`."""

import time
from dataclasses import dataclass, field

import numpy as np

from log import get_logger

from steps.background import composite, make_background
from steps.color_normalize import apply_gain, average_color, body_color
from steps.crop_scale import crop_and_scale, find_bbox
from steps.decontaminate import decontaminate
from steps.load_image import load_image
from steps.refine_mask import refine_mask
from steps.segment import segment


log = get_logger("bodymorph.process")


@dataclass
class Options:
    target_height: int = 1200
    margin_percent: float = 5
    background_mode: str = "color"  # "color" | "image" | "transparent" | "none"
    edge_softness: float = 3.0  # Kantenweichzeichnung in px (bei ~1024 px Segmentierung)
    edge_erode: int = 2  # Erosion der Maske in px
    background_color: str = "#D3D3D3"
    background_image: object = None  # PIL-Bild
    normalize_color: bool = True
    reference_index: int = None  # None = Durchschnitt aller Bilder


@dataclass
class Prepared:
    rgb: np.ndarray
    alpha: np.ndarray  # Alpha für das Freistellen
    cutout: bool
    position: tuple
    canvas_size: tuple
    color: np.ndarray
    metadata: dict = field(default_factory=dict)


def prepare_image(image_bytes, filename, options, session):
    """Schritte 1-5: laden, freistellen, bereinigen, zuschneiden, skalieren."""
    image = load_image(image_bytes, filename)
    height, width = image.shape[:2]

    canvas_height = int(options.target_height)
    canvas_width = round(width / height * canvas_height)

    raw_mask = segment(image, session)
    alpha = refine_mask(
        raw_mask,
        output_size=(width, height),
        feather_px=options.edge_softness,
        erode_px=options.edge_erode,
    )

    bbox = find_bbox(alpha)

    cutout = options.background_mode != "none"

    if cutout:
        image = decontaminate(image, alpha)

    rgb, scaled_alpha, position, scale = crop_and_scale(
        image, alpha, bbox, (canvas_width, canvas_height),
        options.margin_percent,
    )

    return Prepared(
        rgb=rgb,
        alpha=scaled_alpha,
        cutout=cutout,
        position=position,
        canvas_size=(canvas_width, canvas_height),
        color=body_color(rgb, scaled_alpha),
        metadata={
            "original_size": (width, height),
            "bounding_box": bbox,
            "output_size": (canvas_width, canvas_height),
            "scaled_size": (rgb.shape[1], rgb.shape[0]),
            "scale": scale,
            "position": position,
        },
    )


def render_image(prepared, options, target_color):
    """Schritte 6-7: Farbe angleichen, auf Hintergrund setzen."""
    rgb = prepared.rgb

    if options.normalize_color and target_color is not None:
        rgb = apply_gain(rgb, prepared.color, target_color)

    fallback = None

    if prepared.cutout:
        alpha = prepared.alpha
    else:
        # Ohne Hintergrundentfernung: Originalausschnitt vollständig übernehmen
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        border = np.concatenate([
            rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]
        ])
        fallback = tuple(
            int(v) for v in np.clip(border.mean(axis=0) * 255, 0, 255)
        )

    canvas = make_background(
        prepared.canvas_size,
        options.background_mode,
        color=options.background_color,
        image=options.background_image,
        fallback=fallback,
    )

    return composite(canvas, rgb, alpha, prepared.position)


def process_batch(files, options, session, progress=None):
    """Verarbeitet [(name, bytes), ...].

    Gibt (results, errors) zurück; results = [(index, image, metadata), ...].
    Die Zielfarbe ist der Durchschnitt aller Bilder oder die gewählte Referenz.
    """
    prepared = {}
    errors = []

    total = len(files) * 2
    batch_start = time.monotonic()

    log.info(
        "Batch gestartet: %d Bilder, Hintergrund=%s, Höhe=%d, "
        "Erosion=%s, Weichheit=%s, Farbangleich=%s, Referenz=%s",
        len(files), options.background_mode, options.target_height,
        options.edge_erode, options.edge_softness,
        options.normalize_color, options.reference_index,
    )

    for index, (name, data) in enumerate(files):
        start = time.monotonic()

        try:
            prepared[index] = prepare_image(data, name, options, session)
            log.info(
                "Freigestellt: %s (%d KB, %.1fs)",
                name, len(data) // 1024, time.monotonic() - start,
            )
        except Exception as exc:
            log.exception("Fehler beim Freistellen von %s", name)
            errors.append(f"{name}: {exc}")

        if progress:
            progress((index + 1) / total, f"Freistellen: {name}")

    target = None

    if options.normalize_color and prepared:
        reference = prepared.get(options.reference_index)

        if reference is not None:
            target = reference.color
        else:
            target = average_color([p.color for p in prepared.values()])

        log.info("Zielfarbe (linear RGB): %s", np.round(target, 4).tolist())

    results = []

    for step, (index, item) in enumerate(prepared.items()):
        name = files[index][0]

        try:
            image = render_image(item, options, target)
            results.append((index, image, item.metadata))
        except Exception as exc:
            log.exception("Fehler beim Rendern von %s", name)
            errors.append(f"{name}: {exc}")

        if progress:
            progress(
                (len(files) + step + 1) / total,
                f"Farbangleich: {name}",
            )

    log.info(
        "Batch beendet: %d ok, %d Fehler, %.1fs",
        len(results), len(errors), time.monotonic() - batch_start,
    )

    return results, errors
