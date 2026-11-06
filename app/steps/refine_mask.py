import numpy as np
from scipy import ndimage

from .load_image import resize_float


def _fill_small_holes(binary, max_fraction=0.02):
    """Füllt Löcher im Körper, lässt große Lücken (z. B. zwischen Arm und Rumpf) offen."""
    holes = ~binary
    labels, count = ndimage.label(holes)

    if count == 0:
        return binary

    sizes = ndimage.sum(holes, labels, index=np.arange(1, count + 1))
    limit = binary.sum() * max_fraction

    # Hintergrund außen (berührt den Rand) nie füllen
    border = set(
        np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ]).tolist()
    )

    fill = np.zeros(count + 1, dtype=bool)

    for label, size in enumerate(sizes, start=1):
        if size <= limit and label not in border:
            fill[label] = True

    return binary | fill[labels]


def refine_mask(
    mask,
    output_size,
    threshold=0.4,
    erode_px=0,
    feather_px=0.0,
):
    """Bereinigt die Rohmaske; gibt weiche Alpha-Maske in output_size=(B, H) zurück.

    - niedrige Schwelle: halbtransparente Körperteile bleiben Teil der Person
    - größte Komponente: Restflecken verschwinden
    - kleine Löcher werden gefüllt
    - optionale Erosion: entfernt den Saum aus altem Hintergrund
    """
    binary = mask > threshold

    labels, count = ndimage.label(binary)

    if count == 0:
        raise ValueError("Keine Person erkannt")

    sizes = ndimage.sum(binary, labels, index=np.arange(1, count + 1))
    binary = labels == (int(np.argmax(sizes)) + 1)

    binary = ndimage.binary_closing(binary, iterations=2)
    binary = _fill_small_holes(binary)

    # Vorzeichenbehafteter Abstand zur Kante (innen positiv, in Pixeln der
    # Rohmaske). Er ist glatt und lässt sich ohne Treppenstufen hochskalieren.
    sdf = (
        ndimage.distance_transform_edt(binary)
        - ndimage.distance_transform_edt(~binary)
    ).astype(np.float32)
    sdf = ndimage.gaussian_filter(sdf, sigma=1.0)

    if sdf.shape[::-1] != tuple(output_size):
        sdf = resize_float(sdf, tuple(output_size))

    # Die Kante wird erst in voller Auflösung gelegt: Erosion verschiebt sie,
    # die Weichheit bestimmt die Breite des Übergangs (mind. ein Ausgabepixel).
    scale = output_size[0] / mask.shape[1]
    width = max(2.0 * feather_px, 1.0 / scale)

    alpha = 0.5 + (sdf - erode_px) / width

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
