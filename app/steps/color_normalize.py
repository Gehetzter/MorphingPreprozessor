import numpy as np
from scipy import ndimage

GAIN_MIN = 0.4
GAIN_MAX = 3.0


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _skin_mask(rgb):
    """Einfache Hautfarben-Erkennung im YCbCr-Raum."""
    r, g, b = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)


def body_color(rgb, alpha):
    """Robuste Körperfarbe (Median, linear RGB) aus den Hautpixeln der Person."""
    inner = ndimage.binary_erosion(alpha > 0.99, iterations=3)

    if inner.sum() < 100:
        inner = alpha > 0.5

    skin = inner & _skin_mask(rgb)

    # Zu wenig erkannte Haut: gesamte Person verwenden
    selected = skin if skin.sum() > 0.01 * inner.sum() else inner

    return np.median(srgb_to_linear(rgb[selected]), axis=0)


def average_color(colors):
    return np.mean(np.stack(colors), axis=0)


def apply_gain(rgb, current, target):
    """Gleicht Körperfarbe und Helligkeit per Kanalverstärkung im linearen Raum an."""
    gain = np.clip(target / np.maximum(current, 1e-4), GAIN_MIN, GAIN_MAX)

    return linear_to_srgb(srgb_to_linear(rgb) * gain).astype(np.float32)
