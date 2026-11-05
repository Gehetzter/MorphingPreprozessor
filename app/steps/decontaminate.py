from scipy import ndimage


def decontaminate(image, alpha, interior_threshold=0.98):
    """Ersetzt Randpixel durch die Farbe des nächsten sicheren Körperpixels.

    Entfernt Farbreste des alten Hintergrunds (Saum um die Person).
    """
    interior = alpha >= interior_threshold

    if not interior.any():
        return image

    _, (iy, ix) = ndimage.distance_transform_edt(
        ~interior, return_indices=True
    )

    result = image.copy()
    edge = ~interior
    result[edge] = image[iy[edge], ix[edge]]

    return result
