"""Garment/foreground segmentation to isolate clothing before color analysis.

The old color pipeline ran K-Means over the whole image, so background and skin
polluted the dominant colors. This module returns only the foreground (garment)
pixels. It degrades gracefully: rembg (u2net) -> GrabCut (OpenCV) -> center crop,
so the app keeps working even if the heavier deps are missing.
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Fraction of pixels a mask must keep to be considered valid
_MIN_FG_RATIO = 0.05

# Lazy singletons so the model loads once per process
_rembg_session = None
_rembg_disabled = False


def _get_rembg_session():
    """Load the rembg session once; return None if rembg is unavailable."""
    global _rembg_session, _rembg_disabled
    if _rembg_disabled:
        return None
    if _rembg_session is not None:
        return _rembg_session
    try:
        from rembg import new_session

        # u2netp is the lighter variant, enough for garment isolation
        _rembg_session = new_session("u2netp")
        logger.info("Segmentacion: rembg (u2netp) disponible")
        return _rembg_session
    except Exception as e:
        logger.warning(f"Segmentacion: rembg no disponible, usando fallback ({e})")
        _rembg_disabled = True
        return None


def _mask_from_rembg(image: Image.Image):
    """Return a boolean foreground mask via rembg, or None on failure."""
    session = _get_rembg_session()
    if session is None:
        return None
    try:
        from rembg import remove

        alpha = remove(image, session=session, only_mask=True)
        mask = np.array(alpha) > 128
        return mask
    except Exception as e:
        logger.warning(f"Segmentacion rembg fallo: {e}")
        return None


def _mask_from_grabcut(img_array: np.ndarray):
    """Return a boolean foreground mask via OpenCV GrabCut, or None on failure."""
    try:
        import cv2

        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        # Init rectangle: assume the garment sits within the central 80%
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return fg
    except Exception as e:
        logger.warning(f"Segmentacion GrabCut fallo: {e}")
        return None


def _mask_from_center(img_array: np.ndarray):
    """Fallback: keep the central region where the garment usually is."""
    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    top, bottom = int(h * 0.2), int(h * 0.85)
    left, right = int(w * 0.25), int(w * 0.75)
    mask[top:bottom, left:right] = True
    return mask


def get_foreground_pixels(image: Image.Image, size=(300, 300)):
    """Isolate the garment and return its pixels for color analysis.

    Args:
        image: PIL image (any mode).
        size: resize target used for a consistent, fast segmentation.

    Returns:
        (pixels, method) where pixels is an (N, 3) float32 RGB array of the
        garment, and method is the technique that produced the mask.
    """
    image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(image)

    def _valid(m):
        return m is not None and m.sum() >= _MIN_FG_RATIO * m.size

    mask, method = None, "center"
    rembg_mask = _mask_from_rembg(image)
    if _valid(rembg_mask):
        mask, method = rembg_mask, "rembg"
    else:
        grabcut_mask = _mask_from_grabcut(img_array)
        if _valid(grabcut_mask):
            mask, method = grabcut_mask, "grabcut"

    if mask is None:
        mask = _mask_from_center(img_array)

    pixels = img_array[mask].astype(np.float32)
    logger.info(f"Segmentacion: {method}, {len(pixels)} pixeles de prenda")
    return pixels, method
