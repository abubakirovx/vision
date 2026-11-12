"""Validate images and decode in memory; no uploads or frames reach disk."""
import base64
import binascii
import io

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from .config import MAX_IMAGE_BYTES


def decode_image(encoded: str) -> np.ndarray:
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or header not in ("data:image/jpeg;base64", "data:image/png;base64"):
            raise HTTPException(422, "Use a JPEG or PNG image.")
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise HTTPException(413, "Image exceeds the 5 MB limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "Invalid base64 image.") from None
    if not raw:
        raise HTTPException(422, "Image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 5 MB limit.")
    try:
        # Check dimensions before OpenCV allocates the decoded frame.
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise HTTPException(422, "Use a JPEG or PNG image.")
            if max(image.size) > 4096 or min(image.size) < 16:
                raise HTTPException(422, "Image dimensions must be between 16 and 4096 pixels.")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(422, "The image could not be decoded.") from None
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "The image could not be decoded.")
    height, width = frame.shape[:2]
    if max(height, width) > 1280:
        scale = 1280 / max(height, width)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return frame


def encode_jpeg(frame: np.ndarray) -> str:
    success, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(422, "The image could not be encoded.")
    return base64.b64encode(image).decode("ascii")

