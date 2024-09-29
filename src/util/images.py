import logging
from typing import Tuple

import cv2

logger = logging.getLogger(__name__)


def resize_image(image, max_width, max_height):
    """
    Resize the image while preserving aspect ratio, ensuring that the image
    does not exceed the maximum dimensions specified. For downscaling,
    we use a method that retains as much quality as possible.

    Args:
        image (numpy.ndarray): Input image to be resized.
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.

    Returns:
        numpy.ndarray: Resized image.
    """
    img_height, img_width = image.shape[:2]
    logger.info(f"Original image size: {img_width}x{img_height}")

    # Calculate the scaling factor for width and height
    width_scale = max_width / img_width
    height_scale = max_height / img_height

    # Choose the smallest scaling factor to maintain aspect ratio and fit within the box
    scale = min(width_scale, height_scale, 1)  # Ensure we don't upscale for now

    # Compute the new dimensions
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    # Resize the image using high-quality downscaling
    if scale < 1:
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image  # If the image is smaller, we keep the original size for now

    logger.info(f"Resized image size: {new_width}x{new_height}")
    return resized_image


def ai_upscaler(image, target_dims: Tuple[int, int]):
    """
    Upscale the image using a state-of-the-art AI model.
    """
    pass
