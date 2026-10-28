import cv2
import numpy as np
import io
import logging

LOGGER = logging.getLogger(__name__)

class ImageValidator:
    """Validates if an uploaded image is likely a medical MRI scan."""

    @staticmethod
    def is_valid_mri(image_bytes: bytes) -> bool:
        """
        Check if the image is a valid brain MRI.
        Uses heuristics like color saturation and background darkness.
        """
        try:
            if not image_bytes:
                return False

            # Decode the image with OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Fallback to PIL if OpenCV fails to decode (e.g., unusual formats or color profiles)
            if image is None:
                try:
                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as pil_err:
                    LOGGER.warning("PIL fallback decode failed: %s", pil_err)

            if image is None:
                LOGGER.warning("Image could not be decoded.")
                return False

            height, width = image.shape[:2]
            if height < 32 or width < 32:
                LOGGER.warning("Image dimensions too small: %dx%d", width, height)
                return False

            # Check 1: Grayscale / Low Color Saturation on meaningful pixels
            # MRIs are grayscale scans. Colored images (e.g. photos, cartoons, documents)
            # have high saturation and high variance across color channels.
            # In dark background pixels, JPEG compression artifacts inflate HSV saturation.
            # Therefore, evaluate saturation and channel std only on pixels above dark threshold.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            val = hsv[:, :, 2]
            sat = hsv[:, :, 1]
            bright_pixels = val > 25

            if not np.any(bright_pixels):
                LOGGER.warning("Image is completely dark/blank.")
                return False

            bright_sat = float(np.mean(sat[bright_pixels]))
            channel_std = float(np.mean(np.std(image.astype(np.float32), axis=2)[bright_pixels]))

            # Normal photos have high saturation (> 65) or high channel deviation (> 20)
            if bright_sat > 65.0 or channel_std > 20.0:
                LOGGER.warning(
                    "Image failed color/saturation check: bright_sat=%.2f, channel_std=%.2f",
                    bright_sat, channel_std
                )
                return False

            # Check 2: Background Darkness or Central Brain Structure
            # Typical brain MRIs feature dark outer borders / corners (air surrounding skull)
            cw = max(5, min(15, width // 10))
            ch = max(5, min(15, height // 10))
            corners = [
                image[0:ch, 0:cw],
                image[0:ch, width - cw:width],
                image[height - ch:height, 0:cw],
                image[height - ch:height, width - cw:width]
            ]
            corner_means = [float(np.mean(corner)) for corner in corners]
            dark_corners = sum(1 for m in corner_means if m < 65)

            if dark_corners < 2:
                # Cropped MRIs may not have fully black corners; verify central tissue contrast
                center = image[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]
                edge_mask = np.ones((height, width), dtype=bool)
                edge_mask[int(height * 0.15):int(height * 0.85), int(width * 0.15):int(width * 0.85)] = False

                center_mean = float(np.mean(center))
                edges_mean = float(np.mean(image[edge_mask]))

                if center_mean <= edges_mean + 5:
                    LOGGER.warning(
                        "Image failed structural check: dark_corners=%d, center_mean=%.2f, edges_mean=%.2f",
                        dark_corners, center_mean, edges_mean
                    )
                    return False

            # Check 3: Aspect Ratio
            aspect_ratio = width / float(height)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                LOGGER.warning("Image failed aspect ratio check: %.2f", aspect_ratio)
                return False

            return True

        except Exception as e:
            LOGGER.error("Image validation exception: %s", e)
            return False
