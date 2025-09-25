import cv2
import numpy as np

class Point:
    screen_width_mm = 530
    screen_height_mm = 300
    screen_pixels_x = 1920
    screen_pixels_y = 1080
    def __init__(self, position: np.ndarray, color: cv2.typing.Scalar = (100, 100, 100), size: float = 10):
        self.position = position
        self.color = color
        self.size = size

    def project(self, observer: np.ndarray, position: np.ndarray) -> np.ndarray | None:
        direction = self.position - observer - position

        if direction[2] <= 0:
            return None

        t = -observer[2] / direction[2]

        projected = observer + t * direction

        centered_x = projected[0]
        centered_y = -projected[1]

        scale_x = self.screen_pixels_x / self.screen_width_mm
        scale_y = self.screen_pixels_y / self.screen_height_mm

        pixel_x = int(centered_x * scale_x) + self.screen_pixels_x // 2
        pixel_y = int(centered_y * scale_y) + self.screen_pixels_y // 2

        size = max(2, int(self.size * 1000 / np.linalg.norm(direction)))

        if -size <= pixel_x < self.screen_pixels_x + size and -size <= pixel_y < self.screen_pixels_y + size:
            return np.array([pixel_x, pixel_y, size])
        else:
            return None