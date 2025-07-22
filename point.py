import cv2
import numpy as np

class Point:
    def __init__(self, position: np.ndarray, color: cv2.typing.Scalar = (100, 100, 100), size: float = 10):
        self.screen_width_mm = 530
        self.screen_height_mm = 300
        self.screen_pixels_x = 1920
        self.screen_pixels_y = 1080
        self.position = position
        self.color = color
        self.size = size

    def project(self, observer: np.ndarray) -> np.ndarray | None:
        # POPRAWKA 1: Kierunek to punkt MINUS obserwator
        direction = self.position - observer

        # POPRAWKA 2: Sprawdź czy punkt jest za obserwatorem
        if direction[2] <= 0:
            return None  # punkt za kamerą lub w tej samej płaszczyźnie

        # Parametr t dla przecięcia z płaszczyzną Z=0 (monitor)
        t = -observer[2] / direction[2]

        if t <= 0:
            return None  # nie powinno się zdarzyć, ale dla bezpieczeństwa

        # Położenie na monitorze
        projected = observer + t * direction

        # POPRAWKA 3: Centrowanie - (0,0) ma być na środku ekranu
        centered_x = projected[0] + self.screen_width_mm / 2
        centered_y = -projected[1] + self.screen_height_mm / 2  # odwrócenie Y

        # Skalowanie z milimetrów na piksele
        scale_x = self.screen_pixels_x / self.screen_width_mm
        scale_y = self.screen_pixels_y / self.screen_height_mm

        pixel_x = int(centered_x * scale_x)
        pixel_y = int(centered_y * scale_y)

        # POPRAWKA 4: Sprawdź granice ekranu
        if 0 <= pixel_x < self.screen_pixels_x and 0 <= pixel_y < self.screen_pixels_y:
            return np.array([pixel_x, pixel_y, self.size])
        else:
            return None