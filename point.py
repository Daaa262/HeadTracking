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
        centered_x = projected[0]
        centered_y = -projected[1]

        # Skalowanie z milimetrów na piksele
        scale_x = self.screen_pixels_x / self.screen_width_mm
        scale_y = self.screen_pixels_y / self.screen_height_mm

        pixel_x = int(centered_x * scale_x) + self.screen_pixels_x // 2
        pixel_y = int(centered_y * scale_y) + self.screen_pixels_y // 2

        # wielkosc kropki
        size = max(2, int(self.size * 1000 / np.linalg.norm(direction)))

        # POPRAWKA 4: Sprawdź granice ekranu
        if -size <= pixel_x < self.screen_pixels_x + size and -size <= pixel_y < self.screen_pixels_y + size:
            return np.array([pixel_x, pixel_y, size])
        else:
            return None