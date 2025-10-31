from config import Config

import random
import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory
from point import Point

def run(shm_dynamic_config_name, shm_viewpoint_name):
    shm_dynamic_config = SharedMemory(name=shm_dynamic_config_name)
    shared_dynamic_config = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_config.buf)

    shm_viewpoint = SharedMemory(name=shm_viewpoint_name)
    shared_viewpoint = numpy.ndarray(
        shape=(3,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )

    points = []
    for i in range(1000):
        color_choice = random.randint(1, 7)
        color = (random.randint(200, 255) * (color_choice & 1), random.randint(200, 255) * (color_choice & 2), random.randint(200, 255) * (color_choice & 4))

        points.append(Point(numpy.array([
            random.randint(-6 * (1000 - i) - 100, 6 * (1000 - i) + 100),
            random.randint(-6 * (1000 - i) - 100, 6 * (1000 - i) + 100),
            9600 - i ** 0.9 * 20
        ]), color, random.randint(0, int(i ** 1.1) // 50 + 1)))

    image = numpy.zeros((Point.screen_pixels_y, Point.screen_pixels_x, 3), dtype=numpy.uint8)

    x = 0
    last_time = time.time()
    while True:
        image.fill(0)

        cv2.putText(image, f"X: {shared_viewpoint[0]:.2f}mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Y: {shared_viewpoint[1]:.2f}mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Z: {shared_viewpoint[2]:.2f}mm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Smoothing Factor: {shared_dynamic_config["smoothing_factor"][0]:.6f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for point in points:
            projected_point = point.project(numpy.array([shared_viewpoint[0], shared_viewpoint[1], shared_viewpoint[2]]))
            if projected_point is not None:
                cv2.circle(image, (projected_point[0], projected_point[1]), projected_point[2], point.color, -1)

        cv2.imshow("debug", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            shared_dynamic_config["smoothing_factor"][0] /= 2
        if key == ord('2'):
            shared_dynamic_config["smoothing_factor"][0] *= 2

        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[debug] FPS: {x}")
            x = 0
            last_time = now