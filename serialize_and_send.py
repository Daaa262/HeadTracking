from config import Config

import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_smoothed_viewpoint_name):
    shm_smoothed_viewpoint = SharedMemory(name=shm_smoothed_viewpoint_name)
    shared_smoothed_viewpoint = numpy.ndarray(
        shape=(3,),
        dtype=numpy.float32,
        buffer=shm_smoothed_viewpoint.buf
    )

    x = 0
    last_time = time.time()
    while True:
        frame = numpy.zeros((Config.Screen.height, Config.Screen.width, 3), numpy.uint8)
        x, y, z = shared_smoothed_viewpoint[0], shared_smoothed_viewpoint[1], shared_smoothed_viewpoint[2]

        frame.fill(0)

        cv2.putText(frame, f"X: {x:.1f}mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Y: {y:.1f}mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Z: {z:.1f}mm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[serialize_and_send] FPS: {x}")
            x = 0
            last_time = now