from config import Config

import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_dynamic_config_name, shm_frame_name):
    shm_dynamic_config = SharedMemory(name=shm_dynamic_config_name)
    shared_dynamic_config = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_config.buf)

    shm_frame = SharedMemory(name=shm_frame_name)
    shared_frame = numpy.ndarray(
        shape=(Config.Camera.height, Config.Camera.width, 3),
        dtype=numpy.uint8,
        buffer=shm_frame.buf
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.Camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.Camera.height)
    cap.set(cv2.CAP_PROP_FPS, Config.Camera.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.Camera.buffer_size)

    x = 0
    last_time = time.time()
    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[camera] Frame capture failed.")
                time.sleep(0.1)
                continue

            shared_frame[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if shared_dynamic_config["debug_mode"][0] & 2:
                x += 1
                now = time.time()
                if now - last_time >= 1:
                    print(f"[camera] FPS: {x}")
                    x = 0
                    last_time = now
    finally:
        cap.release()
        shm_frame.close()
        shm_dynamic_config.close()