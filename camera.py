from config import Config
import os
import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def run(shm_dynamic_data_name, shm_frame_name, lock_frame):
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_data.buf)

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

    frames = 0
    last_time = time.time()
    try:
        while shared_dynamic_data['running_flag'][0]:
            frames += 1
            now = time.time()
            if now - last_time >= 1:
                shared_dynamic_data["camera_fps"][0] = frames
                frames = 0
                last_time = now

            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.1)
                continue

            with lock_frame:
                shared_frame[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    finally:
        cap.release()
        shm_dynamic_data.close()
        shm_frame.close()