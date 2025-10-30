from config import Config

import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_frame_name):
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

    if cap.get(cv2.CAP_PROP_FPS) != Config.Camera.fps:
        print("Camera does not support ", Config.Camera.width, "x", Config.Camera.height, " resolution. Set to ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cap.get(cv2.CAP_PROP_FPS) != Config.Camera.fps:
        print("Camera does not support ", cap.get(cv2.CAP_PROP_FPS), " frames per second. Set to ", Config.Camera.fps)

    x = 0
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        shared_frame[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[camera] FPS: {x}")
            x = 0
            last_time = now