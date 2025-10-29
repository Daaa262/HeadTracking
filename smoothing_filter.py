from config import Config

import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_viewpoint_name, shm_smoothed_viewpoint_name):
    shm_viewpoint = SharedMemory(name=shm_viewpoint_name)
    shared_viewpoint = numpy.ndarray(
        shape=(3,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )

    shm_smoothed_viewpoint = SharedMemory(name=shm_smoothed_viewpoint_name)
    shared_smoothed_viewpoint = numpy.ndarray(
        shape=(3,),
        dtype=numpy.float32,
        buffer=shm_smoothed_viewpoint.buf
    )

    previous_viewpoint = shared_viewpoint

    x = 0
    last_time = time.time()
    while True:
        distance = numpy.linalg.norm(shared_viewpoint - previous_viewpoint)
        alpha = 1 - numpy.exp(-Config.SmoothingFilter.factor * distance)
        previous_viewpoint = alpha * shared_viewpoint + (1 - alpha) * previous_viewpoint
        shared_smoothed_viewpoint[:] = previous_viewpoint


        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[smoothing_filter] FPS: {x}")
            x = 0
            last_time = now