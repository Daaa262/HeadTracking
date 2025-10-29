from config import Config

import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_landmarks_name, shm_viewpoint_name):
    shm_landmarks = SharedMemory(name=shm_landmarks_name)
    shared_landmarks = numpy.ndarray(
        shape=(7, 2),
        dtype=numpy.float32,
        buffer=shm_landmarks.buf,
    )

    shm_viewpoint = SharedMemory(name=shm_viewpoint_name)
    shared_viewpoint = numpy.ndarray(
        shape=(3,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )

    x = 0
    last_time = time.time()
    while True:
        success, rotation_vector, translation_vector = cv2.solvePnP(Config.FaceModel.model, shared_landmarks, Config.Camera.matrix, Config.Camera.dist_coefficients, flags=Config.FaceModel.PNPMethod)
        if success:
            shared_viewpoint[:] = translation_vector.flatten()


        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[viewpoint] FPS: {x}")
            x = 0
            last_time = now