from config import Config

import json
import cv2
import numpy
import time
import socket
from multiprocessing.shared_memory import SharedMemory

def run(shm_dynamic_config_name, shm_landmarks_name, shm_viewpoint_name):
    shm_dynamic_config = SharedMemory(name=shm_dynamic_config_name)
    shared_dynamic_config = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_config.buf)

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

    live_viewpoint = shared_viewpoint.copy()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    x = 0
    last_time = time.time()
    while True:
        success, rotation_vector, translation_vector = cv2.solvePnP(Config.FaceModel.model, shared_landmarks, Config.Camera.matrix, Config.Camera.dist_coefficients, flags=Config.FaceModel.PNPMethod)
        if success:
            live_viewpoint = translation_vector.flatten()
            live_viewpoint[0] = -live_viewpoint[0] + Config.Camera.position_relative_to_monitor_center_x_mm
            live_viewpoint[1] = -live_viewpoint[1] + Config.Camera.position_relative_to_monitor_center_y_mm
            live_viewpoint[2] = -live_viewpoint[2]

        distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint)
        alpha = (1 - numpy.exp(-shared_dynamic_config["smoothing_factor"][0] * distance))
        shared_viewpoint[:] = alpha * live_viewpoint + (1 - alpha) * shared_viewpoint

        left = (shared_viewpoint[0] - Config.Screen.width_mm / 2) * Config.Other.frustumNear / -shared_viewpoint[2]
        right = (shared_viewpoint[0] + Config.Screen.width_mm / 2) * Config.Other.frustumNear / -shared_viewpoint[2]
        bottom = (shared_viewpoint[1] - Config.Screen.height_mm / 2) * Config.Other.frustumNear / -shared_viewpoint[2]
        top = (shared_viewpoint[1] + Config.Screen.height_mm / 2) * Config.Other.frustumNear / -shared_viewpoint[2]

        data_dict = {
            "timestamp": time.time(),
            "viewpoint": {
                "x": float(shared_viewpoint[0]),
                "y": float(shared_viewpoint[1]),
                "z": float(shared_viewpoint[2])
            },
            "frustum": {
                "left": float(left),
                "right": float(right),
                "bottom": float(bottom),
                "top": float(top),
                "near": float(Config.Other.frustumNear),
                "far": float(Config.Other.frustumFar)
            }
        }

        data = json.dumps(data_dict).encode("utf-8")
        sock.sendto(data, (Config.ResultSending.HOST, Config.ResultSending.PORT))

        x += 1
        now = time.time()
        if now - last_time >= 1:
            print(f"[viewpoint] FPS: {x}")
            x = 0
            last_time = now