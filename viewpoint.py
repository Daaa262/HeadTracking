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
        shape=(9,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )
    shared_viewpoint[7] = Config.Other.frustumNear
    shared_viewpoint[8] = Config.Other.frustumFar

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    x = 0
    last_time = time.time()
    while True:
        success, rotation_vector, translation_vector = cv2.solvePnP(Config.FaceModel.model_mm, shared_landmarks, Config.Camera.matrix, Config.Camera.dist_coefficients, flags=Config.FaceModel.PNPMethod)
        if success:
            live_viewpoint = translation_vector.flatten()
            live_viewpoint[0] = live_viewpoint[0] - Config.Camera.position_relative_to_monitor_center_x_mm
            live_viewpoint[1] = -(live_viewpoint[1] - Config.Camera.position_relative_to_monitor_center_y_mm)
            live_viewpoint[2] = live_viewpoint[2] - Config.Camera.position_relative_to_monitor_center_z_mm

            distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint[:3])
            alpha = (1 - numpy.exp(-shared_dynamic_config["smoothing_factor"][0] * distance))
            shared_viewpoint[:3] = alpha * live_viewpoint + (1 - alpha) * shared_viewpoint[:3]

            shared_viewpoint[3] = (shared_viewpoint[0] - Config.Screen.width_mm / 2) * Config.Other.frustumNear / shared_viewpoint[2]
            shared_viewpoint[4] = (shared_viewpoint[0] + Config.Screen.width_mm / 2) * Config.Other.frustumNear / shared_viewpoint[2]
            shared_viewpoint[5] = (shared_viewpoint[1] - Config.Screen.height_mm / 2) * Config.Other.frustumNear / shared_viewpoint[2]
            shared_viewpoint[6] = (shared_viewpoint[1] + Config.Screen.height_mm / 2) * Config.Other.frustumNear / shared_viewpoint[2]

            data_dict = {
                "timestamp": time.time(),
                "viewpoint": {
                    "x": float(shared_viewpoint[0]),
                    "y": float(shared_viewpoint[1]),
                    "z": float(shared_viewpoint[2])
                },
                "frustum": {
                    "left": float(shared_viewpoint[3]),
                    "right": float(shared_viewpoint[4]),
                    "bottom": float(shared_viewpoint[5]),
                    "top": float(shared_viewpoint[6]),
                    "near": float(shared_viewpoint[7]),
                    "far": float(shared_viewpoint[8])
                }
            }

            data = json.dumps(data_dict).encode("utf-8")
            sock.sendto(data, (Config.ResultSending.HOST, Config.ResultSending.PORT))

        if shared_dynamic_config["debug_mode"][0] & 2:
            x += 1
            now = time.time()
            if now - last_time >= 1:
                print(f"[viewpoint] FPS: {x}")
                x = 0
                last_time = now