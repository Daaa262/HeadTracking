from config import Config

import json
import cv2
import numpy
import time
import socket
from multiprocessing.shared_memory import SharedMemory

def run(shm_dynamic_data_name, shm_landmarks_name, shm_viewpoint_name):
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_data.buf)

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

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        frames = 0
        last_time = time.time()
        last_time_smoothed = time.time()
        while shared_dynamic_data['running_flag'][0]:
            now = time.time()
            if now - last_time >= 1:
                shared_dynamic_data["viewpoint_fps"][0] = frames
                frames = 0
                last_time = now

            if now - last_time_smoothed >= Config.Other.smoothingFrequency:
                frames += 1
                last_time_smoothed = now

                success, rotation_vector, translation_vector = cv2.solvePnP(Config.FaceModel.model_mm, shared_landmarks, Config.Camera.matrix, Config.Camera.dist_coefficients, flags=Config.FaceModel.PNPMethod)
                if success:
                    live_viewpoint = translation_vector.flatten()
                    live_viewpoint[0] = -(live_viewpoint[0] - Config.Camera.position_relative_to_monitor_center_x_mm)
                    live_viewpoint[1] = -(live_viewpoint[1] - Config.Camera.position_relative_to_monitor_center_y_mm)
                    live_viewpoint[2] = live_viewpoint[2] - Config.Camera.position_relative_to_monitor_center_z_mm

                    distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint[:3])
                    alpha = (1 - numpy.exp(-shared_dynamic_data["smoothing_factor"][0] * distance))
                    shared_viewpoint[:3] = alpha * live_viewpoint + (1 - alpha) * shared_viewpoint[:3]

                    forward = -shared_viewpoint[:3]
                    forward /= numpy.linalg.norm(forward)
                    right = numpy.cross(forward, [0, 1, 0])
                    right /= numpy.linalg.norm(right)
                    up = numpy.cross(right, forward)
                    up /= numpy.linalg.norm(up)

                    corners = [
                        [Config.Screen.width_mm / 2.0,  Config.Screen.height_mm / 2.0,  0],
                        [-Config.Screen.width_mm / 2.0, Config.Screen.height_mm / 2.0,  0],
                        [Config.Screen.width_mm / 2.0,  -Config.Screen.height_mm / 2.0, 0],
                        [-Config.Screen.width_mm / 2.0, -Config.Screen.height_mm / 2.0, 0]
                    ]

                    x, y = [], []
                    for corner in corners:
                        corner_vector = corner - shared_viewpoint[:3]
                        x.append(numpy.dot(corner_vector, right) * Config.Other.frustumNear / numpy.dot(corner_vector, forward))
                        y.append(numpy.dot(corner_vector, up) * Config.Other.frustumNear / numpy.dot(corner_vector, forward))

                    shared_viewpoint[3] = min(x)
                    shared_viewpoint[4] = max(x)
                    shared_viewpoint[5] = min(y)
                    shared_viewpoint[6] = max(y)
                    shared_viewpoint[7] = Config.Other.frustumNear
                    shared_viewpoint[8] = Config.Other.frustumFar

                    if Config.ResultSending.on:
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

    finally:
        shm_dynamic_data.close()
        shm_landmarks.close()
        shm_viewpoint.close()