from config import Config

import json
import cv2
import numpy
import time
import socket
from multiprocessing.shared_memory import SharedMemory

def make_projection(eye):
    left = (-Config.Screen.width_mm / 2 - eye[0]) * Config.Other.frustumNear / eye[2]
    right = (Config.Screen.width_mm / 2 - eye[0]) * Config.Other.frustumNear / eye[2]
    bottom = (-Config.Screen.height_mm / 2 - eye[1]) * Config.Other.frustumNear / eye[2]
    top = (Config.Screen.height_mm / 2 - eye[1]) * Config.Other.frustumNear / eye[2]

    m = numpy.zeros((4, 4))
    m[0, 0] = 2.0 * Config.Other.frustumNear / (right - left)
    m[0, 2] = (right + left) / (right - left)
    m[1, 1] = 2.0 * Config.Other.frustumNear / (top - bottom)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(Config.Other.frustumFar + Config.Other.frustumNear) / (
                Config.Other.frustumFar - Config.Other.frustumNear)
    m[2, 3] = -2.0 * Config.Other.frustumFar * Config.Other.frustumNear / (
                Config.Other.frustumFar - Config.Other.frustumNear)
    m[3, 2] = -1.0
    return m

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
        shape=(35,),
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

                success, rotation_vector, translation_vector = cv2.solvePnP(Config.FaceModel.model_mm, shared_landmarks[:], Config.Camera.matrix, Config.Camera.dist_coefficients, flags=Config.FaceModel.PNPMethod)
                if success:
                    live_viewpoint = translation_vector.flatten()
                    live_viewpoint[0] = -(live_viewpoint[0] - Config.Camera.position_offset_x_mm)
                    live_viewpoint[1] = -(live_viewpoint[1] - Config.Camera.position_offset_y_mm)
                    live_viewpoint[2] = live_viewpoint[2] - Config.Camera.position_offset_z_mm

                    distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint[:3])
                    alpha = (1 - numpy.exp(-shared_dynamic_data["smoothing_factor"][0] * distance))
                    shared_viewpoint[:3] = alpha * live_viewpoint + (1 - alpha) * shared_viewpoint[:3]

                    projection = make_projection(shared_viewpoint[:3])
                    shared_viewpoint[3:19] = projection.T.flatten()
                    shared_viewpoint[19:35] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -shared_viewpoint[0], -shared_viewpoint[1], -shared_viewpoint[2], 1]

                    if Config.ResultSending.on:
                        data_dict = {
                            "timestamp": time.time(),
                            "viewpoint": {
                                "x": float(shared_viewpoint[0]),
                                "y": float(shared_viewpoint[1]),
                                "z": float(shared_viewpoint[2])
                            },
                            "view matrix": {
                                shared_viewpoint[3:19]
                            },
                            "projection matrix": {
                                shared_viewpoint[19:35]
                            }
                        }

                        data = json.dumps(data_dict).encode("utf-8")
                        sock.sendto(data, (Config.ResultSending.HOST, Config.ResultSending.PORT))

    finally:
        shm_dynamic_data.close()
        shm_landmarks.close()
        shm_viewpoint.close()