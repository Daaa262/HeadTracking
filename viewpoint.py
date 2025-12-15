import json
import cv2
import numpy
import time
import socket
from multiprocessing.shared_memory import SharedMemory

def make_projection(config, eye):
    left = (-config.screen.width_mm / 2 - eye[0]) * config.other.nearPlane / eye[2]
    right = (config.screen.width_mm / 2 - eye[0]) * config.other.nearPlane / eye[2]
    bottom = (-config.screen.height_mm / 2 - eye[1]) * config.other.nearPlane / eye[2]
    top = (config.screen.height_mm / 2 - eye[1]) * config.other.nearPlane / eye[2]

    m = numpy.zeros((4, 4))
    m[0, 0] = 2.0 * config.other.nearPlane / (right - left)
    m[0, 2] = (right + left) / (right - left)
    m[1, 1] = 2.0 * config.other.nearPlane / (top - bottom)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(config.other.farPlane + config.other.nearPlane) / (
            config.other.farPlane - config.other.nearPlane)
    m[2, 3] = -2.0 * config.other.farPlane * config.other.nearPlane / (
            config.other.farPlane - config.other.nearPlane)
    m[3, 2] = -1.0
    return m

def run(config, shm_dynamic_data_name, shm_landmarks_name, shm_viewpoint_name, lock_landmarks, lock_viewpoint):
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(config.debug.dynamic_fields),
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

            if now - last_time_smoothed >= config.other.smoothingFrequency:
                frames += 1
                last_time_smoothed = now

                with lock_landmarks:
                    success, rotation_vector, translation_vector = cv2.solvePnP(config.face.model_mm, shared_landmarks[:], config.camera.matrix, config.camera.dist_coefficients, flags=config.face.PNPMethod)
                if success:
                    live_viewpoint = translation_vector.flatten()
                    live_viewpoint[0] = -(live_viewpoint[0] - config.camera.position_offset_x_mm)
                    live_viewpoint[1] = -(live_viewpoint[1] - config.camera.position_offset_y_mm)
                    live_viewpoint[2] = live_viewpoint[2] - config.camera.position_offset_z_mm

                    with lock_viewpoint:
                        distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint[:3])
                        alpha = (1 - numpy.exp(-shared_dynamic_data["smoothing_factor"][0] * distance))
                        shared_viewpoint[:3] = alpha * live_viewpoint + (1 - alpha) * shared_viewpoint[:3]

                        projection = make_projection(config, shared_viewpoint[:3])
                        shared_viewpoint[3:19] = projection.T.flatten()
                        shared_viewpoint[19:35] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -shared_viewpoint[0], -shared_viewpoint[1], -shared_viewpoint[2], 1]

                    if config.resultSending.on:
                        data_dict = {
                            "timestamp": time.time(),
                            "viewpoint": {
                                "x": float(shared_viewpoint[0]),
                                "y": float(shared_viewpoint[1]),
                                "z": float(shared_viewpoint[2])
                            },
                            "view_matrix": list(map(float, shared_viewpoint[3:19])),
                            "projection_matrix": list(map(float, shared_viewpoint[19:35]))
                        }

                        data = json.dumps(data_dict).encode("utf-8")
                        sock.sendto(data, (config.resultSending.HOST, config.resultSending.PORT))

    finally:
        shm_dynamic_data.close()
        shm_landmarks.close()
        shm_viewpoint.close()