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

    m = numpy.zeros(16)
    m[0] = 2.0 * config.other.nearPlane / (right - left)
    m[5] = 2.0 * config.other.nearPlane / (top - bottom)
    m[8] = (right + left) / (right - left)
    m[9] = (top + bottom) / (top - bottom)
    m[10] = (config.other.farPlane + config.other.nearPlane) / (config.other.nearPlane - config.other.farPlane)
    m[11] = -1.0
    m[14] = 2.0 * config.other.farPlane * config.other.nearPlane / (config.other.nearPlane - config.other.farPlane)

    return m

def run(config, shm_dynamic_data_name, shm_pipeline_ids_name, shm_landmarks_name, shm_viewpoint_name, lock_landmarks, lock_viewpoint, shm_latency_ring_name):
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(config.debug.dynamic_fields),
        buffer=shm_dynamic_data.buf)

    shm_pipeline_ids = SharedMemory(name=shm_pipeline_ids_name)
    shared_pipeline_ids = numpy.ndarray(
        shape=(2,),
        dtype=numpy.int64,
        buffer=shm_pipeline_ids.buf)

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

    shm_latency_ring = SharedMemory(name=shm_latency_ring_name)
    latency_ring = numpy.ndarray(
        shape=(32,),
        dtype=numpy.int64,
        buffer=shm_latency_ring.buf)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    live_viewpoint = None

    test_started = False
    total_frames = 0
    test_started_time = None

    latency_test_started = False
    min_latency = None
    total_latency = None
    total_processed_frames = 0
    max_latency = None
    latency_test_started_time = None

    try:
        frames = 0
        last_time = time.perf_counter()
        last_processed = 0
        last_smooth = time.perf_counter()
        while shared_dynamic_data['running_flag'][0]:
            if shared_dynamic_data['test'][0] and not test_started and not latency_test_started:
                total_frames = 0
                test_started_time = time.perf_counter()
                test_started = True

                min_latency = 10.0 ** 10
                total_latency = 0
                max_latency = 0.0
                latency_test_started_time = time.perf_counter()
                latency_test_started = True

            frames += 1
            now = time.perf_counter()
            if now - last_time >= 1:
                shared_dynamic_data["viewpoint_fps"][0] = frames
                frames = 0
                last_time = now

            if shared_pipeline_ids[1] > last_processed:
                if latency_test_started:
                    if time.perf_counter() - latency_test_started_time > 60:
                        print("[Latency]: Min", min_latency / 10 ** 6, "ms")
                        print("[Latency]: Avg", total_latency / total_processed_frames / 10 ** 6, "ms")
                        print("[Latency]: Max", max_latency / 10 ** 6, "ms")
                        latency_test_started = False

                    total_processed_frames += 1
                    latency = (time.perf_counter_ns() - latency_ring[(shared_pipeline_ids[1] - 1) % 32])
                    total_latency += latency

                    if latency < min_latency:
                        min_latency = latency
                    if latency > max_latency:
                        max_latency = latency

                last_processed = shared_pipeline_ids[1]
                with lock_landmarks:
                    success, _, translation_vector = cv2.solvePnP(config.face.model_mm, shared_landmarks[:], config.camera.matrix, config.camera.dist_coefficients, flags=config.face.PNPMethod)
                if success:
                    live_viewpoint = translation_vector.flatten()
                    live_viewpoint[0] = -(live_viewpoint[0] - config.camera.position_offset_x_mm)
                    live_viewpoint[1] = -(live_viewpoint[1] - config.camera.position_offset_y_mm)
                    live_viewpoint[2] = live_viewpoint[2] - config.camera.position_offset_z_mm

            if live_viewpoint is not None:
                with lock_viewpoint:
                    now_smooth = time.perf_counter()
                    dt = now_smooth - last_smooth
                    last_smooth = now_smooth

                    k = shared_dynamic_data["smoothing_factor"][0]
                    distance = numpy.linalg.norm(live_viewpoint - shared_viewpoint[:3])
                    alpha = 1.0 - numpy.exp(-k * distance * dt)
                    shared_viewpoint[:3] = shared_viewpoint[:3] + alpha * (live_viewpoint - shared_viewpoint[:3])

                    shared_viewpoint[3:19] = make_projection(config, shared_viewpoint[:3])
                    shared_viewpoint[19:35] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -shared_viewpoint[0], -shared_viewpoint[1], -shared_viewpoint[2], 1]

                if config.resultSending.on:
                    data_dict = {
                        "timestamp": time.time(),
                        "viewpoint": {
                            "x": float(shared_viewpoint[0]),
                            "y": float(shared_viewpoint[1]),
                            "z": float(shared_viewpoint[2])
                        },
                        "projection_matrix": list(map(float, shared_viewpoint[3:19])),
                        "view_matrix": list(map(float, shared_viewpoint[19:35]))
                    }

                    data = json.dumps(data_dict).encode("utf-8")
                    sock.sendto(data, (config.resultSending.HOST, config.resultSending.PORT))

            if test_started:
                if time.perf_counter() - test_started_time > 60:
                    print("[Viewpoint]: ", total_frames / 60, "fps")
                    test_started = False

                total_frames += 1
    finally:
        shm_dynamic_data.close()
        shm_landmarks.close()
        shm_viewpoint.close()