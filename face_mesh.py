from config import Config

import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(shm_dynamic_data_name, shm_frame_name, shm_landmarks_name):
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

    shm_landmarks = SharedMemory(name=shm_landmarks_name)
    shared_landmarks = numpy.ndarray(
        shape=(7, 2),
        dtype=numpy.float32,
        buffer=shm_landmarks.buf
    )

    x = 0
    last_time = time.time()
    while shared_dynamic_data['running_flag'][0]:
        shared_frame.flags.writeable = False
        mesh = Config.FaceMesh.face_mesh.process(shared_frame)
        shared_frame.flags.writeable = True

        if mesh.multi_face_landmarks:
            landmarks = mesh.multi_face_landmarks[0].landmark
            for i, index in enumerate(Config.FaceMesh.landmarks.values()):
                shared_landmarks[i, 0] = landmarks[index].x * Config.Camera.width
                shared_landmarks[i, 1] = landmarks[index].y * Config.Camera.height

        x += 1
        now = time.time()
        if now - last_time >= 1:
            shared_dynamic_data["face_mesh_fps"][0] = x
            x = 0
            last_time = now