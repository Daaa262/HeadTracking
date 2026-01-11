from config import Config

import numpy
from multiprocessing import Process, Lock
from multiprocessing.shared_memory import SharedMemory

from camera import run as camera_run
from face_mesh import run as face_mesh_run
from viewpoint import run as viewpoint_run
from debug import run as debug_run

if __name__ == "__main__":
    config = Config()

    dynamic_data = numpy.zeros(1, dtype=numpy.dtype(config.debug.dynamic_fields))
    dynamic_data["smoothing_factor"] = config.other.smoothingFactor
    dynamic_data["running_flag"] = 1
    shm_dynamic_data = SharedMemory(create=True, size=dynamic_data.nbytes)
    shared_dynamic_data = numpy.ndarray(1, dtype=dynamic_data.dtype, buffer=shm_dynamic_data.buf)
    shared_dynamic_data[:] = dynamic_data

    shm_frame = SharedMemory(create=True, size=config.camera.width * config.camera.height * 3)
    shm_landmarks = SharedMemory(create=True, size=56)
    shm_viewpoint = SharedMemory(create=True, size=140)

    lock_frame = Lock()
    lock_landmarks = Lock()
    lock_viewpoint = Lock()

    camera = Process(target = camera_run, args=(config, shm_dynamic_data.name, shm_frame.name, lock_frame))
    face_mesh = Process(target=face_mesh_run, args=(config, shm_dynamic_data.name, shm_frame.name, shm_landmarks.name, lock_frame, lock_landmarks))
    viewpoint = Process(target=viewpoint_run, args=(config, shm_dynamic_data.name, shm_landmarks.name, shm_viewpoint.name, lock_landmarks, lock_viewpoint))
    if config.debug.on:
        debug = Process(target=debug_run, args=(config, shm_dynamic_data.name, shm_viewpoint.name, lock_viewpoint))

    try:
        camera.start()
        face_mesh.start()
        viewpoint.start()
        if config.debug.on:
            # noinspection PyUnboundLocalVariable
            debug.start()

        camera.join()
        face_mesh.join()
        viewpoint.join()
        if config.debug.on:
            debug.join()

    finally:
        shm_dynamic_data.close()
        shm_dynamic_data.unlink()
        shm_frame.close()
        shm_frame.unlink()
        shm_landmarks.close()
        shm_landmarks.unlink()
        shm_viewpoint.close()
        shm_viewpoint.unlink()