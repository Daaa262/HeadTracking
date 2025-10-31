from config import Config

import numpy
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

from camera import run as camera_run
from face_mesh import run as face_mesh_run
from viewpoint import run as viewpoint_run
from debug import run as debug_run

if __name__ == "__main__":
    dynamic_config = numpy.zeros(1, dtype=numpy.dtype(Config.Debug.dynamic_fields))
    dynamic_config["smoothing_factor"] = Config.Other.smoothingFactor
    dynamic_config["debug_mode"] = Config.Debug.mode
    shm_dynamic_config = SharedMemory(create=True, size=dynamic_config.nbytes)
    shared_dynamic_config = numpy.ndarray(1, dtype=dynamic_config.dtype, buffer=shm_dynamic_config.buf)
    shared_dynamic_config[:] = dynamic_config

    shm_frame = SharedMemory(create=True, size=Config.Camera.width * Config.Camera.height * 3)
    shm_landmarks = SharedMemory(create=True, size=56)
    shm_viewpoint = SharedMemory(create=True, size=12)

    camera = Process(target = camera_run, args=(shm_frame.name,))
    face_mesh = Process(target=face_mesh_run, args=(shm_frame.name, shm_landmarks.name))
    viewpoint = Process(target=viewpoint_run, args=(shm_dynamic_config.name, shm_landmarks.name, shm_viewpoint.name))
    debug = Process(target=debug_run, args=(shm_dynamic_config.name, shm_viewpoint.name,))

    camera.start()
    face_mesh.start()
    viewpoint.start()
    debug.start()


