import time

from config import Config

import numpy
from multiprocessing import Process, Lock
from multiprocessing.shared_memory import SharedMemory

from camera import run as camera_run
from face_mesh import run as face_mesh_run
from viewpoint import run as viewpoint_run
from debug import run as debug_run

import psutil
from burn import run as burn_run

"""
Testing scenarios
0 - default
1 - 2 cores K1
2 - 2 cores K2
3 - 2 cores K3
4 - 1 cores
5 - full CPU usage
"""
scenario = 1

if __name__ == "__main__":
    config = Config()

    dynamic_data = numpy.zeros(1, dtype=numpy.dtype(config.debug.dynamic_fields))
    dynamic_data["smoothing_factor"] = config.other.smoothingFactor
    dynamic_data["running_flag"] = 1
    dynamic_data["test"] = False
    shm_dynamic_data = SharedMemory(create=True, size=dynamic_data.nbytes)
    shared_dynamic_data = numpy.ndarray(1, dtype=dynamic_data.dtype, buffer=shm_dynamic_data.buf)
    shared_dynamic_data[:] = dynamic_data

    shm_pipeline_ids = SharedMemory(create=True, size=numpy.int64().nbytes * 2)
    pipeline_ids = numpy.ndarray(2, dtype=numpy.int64, buffer=shm_pipeline_ids.buf)
    pipeline_ids[:] = 0

    shm_latency_ring = SharedMemory(create=True, size=numpy.int64().nbytes * 32)
    latency_ring = numpy.ndarray(32, dtype=numpy.int64, buffer=shm_latency_ring.buf)

    shm_frame = SharedMemory(create=True, size=config.camera.width * config.camera.height * 3)
    shm_landmarks = SharedMemory(create=True, size=56)
    shm_viewpoint = SharedMemory(create=True, size=140)

    lock_frame = Lock()
    lock_landmarks = Lock()
    lock_viewpoint = Lock()

    camera = Process(target = camera_run, args=(config, shm_dynamic_data.name, shm_pipeline_ids.name, shm_frame.name, lock_frame, shm_latency_ring.name))
    face_mesh = Process(target=face_mesh_run, args=(config, shm_dynamic_data.name, shm_pipeline_ids.name, shm_frame.name, shm_landmarks.name, lock_frame, lock_landmarks))
    viewpoint = Process(target=viewpoint_run, args=(config, shm_dynamic_data.name, shm_pipeline_ids.name, shm_landmarks.name, shm_viewpoint.name, lock_landmarks, lock_viewpoint, shm_latency_ring.name))
    if config.debug.on:
        debug = Process(target=debug_run, args=(config, shm_dynamic_data.name, shm_viewpoint.name, lock_viewpoint))

    try:
        camera.start()
        face_mesh.start()
        viewpoint.start()
        if config.debug.on:
            # noinspection PyUnboundLocalVariable
            debug.start()

        if scenario != 0:
            time.sleep(5)
        if scenario == 1:
            psutil.Process(camera.pid).cpu_affinity([0])
            psutil.Process(face_mesh.pid).cpu_affinity([0])
            psutil.Process(viewpoint.pid).cpu_affinity([1])
            psutil.Process(debug.pid).cpu_affinity([1])
        elif scenario == 2:
            psutil.Process(camera.pid).cpu_affinity([0])
            psutil.Process(face_mesh.pid).cpu_affinity([1])
            psutil.Process(viewpoint.pid).cpu_affinity([0])
            psutil.Process(debug.pid).cpu_affinity([1])
        elif scenario == 3:
            psutil.Process(camera.pid).cpu_affinity([0])
            psutil.Process(face_mesh.pid).cpu_affinity([1])
            psutil.Process(viewpoint.pid).cpu_affinity([1])
            psutil.Process(debug.pid).cpu_affinity([0])
        elif scenario == 4:
            psutil.Process(camera.pid).cpu_affinity([0])
            psutil.Process(face_mesh.pid).cpu_affinity([0])
            psutil.Process(viewpoint.pid).cpu_affinity([0])
            psutil.Process(debug.pid).cpu_affinity([0])
        elif scenario == 5:
            burn = Process(target=burn_run)

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