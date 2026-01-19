import os
import cv2
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def run(config, shm_dynamic_data_name, shm_pipeline_ids_name, shm_frame_name, lock_frame):
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

    shm_frame = SharedMemory(name=shm_frame_name)
    shared_frame = numpy.ndarray(
        shape=(config.camera.height, config.camera.width, 3),
        dtype=numpy.uint8,
        buffer=shm_frame.buf
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.camera.buffer_size)

    frames = 0
    last_time = time.perf_counter()
    try:
        while shared_dynamic_data['running_flag'][0]:
            frames += 1
            now = time.perf_counter()
            if now - last_time >= 1:
                shared_dynamic_data["camera_fps"][0] = frames
                frames = 0
                last_time = now

            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            with lock_frame:
                shared_frame[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shared_pipeline_ids[0] += 1

    finally:
        cap.release()
        shm_dynamic_data.close()
        shm_frame.close()