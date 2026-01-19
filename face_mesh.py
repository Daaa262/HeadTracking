import mediapipe
import numpy
import time
from multiprocessing.shared_memory import SharedMemory

def run(config, shm_dynamic_data_name, shm_pipeline_ids_name, shm_frame_name, shm_landmarks_name, lock_frame, lock_landmarks):
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

    shm_landmarks = SharedMemory(name=shm_landmarks_name)
    shared_landmarks = numpy.ndarray(
        shape=(7, 2),
        dtype=numpy.float32,
        buffer=shm_landmarks.buf
    )

    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

    try:
        frames = 0
        last_time = time.perf_counter()
        last_processed = 0
        while shared_dynamic_data['running_flag'][0]:
            if shared_pipeline_ids[0] == last_processed:
                continue
            else:
                last_processed = shared_pipeline_ids[0]

            frames += 1
            now = time.perf_counter()
            if now - last_time >= 1:
                shared_dynamic_data["face_mesh_fps"][0] = frames
                frames = 0
                last_time = now

            with lock_frame:
                shared_frame.flags.writeable = False
                mesh = face_mesh.process(shared_frame)
                shared_frame.flags.writeable = True

            if mesh.multi_face_landmarks:
                landmarks = mesh.multi_face_landmarks[0].landmark
                with lock_landmarks:
                    for i, index in enumerate(config.face.landmarks.values()):
                        shared_landmarks[i, 0] = landmarks[index].x * config.camera.width
                        shared_landmarks[i, 1] = landmarks[index].y * config.camera.height
                shared_pipeline_ids[1] += 1

    finally:
        shm_dynamic_data.close()
        shm_frame.close()
        shm_landmarks.close()