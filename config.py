import cv2
import mediapipe
import numpy
from screeninfo import get_monitors

class Config:
    """Global configuration"""

    class Screen:
        width = get_monitors()[0].width
        height = get_monitors()[0].height
        width_mm = get_monitors()[0].width_mm
        height_mm = get_monitors()[0].height_mm

    class Camera:
        position_relative_to_monitor_center_x_mm = 0
        position_relative_to_monitor_center_y_mm = 0#get_monitors()[0].height_mm / 2
        position_relative_to_monitor_center_z_mm = 0

        width = 1920
        height = 1080
        fps = 30
        buffer_size = 1

        diagonal_fov_deg = 78.0

        diag_px = numpy.hypot(width, height)
        hfov = 2 * numpy.arctan((width / diag_px) * numpy.tan(numpy.radians(diagonal_fov_deg) / 2))
        vfov = 2 * numpy.arctan((height / diag_px) * numpy.tan(numpy.radians(diagonal_fov_deg) / 2))

        fx = width / (2 * numpy.tan(hfov / 2))
        fy = height / (2 * numpy.tan(vfov / 2))
        cx = width / 2.0
        cy = height / 2.0

        matrix = numpy.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=numpy.float64)

        dist_coefficients = numpy.zeros((5, 1), dtype=numpy.float64)

    class FaceMesh:
        mp_face_mesh = mediapipe.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        landmarks = {
            'nose_tip': 1,
            'upper_lip': 13,
            'left_cheek': 234,
            'right_cheek': 454,
            'forehead': 10,
            'left_eye_corner': 33,
            'right_eye_corner': 263
        }

    class FaceModel:
        model_mm = numpy.array([
            [0.0, -30.0, 5.0],
            [0.0, -50.0, -5.0],
            [-65.0, 0.0, -40.0],
            [65.0, 0.0, -40.0],
            [0.0, 50.0, -15.0],
            [-35.0, -15.0, -10.0],
            [35.0, -15.0, -10.0],
        ], dtype=numpy.float64)

        PNPMethod = cv2.SOLVEPNP_ITERATIVE

    class ResultSending:
        on = 1
        HOST = "127.0.0.1"
        PORT = 9999

    class Debug:
        on = True
        dynamic_fields = [
            ("smoothing_factor", numpy.float64),
            ("running_flag", numpy.int32),
            ("camera_fps", numpy.int32),
            ("face_mesh_fps", numpy.int32),
            ("viewpoint_fps", numpy.int32)
        ]

    class Other:
        smoothingFactor = 0.001
        smoothingFrequency = 0.005
        frustumNear = 1.0
        frustumFar = 10000.0