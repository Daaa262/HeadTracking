import cv2
import numpy
from screeninfo import get_monitors

class Config:
    class Screen:
        primary = 0

        def __init__(self):
            self.width = get_monitors()[self.primary].width
            self.height = get_monitors()[self.primary].height
            self.width_mm = get_monitors()[self.primary].width_mm
            self.height_mm = get_monitors()[self.primary].height_mm

    class Camera:
        position_offset_x_mm = 0.0
        position_offset_y_mm = 148.0
        position_offset_z_mm = 0.0

        auto_detect_camera_parameters = False
        width = 1920
        height = 1080
        fps = 30
        buffer_size = 1

        diagonal_fov_deg = 78.0
        dist_coefficients = numpy.zeros((5, 1), dtype=numpy.float64)

        def __init__(self):
            if self.auto_detect_camera_parameters:
                cap = cv2.VideoCapture(0)
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                self.buffer_size = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))
                cap.release()

            diagonal_px = numpy.hypot(self.width, self.height)
            h_fov = 2 * numpy.arctan((self.width / diagonal_px) * numpy.tan(numpy.radians(self.diagonal_fov_deg) / 2))
            v_fov = 2 * numpy.arctan((self.height / diagonal_px) * numpy.tan(numpy.radians(self.diagonal_fov_deg) / 2))

            fx = self.width / (2 * numpy.tan(h_fov / 2))
            fy = self.height / (2 * numpy.tan(v_fov / 2))
            cx = self.width / 2.0
            cy = self.height / 2.0

            self.matrix = numpy.array(
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]],
                dtype=numpy.float64)

    class Face:
        landmarks = {
            'nose_tip': 1,
            'upper_lip': 13,
            'left_cheek': 234,
            'right_cheek': 454,
            'forehead': 10,
            'left_eye_corner': 33,
            'right_eye_corner': 263
        }

        model_mm = numpy.array([
            [0.0, -15.0, 15.0],
            [0.0, -35.0, 5.0],
            [-65.0, 15.0, -30.0],
            [65.0, 15.0, -30.0],
            [0.0, 65.0, -5.0],
            [-35.0, 0.0, 0.0],
            [35.0, 0.0, 0.0],
        ], dtype=numpy.float32)

        PNPMethod = cv2.SOLVEPNP_ITERATIVE

    class ResultSending:
        on = True
        HOST = "127.0.0.1"
        PORT = 9999

    class Debug:
        on = True
        dynamic_fields = [
            ("smoothing_factor", numpy.float64),
            ("running_flag", numpy.int32),
            ("camera_fps", numpy.int32),
            ("face_mesh_fps", numpy.int32),
            ("viewpoint_fps", numpy.int32),
            ("test", numpy.bool_),
            ("latency_ready", numpy.float64),
            ("latency", numpy.float64)
        ]

    class Other:
        smoothingFactor = 0.05
        smoothingFrequency = 0.005
        nearPlane = 1.0
        farPlane = 5000.0

    def __init__(self):
        self.screen = self.Screen()
        self.camera = self.Camera()
        self.face = self.Face()
        self.resultSending = self.ResultSending()
        self.debug = self.Debug()
        self.other = self.Other()