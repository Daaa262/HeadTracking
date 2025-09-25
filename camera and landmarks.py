import random
import winreg
import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Tuple, Optional
from dataclasses import dataclass
from screeninfo import get_monitors
from point import Point

@dataclass
class TrackerConfig:
    smoothing_factor: float = 0.4
    max_jump_threshold: float = 150.0
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    distance_bounds: Tuple[float, float] = (80.0, 1500.0)
    max_bad_frames: int = 3
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 60
    debug_mode: bool = True

class HeadTracker:
    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

        self.previous_translation: Optional[np.ndarray] = None
        self.consecutive_bad_frames: int = 0
        self.current_translation: Optional[np.ndarray] = None

        self.essential_landmarks = {
            'nose_tip': 1,
            'upper_lip': 13,
            'left_cheek': 234,
            'right_cheek': 454,
            'forehead': 10,
            'left_eye_corner': 33,
            'right_eye_corner': 263
        }

        self.face_3d_model = np.array([
            [0.0, -30.0, 5.0],
            [0.0, -50.0, -5.0],
            [-65.0, 0.0, -40.0],
            [65.0, 0.0, -40.0],
            [0.0, 50.0, -15.0],
            [-35.0, -15.0, -10.0],
            [35.0, -15.0, -10.0],
        ], dtype=np.float64)

        self.face_landmark_indices = [1, 13, 234, 454, 10, 33, 263]

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coefficients = np.zeros((4, 1))

    def setup_camera(self) -> Optional[cv2.VideoCapture]:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                return self.configure_camera(cap)
            cap.release()
        except Exception as e:
            if self.config.debug_mode:
                print(f"Camera failed: {e}")

        print("Error: No camera available!")
        return None

    def configure_camera(self, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = cap.read()
        if not ret:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.frame_width)
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.frame_height)
        else:
            actual_height, actual_width = frame.shape[:2]

        actual_width = int(actual_width)
        actual_height = int(actual_height)

        Point.screen_pixels_x = get_monitors()[0].width
        Point.screen_pixels_y = get_monitors()[0].height
        Point.screen_width_mm = get_monitors()[0].width_mm
        Point.screen_height_mm = get_monitors()[0].height_mm

        dFoV_deg = 78.0
        diag_px = np.hypot(actual_width, actual_height)
        hfov = 2 * np.arctan((actual_width / diag_px) * np.tan(np.radians(dFoV_deg) / 2))
        vfov = 2 * np.arctan((actual_height / diag_px) * np.tan(np.radians(dFoV_deg) / 2))
        fx = actual_width / (2 * np.tan(hfov / 2))
        fy = actual_height / (2 * np.tan(vfov / 2))
        cx = actual_width / 2.0
        cy = actual_height / 2.0
        self.camera_matrix = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0, 0, 1]], dtype=np.float64)
        self.dist_coefficients = np.zeros((5, 1), dtype=np.float64)

        if self.config.debug_mode:
            print(f"Camera initialized (actual): {actual_width}x{actual_height}")
            print("camera_matrix:\n", self.camera_matrix)
        return cap

    def get_face_landmarks_2d(self, landmarks, frame_width: int, frame_height: int) -> Optional[np.ndarray]:
        try:
            face_2d = []
            for idx in self.face_landmark_indices:
                if idx >= len(landmarks.landmark):
                    if self.config.debug_mode:
                        print(f"Invalid landmark index: {idx}")
                    return None

                landmark = landmarks.landmark[idx]
                x = landmark.x * frame_width
                y = landmark.y * frame_height
                face_2d.append([x, y])

            if len(face_2d) < 6:
                if self.config.debug_mode:
                    print(f"Insufficient points for PnP: {len(face_2d)} < 6")
                return None

            return np.array(face_2d, dtype=np.float64)

        except Exception as e:
            if self.config.debug_mode:
                print(f"Error extracting 2D points: {e}")
            return None

    def is_valid_pose(self, translation_vector: np.ndarray, distance: float) -> bool:
        if not (self.config.distance_bounds[0] <= distance <= self.config.distance_bounds[1]):
            return False

        if self.previous_translation is not None:
            jump = np.linalg.norm(translation_vector - self.previous_translation)
            if jump > self.config.max_jump_threshold:
                self.consecutive_bad_frames += 1
                return self.consecutive_bad_frames < self.config.max_bad_frames

        self.consecutive_bad_frames = 0
        return True

    def smooth_translation(self, new_translation):
        if self.previous_translation is None:
            self.previous_translation = new_translation.copy()
            return new_translation
        alpha = self.config.smoothing_factor
        smoothed = alpha * new_translation + (1 - alpha) * self.previous_translation
        self.previous_translation = smoothed.copy()
        return smoothed

    def calculate_head_pose(self, landmarks, frame_width: int, frame_height: int) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], float]:
        if self.camera_matrix is None:
            return None, None, 0.0

        face_2d = self.get_face_landmarks_2d(landmarks, frame_width, frame_height)
        if face_2d is None:
            return None, None, 0.0

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.face_3d_model,
                face_2d,
                self.camera_matrix,
                self.dist_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                distance = float(np.linalg.norm(translation_vector))

                if not self.is_valid_pose(translation_vector, distance):
                    if (self.consecutive_bad_frames >= self.config.max_bad_frames and
                            self.previous_translation is not None):
                        return None, self.previous_translation, float(np.linalg.norm(self.previous_translation))

                smoothed_translation = self.smooth_translation(translation_vector)
                smoothed_distance = float(np.linalg.norm(smoothed_translation))

                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                return rotation_matrix, smoothed_translation, smoothed_distance

        except Exception as e:
            if self.config.debug_mode:
                print(f"PnP calculation error: {e}")

        return None, None, 0.0

    def draw_debug_info(self, frame: np.ndarray, landmarks, offset_x: float, offset_y: float, offset_z: float,
                        distance: float) -> None:
        if not self.config.debug_mode:
            return

        frame_height, frame_width = frame.shape[:2]

        for name, idx in self.essential_landmarks.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                color = (0, 255, 0)
                cv2.circle(frame, (x, y), 3, color, -1)

        if True:
            info_lines = [
                f"Distance: {distance:.1f}mm",
                f"Offset X: {offset_x:.1f}mm",
                f"Offset Y: {offset_y:.1f}mm",
                f"Offset Z: {offset_z:.1f}mm",
                f"Reference: Upper Lip (yawn-stable)"
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 60 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_axes(self, frame, rvec, tvec):
        if self.camera_matrix is None: return
        axis = np.float64([[0, 0, 0], [30, 0, 0], [0, 30, 0], [0, 0, 30]])
        pts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coefficients)
        pts = pts.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 2)
        cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 2)
        cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 2)

    def get_monitor_size(self):
        path = r"SYSTEM\CurrentControlSet\Enum\DISPLAY"

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as display_key:
                for i in range(winreg.QueryInfoKey(display_key)[0]):
                    subkey_name = winreg.EnumKey(display_key, i)
                    subkey_path = f"{path}\\{subkey_name}"

                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey_path) as subkey:
                        for j in range(winreg.QueryInfoKey(subkey)[0]):
                            monitor_id = winreg.EnumKey(subkey, j)
                            full_path = f"{subkey_path}\\{monitor_id}\\Device Parameters"

                            try:
                                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, full_path) as monitor_key:
                                    edid_raw, _ = winreg.QueryValueEx(monitor_key, "EDID")
                                    width_mm = edid_raw[21] * 10
                                    height_mm = edid_raw[22] * 10

                                    return {
                                        "width": width_mm,
                                        "height": height_mm
                                    }
                            except FileNotFoundError:
                                continue
        except Exception as e:
            print(f"Błąd: {e}")

        return None

    def run(self) -> None:
        cap = self.setup_camera()
        if cap is None:
            return

        cv2.namedWindow("kosmos", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("kosmos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        monitor_size = self.get_monitor_size()
        print(monitor_size.get("width"))
        print(monitor_size.get("height"))

        image = np.zeros((500, 500, 3), dtype=np.uint8)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        points = []
        for i in range(1000):
            color_choice = random.randint(0, 5)

            if color_choice == 0:
                color = (0, 0, random.randint(200, 255))
            elif color_choice == 1:
                color = (0, random.randint(150, 230), random.randint(230, 255))
            elif color_choice == 2:
                color = (random.randint(220, 255), 0, 0)
            elif color_choice == 3:
                color = (random.randint(220, 255), random.randint(180, 200), 0)
            else:
                color = (255, 255, 255)

            points.append(Point(np.array([
                random.randint(-6 * (1000 - i) - 100, 6 * (1000 - i) + 100),
                random.randint(-6 * (1000 - i) - 100, 6 * (1000 - i) + 100),
                9600 - i ** 0.9 * 20
            ]), color, random.randint(0, int(i ** 1.1) // 50 + 1)))

        print("Head Tracker started!")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug mode")

        position = np.array([0, 0, 0])

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read frame from camera")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for facial_landmarks in results.multi_face_landmarks:

                        rotation_matrix, translation_vector, distance = self.calculate_head_pose(
                            facial_landmarks, frame_width, frame_height
                        )

                        if translation_vector is not None:
                            self.current_translation = translation_vector

                            offset_x, offset_y, offset_z = translation_vector[0].item(), translation_vector[1].item(), translation_vector[2].item()

                            image = np.zeros((Point.screen_pixels_y, Point.screen_pixels_x, 3), dtype=np.uint8)

                            for point in points:
                                projected_point = point.project(np.array([-offset_x, -offset_y, -offset_z]), position)
                                if projected_point is not None:
                                    cv2.circle(image, (projected_point[0], projected_point[1]), projected_point[2], point.color, -1)

                            self.draw_debug_info(frame, facial_landmarks, offset_x, offset_y, offset_z, distance)

                            if rotation_matrix is not None and self.config.debug_mode:
                                self.draw_axes(frame, rotation_matrix, translation_vector)
                        else:
                            if self.config.debug_mode:
                                cv2.putText(frame, "Tracking unstable", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    if self.config.debug_mode:
                        cv2.putText(frame, "No face detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.config.debug_mode:
                    cv2.imshow("debug", frame)
                cv2.imshow("kosmos", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    position[0] += 5
                elif key == ord('a'):
                    position[0] -= 5
                elif key == ord('w'):
                    position[2] += 5
                elif key == ord('s'):
                    position[2] -= 5
                elif key == ord('1'):
                    self.config.smoothing_factor *= 1.1
                    print(self.config.smoothing_factor)
                elif key == ord('2'):
                    self.config.smoothing_factor /= 1.1
                    print(self.config.smoothing_factor)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    config = TrackerConfig()
    tracker = HeadTracker(config)
    tracker.run()

if __name__ == "__main__":
    main()