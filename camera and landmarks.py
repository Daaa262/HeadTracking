import cv2
import mediapipe as mp
from typing import Tuple, Optional
import numpy as np
import os
import logging
from dataclasses import dataclass

# Suppress MediaPipe/TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


@dataclass
class TrackerConfig:
    """Configuration for HeadTracker"""
    smoothing_factor: float = 0.2
    max_jump_threshold: float = 150.0
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    distance_bounds: Tuple[float, float] = (80.0, 1500.0)
    max_bad_frames: int = 3
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    debug_mode: bool = True


class HeadTracker:
    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()

        # MediaPipe setup - optimized
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

        # Tracking state
        self.previous_translation: Optional[np.ndarray] = None
        self.consecutive_bad_frames: int = 0
        self.calibrated: bool = False
        self.neutral_position: Optional[np.ndarray] = None
        self.current_translation: Optional[np.ndarray] = None

        # Essential landmarks - minimum 6 points required for PnP
        self.essential_landmarks = {
            'nose_tip': 1,
            'chin': 152,
            'left_cheek': 234,
            'right_cheek': 454,
            'forehead': 10,
            'left_eye_corner': 33,
            'right_eye_corner': 263
        }

        # 3D face model - 7 points for robust PnP estimation
        self.face_3d_model = np.array([
            [0.0, -30.0, 5.0],  # Nose tip (reference point)
            [0.0, -95.6, -12.5],  # Chin
            [-65.0, 0.0, -40.0],  # Left cheek
            [65.0, 0.0, -40.0],  # Right cheek
            [0.0, 50.0, -15.0],  # Forehead
            [-35.0, -15.0, -10.0],  # Left eye corner
            [35.0, -15.0, -10.0],  # Right eye corner
        ], dtype=np.float64)

        # MediaPipe indices corresponding to 3D model
        self.face_landmark_indices = [1, 152, 234, 454, 10, 33, 263]

        # Camera parameters
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coefficients = np.zeros((4, 1))

    def setup_camera(self) -> Optional[cv2.VideoCapture]:
        """Try multiple camera indices and configure optimally"""
        for camera_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    return self._configure_camera(cap)
                cap.release()
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Camera {camera_id} failed: {e}")
                continue

        print("Error: No camera available!")
        return None

    def _configure_camera(self, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        """Configure camera with optimal settings"""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize camera matrix with actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        focal_length = actual_width
        center = (actual_width // 2, actual_height // 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        if self.config.debug_mode:
            print(f"Camera initialized: {actual_width}x{actual_height}")
        return cap

    def get_face_landmarks_2d(self, landmarks, frame_width: int, frame_height: int) -> Optional[np.ndarray]:
        """Extract 2D points for PnP algorithm - requires minimum 6 points"""
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

            # Ensure we have at least 6 points for PnP
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
        """Validate pose with simplified logic"""
        # Check distance bounds
        if not (self.config.distance_bounds[0] <= distance <= self.config.distance_bounds[1]):
            return False

        # Check for excessive jumps
        if self.previous_translation is not None:
            jump = np.linalg.norm(translation_vector - self.previous_translation)
            if jump > self.config.max_jump_threshold:
                self.consecutive_bad_frames += 1
                return self.consecutive_bad_frames < self.config.max_bad_frames

        self.consecutive_bad_frames = 0
        return True

    def smooth_translation(self, new_translation: np.ndarray) -> np.ndarray:
        """Simple exponential smoothing filter"""
        if self.previous_translation is None:
            self.previous_translation = new_translation.copy()
            return new_translation

        # Single smoothing factor - much simpler than adaptive
        alpha = self.config.smoothing_factor
        smoothed = alpha * self.previous_translation + (1 - alpha) * new_translation

        self.previous_translation = smoothed.copy()
        return smoothed

    def calculate_head_pose(self, landmarks, frame_width: int, frame_height: int) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], float]:
        """Calculate head pose using PnP with filtering"""
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
                distance = np.linalg.norm(translation_vector)

                if not self.is_valid_pose(translation_vector, distance):
                    # Return previous valid pose if available
                    if (self.consecutive_bad_frames >= self.config.max_bad_frames and
                            self.previous_translation is not None):
                        return None, self.previous_translation, np.linalg.norm(self.previous_translation)

                smoothed_translation = self.smooth_translation(translation_vector)
                smoothed_distance = np.linalg.norm(smoothed_translation)

                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                return rotation_matrix, smoothed_translation, smoothed_distance

        except Exception as e:
            if self.config.debug_mode:
                print(f"PnP calculation error: {e}")

        return None, None, 0.0

    def calculate_movement_offset(self, current_translation: np.ndarray) -> Tuple[float, float, float]:
        """Calculate offset from neutral position"""
        if self.neutral_position is None:
            return 0.0, 0.0, 0.0

        offset = current_translation - self.neutral_position
        return float(offset[0]), float(offset[1]), float(offset[2])

    def calibrate(self, translation_vector: np.ndarray) -> None:
        """Set neutral position for calibration"""
        self.neutral_position = translation_vector.copy()
        self.calibrated = True
        print("✓ Calibration complete!")

    def send_to_game(self, x: float, y: float, z: float) -> None:
        """Send data to game - placeholder for IPC implementation"""
        if self.config.debug_mode:
            print(f"→ Game: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")

    def draw_debug_info(self, frame: np.ndarray, landmarks, offset_x: float, offset_y: float, offset_z: float,
                        distance: float) -> None:
        """Draw debug information on frame"""
        if not self.config.debug_mode:
            return

        frame_height, frame_width = frame.shape[:2]

        # Draw essential landmarks
        for name, idx in self.essential_landmarks.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                color = (0, 255, 0) if self.calibrated else (0, 0, 255)
                cv2.circle(frame, (x, y), 3, color, -1)

        # Display tracking info
        if self.calibrated:
            total_movement = np.sqrt(offset_x ** 2 + offset_y ** 2 + offset_z ** 2)

            info_lines = [
                f"Distance: {distance:.1f}mm",
                f"Offset X: {offset_x:.1f}mm",
                f"Offset Y: {offset_y:.1f}mm",
                f"Offset Z: {offset_z:.1f}mm",
                f"Movement: {total_movement:.1f}mm",
                f"Status: {'STABLE' if total_movement < 100 else 'ACTIVE'}"
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 60 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Distance: {distance:.1f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def draw_axes(self, frame: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> None:
        """Draw 3D coordinate axes for head orientation"""
        if self.camera_matrix is None:
            return

        try:
            # 3D axis points (30mm length)
            axis_points_3d = np.array([
                [0, -30, 5],  # Origin at nose tip (matches our reference point)
                [30, -30, 5],  # X-axis (red)
                [0, -60, 5],  # Y-axis (green)
                [0, -30, 35]  # Z-axis (blue)
            ], dtype=np.float64)

            # Project to image plane
            axis_points_2d, _ = cv2.projectPoints(
                axis_points_3d,
                cv2.Rodrigues(rotation_matrix)[0],
                translation_vector,
                self.camera_matrix,
                self.dist_coefficients
            )

            points_2d = axis_points_2d.reshape(-1, 2).astype(int)
            origin = tuple(points_2d[0])

            # Draw axes
            cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 2)  # X-red
            cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 2)  # Y-green
            cv2.line(frame, origin, tuple(points_2d[3]), (255, 0, 0), 2)  # Z-blue

        except Exception as e:
            if self.config.debug_mode:
                print(f"Error drawing axes: {e}")

    def run(self) -> None:
        """Main tracking loop"""
        cap = self.setup_camera()
        if cap is None:
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("Head Tracker started!")
        print("Press 'c' to calibrate")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug mode")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read frame from camera")
                    break

                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for facial_landmarks in results.multi_face_landmarks:
                        # Calculate head pose
                        rotation_matrix, translation_vector, distance = self.calculate_head_pose(
                            facial_landmarks, frame_width, frame_height
                        )

                        if translation_vector is not None:
                            self.current_translation = translation_vector

                            if self.calibrated:
                                # Calculate movement offset
                                offset_x, offset_y, offset_z = self.calculate_movement_offset(translation_vector)

                                # Send to game (convert mm to meters)
                                self.send_to_game(offset_x / 1000, offset_y / 1000, offset_z / 1000)

                                # Draw debug info
                                self.draw_debug_info(frame, facial_landmarks, offset_x, offset_y, offset_z, distance)

                                # Draw 3D axes if rotation available
                                if rotation_matrix is not None and self.config.debug_mode:
                                    self.draw_axes(frame, rotation_matrix, translation_vector)
                            else:
                                # Show distance before calibration
                                if self.config.debug_mode:
                                    cv2.putText(frame, f"Distance: {distance:.1f}mm", (10, 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        else:
                            if self.config.debug_mode:
                                cv2.putText(frame, "Tracking unstable", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    if self.config.debug_mode:
                        cv2.putText(frame, "No face detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Status display
                if self.config.debug_mode:
                    status = "CALIBRATED" if self.calibrated else "PRESS 'C' TO CALIBRATE"
                    color = (0, 255, 0) if self.calibrated else (0, 255, 255)
                    cv2.putText(frame, status, (10, frame_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.imshow("Head Tracker", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.current_translation is not None:
                    self.calibrate(self.current_translation)
                elif key == ord('d'):
                    self.config.debug_mode = not self.config.debug_mode
                    if not self.config.debug_mode:
                        cv2.destroyAllWindows()
                    print(f"Debug mode: {'ON' if self.config.debug_mode else 'OFF'}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Entry point with configuration examples"""
    # Example: High performance mode
    # config = TrackerConfig(
    #     debug_mode=False,
    #     smoothing_factor=0.1,
    #     frame_width=320,
    #     frame_height=240
    # )

    # Example: High precision mode
    # config = TrackerConfig(
    #     smoothing_factor=0.3,
    #     max_jump_threshold=100.0,
    #     min_detection_confidence=0.8
    # )

    # Default configuration
    config = TrackerConfig()

    tracker = HeadTracker(config)
    tracker.run()


if __name__ == "__main__":
    main()