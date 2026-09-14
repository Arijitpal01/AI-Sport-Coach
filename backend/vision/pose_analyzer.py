import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseAnalyzer:
    def __init__(self, target_fps=5):
        """
        target_fps:
        Number of video frames per second that we want
        to actually analyze.

        Example:
        Original video = 30 FPS
        target_fps = 5

        We process roughly 5 frames each second
        instead of all 30.
        """

        self.target_fps = target_fps

        model_path = str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "trained_models"
            / "pose_landmarker_full.task"
        )

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not open video."
            )

        original_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if original_fps <= 0:
            cap.release()

            raise ValueError(
                "Invalid video FPS."
            )

        # Example:
        # 30 FPS video / target 5 FPS
        # process every 6th frame
        frame_interval = max(
            1,
            round(
                original_fps
                / self.target_fps
            )
        )

        total_video_frames = 0
        processed_frames = 0
        detected_frames = 0

        extracted_landmarks = []

        selected_landmarks = {
            0: "nose",

            11: "left_shoulder",
            12: "right_shoulder",

            13: "left_elbow",
            14: "right_elbow",

            15: "left_wrist",
            16: "right_wrist",

            23: "left_hip",
            24: "right_hip",

            25: "left_knee",
            26: "right_knee",

            27: "left_ankle",
            28: "right_ankle"
        }

        while True:
            success, frame = cap.read()

            if not success:
                break

            total_video_frames += 1

            # Skip unnecessary frames
            if (
                total_video_frames
                % frame_interval
                != 0
            ):
                continue

            processed_frames += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (
                    total_video_frames
                    / original_fps
                )
                * 1000
            )

            result = (
                self.landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            if not result.pose_landmarks:
                continue

            detected_frames += 1

            pose = result.pose_landmarks[0]

            frame_data = {
                "frame": total_video_frames,
                "timestamp_ms": timestamp_ms,
                "landmarks": {}
            }

            for index, name in (
                selected_landmarks.items()
            ):
                landmark = pose[index]

                frame_data["landmarks"][
                    name
                ] = {
                    "x": round(
                        float(landmark.x),
                        5
                    ),

                    "y": round(
                        float(landmark.y),
                        5
                    ),

                    "z": round(
                        float(landmark.z),
                        5
                    ),

                    "visibility": round(
                        float(
                            landmark.visibility
                        ),
                        5
                    )
                }

            extracted_landmarks.append(
                frame_data
            )

        cap.release()

        if processed_frames > 0:
            detection_rate = (
                detected_frames
                / processed_frames
            )
        else:
            detection_rate = 0

        return {
            "total_frames": total_video_frames,

            "processed_frames": (
                processed_frames
            ),

            "pose_detected_frames": (
                detected_frames
            ),

            "detection_rate": round(
                detection_rate,
                3
            ),

            "original_fps": round(
                original_fps,
                2
            ),

            "target_fps": (
                self.target_fps
            ),

            "landmarks": (
                extracted_landmarks
            )
        }

    def close(self):
        self.landmarker.close()