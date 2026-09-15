import os

import cv2

from backend.vision.pose_analyzer import PoseAnalyzer
from backend.vision.object_detector import ObjectDetector

from backend.features.football_features import (
    extract_football_features
)

from backend.features.running_features import (
    extract_running_features
)

from backend.features.ball_control_features import (
    extract_ball_control_features
)

from backend.features.kicking_features import (
    extract_kicking_features
)

from backend.features.kick_event_features import (
    extract_kick_events
)

from backend.sports.football import (
    analyze_football
)


def _fallback_football_analysis(error):
    analysis = analyze_football({}, {}, {}, {}, {})
    analysis["analysis_mode"] = "degraded"
    analysis["warning"] = (
        "Pose analysis is unavailable in the current server runtime. "
        "A baseline result was returned; redeploy the Render Docker "
        "configuration to enable full video analysis."
    )
    analysis["runtime_error"] = str(error)
    analysis["pose_quality"] = {
        "total_video_frames": 0,
        "processed_frames": 0,
        "pose_detected_frames": 0,
        "detection_rate": 0.0,
        "original_fps": 0.0,
        "analysis_fps": 0,
    }
    analysis["object_detection_quality"] = {
        "detector": "disabled",
        "ball_frames_tested": 0,
        "ball_frames_detected": 0,
        "ball_detection_rate": 0.0,
    }
    return analysis


def analyze_football_video(
    video_path,
    target_fps=5
):
    """
    Run the complete Football prototype analysis.

    Pipeline:

    Video
    -> MediaPipe Pose
    -> YOLO Sports-Ball Detection
    -> Football Features
    -> Running Features
    -> Ball-Control Features
    -> Kicking Features
    -> Kick Events
    -> Heuristic Football Scoring
    """

    # --------------------------------------------------
    # 1. POSE ANALYSIS
    # --------------------------------------------------

    try:
        pose_analyzer = PoseAnalyzer(
            target_fps=target_fps
        )
    except OSError as error:
        if "libGLESv2.so.2" not in str(error):
            raise
        return _fallback_football_analysis(error)

    try:
        pose_result = pose_analyzer.analyze_video(
            video_path
        )

    finally:
        pose_analyzer.close()

    if pose_result["processed_frames"] < 3:
        raise ValueError(
            "The video does not contain enough processed frames."
        )

    if pose_result["pose_detected_frames"] == 0:
        raise ValueError(
            "No usable human pose was detected."
        )

    # --------------------------------------------------
    # 2. YOLO BALL DETECTION
    # --------------------------------------------------

    ball_detection_frames = []
    if os.getenv("ENABLE_OBJECT_DETECTION", "false").lower() == "true":
        detector = ObjectDetector()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not open video for object detection."
            )

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            cap.release()
            raise ValueError("Invalid video FPS.")

        frame_interval = max(1, round(original_fps / target_fps))
        frame_number = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            if frame_number % frame_interval != 0:
                continue

            ball_detection_frames.append({
                "frame": frame_number,
                "detections": detector.detect_sports_ball(frame),
            })

        cap.release()

    # --------------------------------------------------
    # 3. BASE FOOTBALL FEATURES
    # --------------------------------------------------

    features = extract_football_features(
        pose_result["landmarks"],
        ball_detection_frames
    )

    # --------------------------------------------------
    # 4. RUNNING FEATURES
    # --------------------------------------------------

    running_features = extract_running_features(
        pose_result["landmarks"]
    )

    # --------------------------------------------------
    # 5. BALL CONTROL FEATURES
    # --------------------------------------------------

    ball_control_features = (
        extract_ball_control_features(
            pose_result["landmarks"],
            ball_detection_frames
        )
    )

    # --------------------------------------------------
    # 6. KICKING POSTURE FEATURES
    # --------------------------------------------------

    kicking_features = extract_kicking_features(
        pose_result["landmarks"]
    )

    # --------------------------------------------------
    # 7. KICK EVENT DETECTION
    # --------------------------------------------------

    kick_event_features = extract_kick_events(
        pose_result["landmarks"],
        ball_detection_frames
    )

    # --------------------------------------------------
    # 8. FOOTBALL SCORING
    # --------------------------------------------------

    analysis = analyze_football(
        features,
        running_features=running_features,
        ball_control_features=ball_control_features,
        kicking_features=kicking_features,
        kick_events=kick_event_features
    )

    # --------------------------------------------------
    # 9. POSE QUALITY INFORMATION
    # --------------------------------------------------

    analysis["pose_quality"] = {

        "total_video_frames":
            pose_result["total_frames"],

        "processed_frames":
            pose_result["processed_frames"],

        "pose_detected_frames":
            pose_result["pose_detected_frames"],

        "detection_rate":
            pose_result["detection_rate"],

        "original_fps":
            pose_result["original_fps"],

        "analysis_fps":
            pose_result["target_fps"]
    }

    # --------------------------------------------------
    # 10. OBJECT DETECTION QUALITY
    # --------------------------------------------------

    ball_frames_tested = features[
        "ball_frames_tested"
    ]

    ball_frames_detected = features[
        "ball_frames_detected"
    ]

    analysis["object_detection_quality"] = {

        "detector":
            "pretrained_yolo_sports_ball",

        "ball_frames_tested":
            ball_frames_tested,

        "ball_frames_detected":
            ball_frames_detected,

        "ball_detection_rate":
            features["ball_detection_rate"],

        "warning": (
            "The pretrained YOLO model detects the "
            "generic 'sports ball' class. It is not "
            "a custom-trained football detector."
        )
    }

    # --------------------------------------------------
    # 11. RETURN FINAL RESULT
    # --------------------------------------------------

    return analysis