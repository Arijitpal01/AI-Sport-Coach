from pathlib import Path


VIDEO_PATH = str(
    Path(__file__).resolve().parent
    / "uploads"
    / "FREESTYLE 😱🔥 FOOTBALL SKILLS ⚽️⭐️ BEACH FOOTBALL "
    "valerikostovofficial - v7skills (1080p, h264, youtube).mp4"
)
MIN_VISIBILITY = 0.70


def landmarks_are_visible(*landmarks):
    return all(
        landmark["visibility"] >= MIN_VISIBILITY
        for landmark in landmarks
    )


def run_angle_summary(video_path):
    from backend.features.angles import calculate_angle, summarize_angles
    from backend.vision.pose_analyzer import PoseAnalyzer

    analyzer = PoseAnalyzer()
    try:
        result = analyzer.analyze_video(video_path)
    finally:
        analyzer.close()

    angle_sets = {
        "Left elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "Right elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "Left knee": ("left_hip", "left_knee", "left_ankle"),
        "Right knee": ("right_hip", "right_knee", "right_ankle"),
    }
    angles = {name: [] for name in angle_sets}
    skipped_measurements = 0

    for frame_data in result["landmarks"]:
        landmarks = frame_data["landmarks"]
        for name, landmark_names in angle_sets.items():
            selected = [landmarks[key] for key in landmark_names]
            if landmarks_are_visible(*selected):
                angles[name].append(calculate_angle(*selected))
            else:
                skipped_measurements += 1

    print("\nFILTERED ANGLE SUMMARY")
    print("======================")
    print("Pose detected frames:", result["pose_detected_frames"])
    print("Detection rate:", result["detection_rate"])
    print("Minimum visibility:", MIN_VISIBILITY)
    print()
    for name, values in angles.items():
        print(f"{name}:", summarize_angles(values))
        print(f"Valid {name.lower()} measurements:", len(values))
    print("Skipped low-visibility measurements:", skipped_measurements)


if __name__ == "__main__":
    run_angle_summary(VIDEO_PATH)
