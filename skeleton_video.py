"""Step 2: Free video → MediaPipe Pose skeleton detection.

Processes baseball video clips using MediaPipe Pose to detect 33 body keypoints,
generates skeleton overlay images, and exports keypoint coordinates to CSV.

Usage:
    python skeleton_video.py --input data/videos/batting.mp4
    python skeleton_video.py --input data/videos/pitching.mp4 --output data/output/pitch_skeleton

Output:
    {output}_overlay.png     — Skeleton overlay on a key frame
    {output}_keypoints.csv   — All keypoint coordinates per frame
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe landmark names (33 keypoints)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def process_video(video_path, output_prefix, max_frames=300):
    """Process video with MediaPipe Pose and extract keypoints."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Subsample for RAM-constrained environments
    step = max(1, total_frames // max_frames)

    all_keypoints = []
    best_frame = None
    best_frame_overlay = None
    best_visibility = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                avg_vis = np.mean([lm.visibility for lm in landmarks])

                for i, lm in enumerate(landmarks):
                    all_keypoints.append({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps if fps > 0 else 0,
                        "landmark_id": i,
                        "landmark_name": LANDMARK_NAMES[i],
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    })

                # Track best frame (highest average visibility)
                if avg_vis > best_visibility:
                    best_visibility = avg_vis
                    best_frame = rgb.copy()
                    overlay = frame.copy()
                    mp_drawing.draw_landmarks(
                        overlay,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    best_frame_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            processed += 1
            frame_idx += 1

            if processed % 50 == 0:
                print(f"  Processed {processed} frames...")

    cap.release()
    print(f"  Total processed: {processed} frames, keypoints detected in {len(all_keypoints) // 33} frames")

    return all_keypoints, best_frame, best_frame_overlay, fps


def save_results(all_keypoints, best_frame, best_frame_overlay, output_prefix):
    """Save keypoint CSV and overlay images."""
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        csv_path = f"{output_prefix}_keypoints.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Keypoints CSV: {csv_path} ({len(df)} rows)")

    # Save overlay image
    if best_frame_overlay is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(best_frame)
        axes[0].set_title("Original Frame")
        axes[0].axis("off")

        axes[1].imshow(best_frame_overlay)
        axes[1].set_title("MediaPipe Pose Detection")
        axes[1].axis("off")

        img_path = f"{output_prefix}_overlay.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Overlay image: {img_path}")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose skeleton detection on video")
    parser.add_argument("--input", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output prefix (default: data/output/<video_name>)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to process (subsamples if video is longer)")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        print("Download a free baseball video from https://www.pexels.com/search/videos/baseball/")
        print("Save it to data/videos/ and try again.")
        return

    if args.output:
        output_prefix = args.output
    else:
        output_prefix = f"data/output/{video_path.stem}"

    keypoints, best_frame, overlay, fps = process_video(video_path, output_prefix, args.max_frames)
    save_results(keypoints, best_frame, overlay, output_prefix)
    print("\nDone!")


if __name__ == "__main__":
    main()
