#!/usr/bin/env python3
"""
Simple YOLO Pose Estimation Demo

Run pose estimation on a webcam feed or a video file.

Usage:
    python pose_demo.py                  # Interactive menu
    python pose_demo.py --webcam         # Use webcam directly (default cam 0)
    python pose_demo.py --webcam 1       # Use webcam at index 1
    python pose_demo.py --video path.mp4 # Use a video file directly

Controls:
    q / ESC  - Quit
    s        - Save current frame as screenshot

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────
MODEL_NAME = "yolo11n-pose.pt"
WINDOW_NAME = "YOLO Pose Estimation"
SCREENSHOT_DIR = "screenshots"


def select_source():
    """Interactive menu to choose webcam or video file."""
    print("\n" + "=" * 50)
    print("  YOLO Pose Estimation Demo")
    print("=" * 50)
    print("\n  Select input source:\n")
    print("  [1] Webcam")
    print("  [2] Video file")
    print("  [q] Quit")
    print()

    while True:
        choice = input("  Enter choice (1/2/q): ").strip().lower()

        if choice == "1":
            cam_idx = input("  Camera index [0]: ").strip()
            cam_idx = int(cam_idx) if cam_idx else 0
            return cam_idx

        elif choice == "2":
            video_path = input("  Enter video file path: ").strip().strip('"').strip("'")
            if not os.path.isfile(video_path):
                print(f"  [ERROR] File not found: {video_path}")
                continue
            return video_path

        elif choice == "q":
            print("  Bye!")
            sys.exit(0)

        else:
            print("  Invalid choice. Try again.")


def run_pose_estimation(source, model_path=MODEL_NAME):
    """Run pose estimation on the given source (camera index or video path)."""
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded.\n")

    # Open video source
    is_webcam = isinstance(source, int)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        label = f"webcam {source}" if is_webcam else source
        print(f"[ERROR] Cannot open {label}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_label = f"Webcam {source}" if is_webcam else Path(source).name

    # For video files, scale down to 640px wide for faster processing
    scale = 1.0
    if not is_webcam and width > 640:
        scale = 640 / width
        width = 640
        height = int(height * scale)
        print(f"Source: {source_label} (scaled to {width}x{height})")
    else:
        print(f"Source: {source_label}")

    print(f"Resolution: {width}x{height} @ {fps:.0f} FPS")
    print(f"\nControls: [q/ESC] Quit  |  [s] Screenshot\n")

    # Create screenshot directory
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                print("[ERROR] Lost webcam feed.")
                break
            else:
                print("Video ended.")
                break

        frame_count += 1

        # Scale down video frames
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height))

        # Run pose estimation
        results = model(frame, verbose=False)

        # Draw results on frame
        annotated = results[0].plot()

        # Calculate and display FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Display person count
        num_persons = len(results[0].keypoints) if results[0].keypoints is not None else 0
        cv2.putText(
            annotated,
            f"Persons: {num_persons}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Show frame
        cv2.imshow(WINDOW_NAME, annotated)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"pose_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, annotated)
            print(f"Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({current_fps:.1f} FPS avg)")


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose Estimation Demo")
    parser.add_argument(
        "--webcam",
        nargs="?",
        const=0,
        type=int,
        metavar="INDEX",
        help="Use webcam (default index: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Path to a video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Path to YOLO pose model (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    # Determine source
    if args.video:
        if not os.path.isfile(args.video):
            print(f"[ERROR] File not found: {args.video}")
            sys.exit(1)
        source = args.video
    elif args.webcam is not None:
        source = args.webcam
    else:
        source = select_source()

    run_pose_estimation(source, model_path=args.model)


if __name__ == "__main__":
    main()
