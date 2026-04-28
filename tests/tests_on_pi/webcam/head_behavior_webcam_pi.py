#!/usr/bin/env python3
"""Webcam launcher for the Pi head-behavior test."""

import argparse
from pathlib import Path

import cv2

from _webcam_common import (
    add_webcam_args,
    capture_fps,
    open_webcam,
    read_warmup_frame,
    require_file,
    require_flask,
    require_hailo,
    webcam_source_label,
)

import front_node_head_behavior_pi as head_mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AISENTINEL - Head Behavior Detection (Pi webcam + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/webcam/head_behavior_webcam_pi.py
  python3 tests/tests_on_pi/webcam/head_behavior_webcam_pi.py --camera 1 --port 9090
  python3 tests/tests_on_pi/webcam/head_behavior_webcam_pi.py --model /path/to/yolo_pose_model.hef
        """,
    )
    add_webcam_args(parser)
    parser.add_argument("--model", default=None, help=f"Path to pose HEF model (default: {head_mod.POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080, help="Flask web server port (default: 8080)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Person detection confidence (default: 0.5)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AISENTINEL - Head Behavior Detection (Pi webcam + Hailo)")
    print("  Detects: Head Tilting | Shoulder Turn")
    print("=" * 70)
    print()

    model_arg = head_mod.pi_ui.select_pose_model(args.model)
    if not model_arg:
        head_mod.log_info("No pose model selected. Exiting.")
        return

    require_hailo(head_mod.HAILO_AVAILABLE, head_mod.TC)
    require_flask(head_mod.FLASK_AVAILABLE, head_mod.TC)
    model_path = Path(model_arg)
    require_file(model_path, "Pose HEF model", head_mod.TC)

    cap, opened_as = open_webcam(
        args.camera,
        args.width,
        args.height,
        args.fps,
        use_mjpg=not args.no_mjpg,
    )
    source_label = webcam_source_label(args.camera)
    actual_fps = capture_fps(cap, args.fps)
    first_frame = read_warmup_frame(cap, args.warmup_frames)
    if first_frame is None:
        cap.release()
        raise SystemExit(f"{head_mod.TC.RED}[ERROR] Cannot read a calibration frame from the webcam.{head_mod.TC.RESET}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(f"Opened webcam {opened_as}: {width}x{height} @ {actual_fps:.1f} FPS")

    estimator = head_mod.HailoPoseEstimator(str(model_path), conf_threshold=args.confidence)
    head_mod.log_info("Running pose detection on the calibration frame for student assignment...")
    first_detections = estimator.detect_pose(first_frame)
    tracker = head_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)

    head_mod.log_info(f"Detected {len(first_detections)} persons.")
    student_map, baseline_yaw_map = head_mod.run_assignment_phase(
        first_frame,
        first_detections,
        first_track_ids,
        disp_scale,
    )
    if student_map is None or len(student_map) == 0:
        cap.release()
        head_mod.log_info("No students assigned. Exiting.")
        return

    tracker.keep_only(set(student_map.keys()))
    head_mod.start_web_server(args.port)
    head_mod.log_info(f"Web stream at http://{head_mod.get_local_ip()}:{args.port}")
    head_mod.log_info("Starting webcam detection...")
    head_mod.run_detection(
        cap,
        estimator,
        tracker,
        student_map,
        source_label,
        args.port,
        baseline_yaw_map=baseline_yaw_map,
        source_mode="webcam",
        source_fps=actual_fps,
    )
    cap.release()
    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
