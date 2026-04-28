#!/usr/bin/env python3
"""Webcam launcher for the Pi hands-under-table test."""

import argparse
from pathlib import Path

import cv2
import numpy as np

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

import front_node_hands_under_table_pi as hands_mod
from front_node_test_config import (
    DEFAULT_WEBCAM_CONFIG_PATH,
    add_config_arg,
    apply_hands_config,
    cli_or_config,
    load_test_config,
    path_arg,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AISENTINEL - Hands Under Table Detection (Pi webcam + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/webcam/hands_under_table_webcam_pi.py
  python3 tests/tests_on_pi/webcam/hands_under_table_webcam_pi.py --camera 1 --port 9090
  python3 tests/tests_on_pi/webcam/hands_under_table_webcam_pi.py --pose-model /path/to/pose.hef --hand-model /path/to/hand.hef
        """,
    )
    add_webcam_args(parser)
    add_config_arg(parser, DEFAULT_WEBCAM_CONFIG_PATH)
    parser.add_argument("--pose-model", default=None, help=f"Path to pose HEF model (default: {hands_mod.POSE_MODEL_PATH})")
    parser.add_argument("--hand-model", default=None, help=f"Path to hand HEF model (default: {hands_mod.HAND_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=None, help="Flask web server port (default: config)")
    args = parser.parse_args()
    config = load_test_config(args.config, DEFAULT_WEBCAM_CONFIG_PATH)
    apply_hands_config(hands_mod, config)
    camera_index = cli_or_config(args.camera, config.webcam_source.camera_index)
    capture_width = cli_or_config(args.width, config.webcam_source.capture_width)
    capture_height = cli_or_config(args.height, config.webcam_source.capture_height)
    requested_fps = cli_or_config(args.fps, config.webcam_source.capture_fps)
    warmup_frames = cli_or_config(args.warmup_frames, config.webcam_source.warmup_frames)
    port = cli_or_config(args.port, config.port)
    pose_model_arg_value = cli_or_config(args.pose_model, path_arg(config.pose_model))
    hand_model_arg_value = cli_or_config(args.hand_model, path_arg(config.hand_model))

    print()
    print("=" * 70)
    print("  AISENTINEL - Hands Under Table Detection (Pi webcam + Hailo)")
    print("  Trigger logic: hands disappear near calibrated table-edge lines")
    print("=" * 70)
    print()

    pose_model_arg = hands_mod.pi_ui.select_pose_model(pose_model_arg_value)
    if not pose_model_arg:
        hands_mod.log_info("No pose model selected. Exiting.")
        return
    hand_model_arg = hands_mod.pi_ui.select_hand_model(hand_model_arg_value)
    if not hand_model_arg:
        hands_mod.log_info("No hand model selected. Exiting.")
        return

    require_hailo(hands_mod.HAILO_AVAILABLE, hands_mod.TC)
    require_flask(hands_mod.FLASK_AVAILABLE, hands_mod.TC)
    pose_path = Path(pose_model_arg)
    hand_path = Path(hand_model_arg)
    require_file(pose_path, "Pose HEF model", hands_mod.TC)
    require_file(hand_path, "Hand HEF model", hands_mod.TC)

    cap, opened_as = open_webcam(camera_index, capture_width, capture_height, requested_fps, use_mjpg=not args.no_mjpg)
    source_label = webcam_source_label(camera_index)
    actual_fps = capture_fps(cap, requested_fps)
    first_frame = read_warmup_frame(cap, warmup_frames)
    if first_frame is None:
        cap.release()
        raise SystemExit(f"{hands_mod.TC.RED}[ERROR] Cannot read a calibration frame from the webcam.{hands_mod.TC.RESET}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    hands_mod.log_info(f"Opened webcam {opened_as}: {width}x{height} @ {actual_fps:.1f} FPS")

    shared_vdevice = hands_mod.VDevice()
    hands_mod.log_info("Hailo VDevice created (shared between both models).")
    person_detector = hands_mod.HailoPoseEstimator(
        str(pose_path),
        conf_threshold=hands_mod.PERSON_CONFIDENCE,
        vdevice=shared_vdevice,
    )
    hand_detector = hands_mod.HailoObjectDetector(
        str(hand_path),
        class_names=hands_mod.HAND_MODEL_CLASS_NAMES,
        conf_threshold=hands_mod.HAND_CONFIDENCE,
        vdevice=shared_vdevice,
    )

    hands_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = hands_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        hands_mod.log_info("Cancelled. Exiting.")
        return
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    hands_mod.log_info("Running pose detection on the calibration frame for student assignment...")
    first_detections = person_detector.detect_persons(first_frame)
    first_detections = hands_mod.filter_detections_by_roi(first_detections, roi_polygon)
    tracker = hands_mod.IoUTracker(
        iou_threshold=config.tracking.iou_threshold,
        max_lost=config.tracking.max_lost,
    )
    first_track_ids = tracker.update(first_detections)

    hands_mod.log_info(f"Detected {len(first_detections)} persons within the ROI.")
    student_map = hands_mod.run_assignment_phase(first_frame, first_detections, first_track_ids, disp_scale)
    if student_map is None or len(student_map) == 0:
        cap.release()
        hands_mod.log_info("No students assigned. Exiting.")
        return

    tracker.keep_only(set(student_map.keys()))
    assigned_students = hands_mod.build_assigned_student_list(first_detections, first_track_ids, student_map)
    if len(assigned_students) == 0:
        cap.release()
        hands_mod.log_info("No assigned students available for line calibration. Exiting.")
        return

    hands_mod.log_info("Now draw one student-side table-edge line for each assigned student.")
    student_lines = hands_mod.calibrate_table_edge_lines(first_frame, assigned_students)
    if student_lines is None:
        cap.release()
        hands_mod.log_info("Table-edge calibration cancelled. Exiting.")
        return

    hands_mod.start_web_server(port)
    hands_mod.log_info(f"Web stream at http://{hands_mod.get_local_ip()}:{port}")
    hands_mod.log_info("Starting webcam detection...")
    hands_mod.run_detection(
        cap,
        person_detector,
        hand_detector,
        tracker,
        student_map,
        assigned_students,
        student_lines,
        source_label,
        port,
        roi_polygon=roi_polygon,
        source_mode="webcam",
        source_fps=actual_fps,
    )
    cap.release()
    hands_mod.log_info("Done!")


if __name__ == "__main__":
    main()
