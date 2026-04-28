#!/usr/bin/env python3
"""Webcam launcher for the Pi cellphone / cheat-sheet test."""

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

import front_node_cellphone_cheat_pi as obj_mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AISENTINEL - Cellphone / Cheat Sheet Detection (Pi webcam + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/webcam/cellphone_cheat_webcam_pi.py
  python3 tests/tests_on_pi/webcam/cellphone_cheat_webcam_pi.py --camera 1 --port 9090
  python3 tests/tests_on_pi/webcam/cellphone_cheat_webcam_pi.py --pose-model /path/to/pose.hef --model /path/to/object.hef
        """,
    )
    add_webcam_args(parser)
    parser.add_argument("--model", default=None, help=f"Path to object HEF model (default: {obj_mod.OBJ_MODEL_PATH})")
    parser.add_argument("--pose-model", default=None, help=f"Path to pose HEF model (default: {obj_mod.POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080, help="Flask web server port (default: 8080)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Base detection confidence (default: 0.25)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AISENTINEL - Cellphone / Cheat Sheet Detection (Pi webcam + Hailo)")
    print("  Detects: phone | cheat_sheet")
    print("=" * 70)
    print()

    pose_model_arg = obj_mod.pi_ui.select_pose_model(args.pose_model)
    if not pose_model_arg:
        obj_mod.log_info("No pose model selected. Exiting.")
        return
    object_model_arg = obj_mod.pi_ui.select_object_model(args.model)
    if not object_model_arg:
        obj_mod.log_info("No object model selected. Exiting.")
        return

    require_hailo(obj_mod.HAILO_AVAILABLE, obj_mod.TC)
    require_flask(obj_mod.FLASK_AVAILABLE, obj_mod.TC)
    pose_path = Path(pose_model_arg)
    object_path = Path(object_model_arg)
    require_file(pose_path, "Pose HEF model", obj_mod.TC)
    require_file(object_path, "Object HEF model", obj_mod.TC)

    cap, opened_as = open_webcam(args.camera, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    source_label = webcam_source_label(args.camera)
    actual_fps = capture_fps(cap, args.fps)
    first_frame = read_warmup_frame(cap, args.warmup_frames)
    if first_frame is None:
        cap.release()
        raise SystemExit(f"{obj_mod.TC.RED}[ERROR] Cannot read a calibration frame from the webcam.{obj_mod.TC.RESET}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    obj_mod.log_info(f"Opened webcam {opened_as}: {width}x{height} @ {actual_fps:.1f} FPS")

    shared_vdevice = obj_mod.VDevice()
    obj_mod.log_info("Hailo VDevice created (shared between both models).")
    person_detector = obj_mod.HailoPoseEstimator(
        str(pose_path),
        conf_threshold=obj_mod.PERSON_CONFIDENCE,
        vdevice=shared_vdevice,
    )
    detector = obj_mod.HailoObjectDetector(
        str(object_path),
        conf_threshold=args.confidence,
        vdevice=shared_vdevice,
    )

    print(f"\n{obj_mod.TC.BOLD}Model classes:{obj_mod.TC.RESET}")
    for idx, name in detector.class_names.items():
        role = "  << ALERT" if name in obj_mod.ALERT_CLASSES else "  << IGNORED"
        thresh = obj_mod.CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){role}")
    print()

    obj_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = obj_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        obj_mod.log_info("ROI calibration cancelled. Exiting.")
        return
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    obj_mod.log_info("Running person detection on the calibration frame for student assignment...")
    first_student_dets = person_detector.detect_persons(first_frame)
    first_student_dets = obj_mod.filter_detections_by_roi(first_student_dets, roi_polygon)
    tracker = obj_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_student_dets)
    obj_mod.log_info(f"Detected {len(first_student_dets)} students on the calibration frame.")

    student_map = obj_mod.run_assignment_phase(first_frame, first_student_dets, first_track_ids, disp_scale)
    if student_map is None or len(student_map) == 0:
        cap.release()
        obj_mod.log_info("No students assigned. Exiting.")
        return

    tracker.keep_only(set(student_map.keys()))
    obj_mod.start_web_server(args.port)
    obj_mod.log_info(f"Web stream at http://{obj_mod.get_local_ip()}:{args.port}")
    obj_mod.log_info("Starting webcam detection...")
    obj_mod.run_detection(
        cap,
        person_detector,
        detector,
        tracker,
        student_map,
        source_label,
        args.port,
        roi_polygon=roi_polygon,
        source_mode="webcam",
        source_fps=actual_fps,
    )
    cap.release()
    obj_mod.log_info("Done!")


if __name__ == "__main__":
    main()
