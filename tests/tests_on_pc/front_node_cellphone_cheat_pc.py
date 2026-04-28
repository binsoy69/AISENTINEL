#!/usr/bin/env python3
"""PC runner for the updated Pi phone/cheat-sheet logic using Ultralytics .pt models."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from front_node_pc_common import (
    CV_WINDOW_PORT_HINT,
    OBJECT_MODEL_CANDIDATES,
    POSE_MODEL_CANDIDATES,
    SCRIPT_DIR,
    UltralyticsObjectDetector,
    UltralyticsPoseEstimator,
    close_cv_window,
    enable_cv_window_stream,
    load_pi_module,
    resolve_model_path,
)


obj_mod = load_pi_module("front_node_cellphone_cheat_pi")
obj_mod.EVIDENCE_DIR = SCRIPT_DIR / "evidence_obj"


def _select_video(video_arg):
    if video_arg:
        return video_arg
    obj_mod.log_info("Opening file dialog...")
    return obj_mod.select_video_dialog()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Phone / Cheat Sheet Detection (PC + Ultralytics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python front_node_cellphone_cheat_pc.py
  python front_node_cellphone_cheat_pc.py --video path/to/exam.mp4
  python front_node_cellphone_cheat_pc.py --pose-model models/archive/yolo26s-pose.pt
  python front_node_cellphone_cheat_pc.py --object-model models/archive/yolov11n-sentinel-new/sentinel_new.pt
        """,
    )
    parser.add_argument("--video", default=None, help="Optional path to a video file")
    parser.add_argument("--pose-model", default=None, help="Path/name for an Ultralytics pose .pt model")
    parser.add_argument("--object-model", "--model", dest="object_model", default=None,
                        help="Path to a .pt detector with phone/cheat_sheet classes")
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--person-confidence", type=float, default=obj_mod.PERSON_CONFIDENCE)
    parser.add_argument("--object-confidence", "--confidence", dest="object_confidence",
                        type=float, default=0.25, help="Base object confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI calibration")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AISENTINEL - Phone / Cheat Sheet Detection (PC + Ultralytics)")
    print("  Logic      : tests_on_pi front_node_cellphone_cheat_pi")
    print("  Hardware   : PC only, no Hailo")
    print("=" * 70)
    print()

    video_path = _select_video(args.video)
    if not video_path:
        obj_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{obj_mod.TC.RED}[ERROR] File not found: {video_path}{obj_mod.TC.RESET}")
        sys.exit(1)

    pose_model = resolve_model_path(
        args.pose_model,
        POSE_MODEL_CANDIDATES,
        fallback_name="yolo11n-pose.pt",
    )
    try:
        object_model = resolve_model_path(args.object_model, OBJECT_MODEL_CANDIDATES)
    except FileNotFoundError as exc:
        print(f"{obj_mod.TC.RED}[ERROR] {exc}{obj_mod.TC.RESET}")
        sys.exit(1)

    obj_mod.log_info(f"Loading pose model: {pose_model}")
    person_detector = UltralyticsPoseEstimator(
        pose_model,
        conf_threshold=args.person_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    obj_mod.log_info(f"Loading object detector: {object_model}")
    detector = UltralyticsObjectDetector(
        object_model,
        conf_threshold=args.object_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    object_classes = ", ".join(f"{idx}:{name}" for idx, name in sorted(detector.names.items()))
    obj_mod.log_info(f"Detector classes: {object_classes}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{obj_mod.TC.RED}[ERROR] Cannot open video: {video_path}{obj_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{obj_mod.TC.RED}[ERROR] Cannot read first frame.{obj_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    obj_mod.log_info(f"Video resolution: {width}x{height}")

    roi_polygon = None
    if not args.no_roi:
        obj_mod.log_info("Draw ROI boundary to limit tracking area, or press S to skip.")
        roi_result = obj_mod.calibrate_roi(first_frame, disp_scale)
        if isinstance(roi_result, str) and roi_result == "CANCEL":
            cap.release()
            obj_mod.log_info("ROI calibration cancelled. Exiting.")
            sys.exit(0)
        roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    obj_mod.log_info("Running person detection on first frame for student assignment...")
    first_student_dets = person_detector.detect_persons(first_frame)
    first_student_dets = obj_mod.filter_detections_by_roi(first_student_dets, roi_polygon)
    tracker = obj_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_student_dets)

    roi_label = " within ROI" if roi_polygon is not None else ""
    obj_mod.log_info(f"Detected {len(first_student_dets)} students{roi_label}.")

    first_obj_dets = [
        d for d in detector.detect(first_frame)
        if d["class_name"] in obj_mod.OBJECT_CLASSES
    ]
    if first_obj_dets:
        obj_mod.log_info(
            "Also detected: "
            + ", ".join(f"{d['class_name']}({d['confidence']:.0%})" for d in first_obj_dets)
        )

    print()
    print(f"  {obj_mod.TC.BOLD}Instructions:{obj_mod.TC.RESET}")
    print("    1. Click on a student bbox to select them")
    print("    2. Type the student number")
    print("    3. Press ENTER to assign")
    print("    4. Repeat, then press S to start")
    print()

    student_map = obj_mod.run_assignment_phase(
        first_frame, first_student_dets, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        obj_mod.log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        obj_mod.log_info("No students assigned. Exiting.")
        sys.exit(0)

    tracker.keep_only(set(student_map.keys()))
    obj_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    window_name = enable_cv_window_stream(obj_mod, "AISENTINEL - Phone / Cheat Sheet")
    obj_mod.log_info(f"OpenCV window: {window_name} (press Q or Esc to stop)")
    obj_mod.log_info("Starting detection...")
    try:
        obj_mod.run_detection(
            cap,
            person_detector,
            detector,
            tracker,
            student_map,
            video_path,
            CV_WINDOW_PORT_HINT,
            roi_polygon=roi_polygon,
        )
    finally:
        close_cv_window(window_name)
        cap.release()
        person_detector.close()
        detector.close()
    obj_mod.log_info("Done!")


if __name__ == "__main__":
    main()
