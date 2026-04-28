#!/usr/bin/env python3
"""PC runner for the updated Pi phone/cheat-sheet logic using Ultralytics .pt models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from front_node_pc_common import (
    CV_WINDOW_PORT_HINT,
    OBJECT_MODEL_CANDIDATES,
    POSE_MODEL_CANDIDATES,
    REPO_ROOT,
    SCRIPT_DIR,
    UltralyticsObjectDetector,
    UltralyticsPoseEstimator,
    close_cv_window,
    enable_cv_window_stream,
    is_readable_checkpoint,
    load_pi_module,
    resolve_model_path,
)


obj_mod = load_pi_module("front_node_cellphone_cheat_pi")
obj_mod.EVIDENCE_DIR = SCRIPT_DIR / "evidence_obj"

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm")
MODEL_SUFFIXES = (".pt",)
FILE_DIALOG_REQUEST = "__AISENTINEL_FILE_DIALOG__"


def _display_path(value):
    path = Path(value)
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def _rglob_files(root, suffixes):
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ],
        key=lambda path: str(path).lower(),
    )


def _dedupe_paths(paths):
    seen = set()
    unique = []
    for value in paths:
        path = Path(value)
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _readable_model_paths(paths):
    readable = []
    for path in _dedupe_paths(paths):
        if path.is_file() and is_readable_checkpoint(path):
            readable.append(path)
    return readable


def _discover_videos():
    found = []
    for root in (REPO_ROOT / "test-videos", SCRIPT_DIR, REPO_ROOT / "tests", REPO_ROOT):
        found.extend(_rglob_files(root, VIDEO_SUFFIXES))
    return _dedupe_paths(found)


def _discover_pose_models():
    discovered = list(POSE_MODEL_CANDIDATES)
    discovered.extend(
        path
        for path in _rglob_files(REPO_ROOT, MODEL_SUFFIXES)
        if "pose" in str(path).lower()
    )
    return _readable_model_paths(discovered)


def _discover_object_models():
    discovered = list(OBJECT_MODEL_CANDIDATES)
    discovered.extend(
        path
        for path in _rglob_files(REPO_ROOT, MODEL_SUFFIXES)
        if "pose" not in str(path).lower()
    )
    return _readable_model_paths(discovered)


def _input_choice(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def _prompt_numbered_path(title, options, manual_prompt, allow_dialog=False):
    options = list(options)

    print()
    print(f"{obj_mod.TC.BOLD}{title}{obj_mod.TC.RESET}")
    if options:
        for idx, option in enumerate(options, start=1):
            default_marker = " [default]" if idx == 1 else ""
            print(f"  {idx}. {_display_path(option)}{default_marker}")
    else:
        print("  No repository matches found.")

    print("  M. Enter path manually")
    if allow_dialog:
        print("  F. Open file dialog")
    print("  Q. Quit")

    default_hint = " [1]" if options else ""
    valid_choices = (
        "a listed number, M, F, or Q"
        if allow_dialog
        else "a listed number, M, or Q"
    )
    while True:
        choice = _input_choice(f"Select option{default_hint}: ")
        if not choice and options:
            return str(options[0])

        lowered = choice.lower()
        if lowered in ("q", "quit", "exit"):
            return None
        if allow_dialog and lowered in ("f", "file", "dialog"):
            return FILE_DIALOG_REQUEST
        if lowered in ("m", "manual", "path", "p"):
            value = _input_choice(f"{manual_prompt}: ").strip().strip('"').strip("'")
            if value:
                return value
            print("  Please enter a path.")
            continue

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return str(options[index - 1])

        print(f"  Choose {valid_choices}.")


def _prompt_existing_file(title, options, manual_prompt, allow_dialog=False):
    while True:
        selected = _prompt_numbered_path(title, options, manual_prompt, allow_dialog)
        if selected is None or selected == FILE_DIALOG_REQUEST:
            return selected
        if os.path.isfile(selected):
            return selected
        print(f"{obj_mod.TC.RED}[ERROR] File not found: {selected}{obj_mod.TC.RESET}")


def _select_video(video_arg):
    if video_arg:
        return video_arg
    selected = _prompt_existing_file(
        "Choose video",
        _discover_videos(),
        "Enter video file path",
        allow_dialog=True,
    )
    if selected == FILE_DIALOG_REQUEST:
        obj_mod.log_info("Opening file dialog...")
        return obj_mod.select_video_dialog()
    return selected


def _select_pose_model(model_arg):
    if model_arg:
        return model_arg

    options = _discover_pose_models()
    if not any(Path(option).name == "yolo11n-pose.pt" for option in options):
        options.append("yolo11n-pose.pt")

    return _prompt_numbered_path(
        "Choose pose model",
        options,
        "Enter pose model path or Ultralytics model name",
    )


def _select_object_model(model_arg):
    if model_arg:
        return model_arg
    return _prompt_existing_file(
        "Choose phone / cheat-sheet object model",
        _discover_object_models(),
        "Enter object model .pt path",
    )


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

    pose_model_arg = _select_pose_model(args.pose_model)
    if not pose_model_arg:
        obj_mod.log_info("No pose model selected. Exiting.")
        sys.exit(0)

    object_model_arg = _select_object_model(args.object_model)
    if not object_model_arg:
        obj_mod.log_info("No object model selected. Exiting.")
        sys.exit(0)

    pose_model = resolve_model_path(
        pose_model_arg,
        POSE_MODEL_CANDIDATES,
        fallback_name="yolo11n-pose.pt",
    )
    try:
        object_model = resolve_model_path(object_model_arg, OBJECT_MODEL_CANDIDATES)
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
