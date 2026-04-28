#!/usr/bin/env python3
"""PC runner for the final all-behavior flow using three Ultralytics .pt models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from front_node_pc_common import (
    CV_WINDOW_PORT_HINT,
    HAND_MODEL_CANDIDATES,
    OBJECT_MODEL_CANDIDATES,
    POSE_MODEL_CANDIDATES,
    SCRIPT_DIR,
    UltralyticsObjectDetector,
    UltralyticsPoseEstimator,
    canonical_label,
    close_cv_window,
    enable_cv_window_stream,
    load_pi_module,
    patch_module_for_pc,
    resolve_model_path,
)


combined_mod = load_pi_module("front_node_all_behavior_pi")
for _submodule_name in ("head_mod", "pass_mod", "hands_mod", "obj_mod"):
    patch_module_for_pc(getattr(combined_mod, _submodule_name))

head_mod = combined_mod.head_mod
hands_mod = combined_mod.hands_mod
obj_mod = combined_mod.obj_mod
setup_io = combined_mod.setup_io

combined_mod.EVIDENCE_DIR = SCRIPT_DIR / "evidence_combined"
combined_mod.HEAD_EVIDENCE_DIR = combined_mod.EVIDENCE_DIR / "head_behavior"
combined_mod.PASSING_EVIDENCE_DIR = combined_mod.EVIDENCE_DIR / "passing_papers"
combined_mod.HANDS_EVIDENCE_DIR = combined_mod.EVIDENCE_DIR / "hands"
combined_mod.OBJECT_EVIDENCE_DIR = combined_mod.EVIDENCE_DIR / "objects"
setup_io.SETUP_PROFILE_DIR = SCRIPT_DIR / "setup_profiles"


def _select_video(video_arg):
    if video_arg:
        return video_arg
    head_mod.log_info("Opening file dialog...")
    return combined_mod.pass_mod.select_video_dialog()


def _print_detector_classes(title, detector, used_names):
    print(f"\n{head_mod.TC.BOLD}{title}:{head_mod.TC.RESET}")
    for idx, name in sorted(detector.names.items()):
        canonical = canonical_label(name)
        role = "  << USED" if canonical in used_names else "  << IGNORED"
        alias = f" -> {canonical}" if canonical != name else ""
        print(f"  [{idx}] {name}{alias}{role}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - All Behavior Detection (PC + Ultralytics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python front_node_all_behavior_pc.py
  python front_node_all_behavior_pc.py --video path/to/exam.mp4
  python front_node_all_behavior_pc.py --pose-model models/archive/yolo26s-pose.pt
  python front_node_all_behavior_pc.py --hand-model models/archive/front_node/my_model.pt
  python front_node_all_behavior_pc.py --object-model models/archive/yolov11n-sentinel-new/sentinel_new.pt
        """,
    )
    parser.add_argument("--video", default=None, help="Optional path to a video file")
    parser.add_argument("--pose-model", default=None, help="Pose .pt model for person/keypoint detection")
    parser.add_argument("--hand-model", default=None, help="Hand .pt detector model")
    parser.add_argument(
        "--object-model",
        "--model",
        dest="object_model",
        default=None,
        help="Object .pt detector model for phone/cheat_sheet",
    )
    parser.add_argument("--pose-confidence", type=float, default=0.5)
    parser.add_argument("--hand-confidence", type=float, default=hands_mod.HAND_CONFIDENCE)
    parser.add_argument("--object-confidence", "--confidence", dest="object_confidence",
                        type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Path to a saved ROI/assignment/table-line setup JSON",
    )
    parser.add_argument(
        "--ignore-saved-calibration",
        action="store_true",
        help="Force manual setup even if a saved setup JSON exists",
    )
    args = parser.parse_args()

    print()
    print("=" * 78)
    print("  AISENTINEL - All Behavior Detection (PC + Ultralytics)")
    print("  Pose model      : person/keypoint tracking")
    print("  Hand model      : hand-only detection")
    print("  Object model    : phone + cheat_sheet detection")
    print("  Hardware        : PC only, no Hailo")
    print("  Logic           : tests_on_pi front_node_all_behavior_pi")
    print("=" * 78)
    print()

    video_path = _select_video(args.video)
    if not video_path:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)

    try:
        pose_model = resolve_model_path(
            args.pose_model,
            POSE_MODEL_CANDIDATES,
            fallback_name="yolo11n-pose.pt",
        )
        hand_model = resolve_model_path(args.hand_model, HAND_MODEL_CANDIDATES)
        object_model = resolve_model_path(args.object_model, OBJECT_MODEL_CANDIDATES)
    except FileNotFoundError as exc:
        print(f"{head_mod.TC.RED}[ERROR] {exc}{head_mod.TC.RESET}")
        sys.exit(1)

    head_mod.log_info(f"Loading pose model  : {pose_model}")
    pose_estimator = UltralyticsPoseEstimator(
        pose_model,
        conf_threshold=args.pose_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    head_mod.log_info(f"Loading hand model  : {hand_model}")
    hand_detector = UltralyticsObjectDetector(
        hand_model,
        conf_threshold=args.hand_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    head_mod.log_info(f"Loading object model: {object_model}")
    object_detector = UltralyticsObjectDetector(
        object_model,
        conf_threshold=args.object_confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    _print_detector_classes("Hand model classes", hand_detector, {hands_mod.CLASS_HAND})
    _print_detector_classes("Object model classes", object_detector, obj_mod.OBJECT_CLASSES)
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(f"Video resolution: {width}x{height}")

    calibration_path = None
    explicit_calibration = bool(args.calibration_file)
    if explicit_calibration:
        calibration_path = Path(args.calibration_file)
        if not calibration_path.exists():
            cap.release()
            print(f"{head_mod.TC.RED}[ERROR] Setup file not found: {calibration_path}{head_mod.TC.RESET}")
            sys.exit(1)
    elif not args.ignore_saved_calibration:
        auto_calibration = setup_io.default_setup_profile_path(video_path)
        if auto_calibration.exists():
            calibration_path = auto_calibration

    setup_bundle = None
    tracker = combined_mod.ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)

    if calibration_path is not None:
        try:
            head_mod.log_info(f"Loading saved setup: {calibration_path}")
            setup_bundle = combined_mod.load_setup_from_profile(
                calibration_path, first_frame, pose_estimator, tracker
            )
        except Exception as exc:
            if explicit_calibration:
                cap.release()
                print(f"{head_mod.TC.RED}[ERROR] Failed to load setup file: {calibration_path}{head_mod.TC.RESET}")
                print(str(exc))
                sys.exit(1)
            head_mod.log_info(
                f"Saved setup could not be used ({exc}). Falling back to manual setup."
            )
            tracker = combined_mod.ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)

    if setup_bundle is None:
        if calibration_path is not None:
            head_mod.log_info("Falling back to manual setup.")
        tracker = combined_mod.ReacquiringLockedIoUTracker(iou_threshold=0.3, max_lost=60)
        setup_bundle = combined_mod.run_manual_setup(
            first_frame,
            pose_estimator,
            tracker,
            disp_scale,
            hand_detector=hand_detector,
            object_detector=object_detector,
        )
        if setup_bundle is None:
            cap.release()
            sys.exit(0)

    window_name = enable_cv_window_stream(combined_mod, "AISENTINEL - All Behavior")
    head_mod.log_info(f"OpenCV window: {window_name} (press Q or Esc to stop)")
    head_mod.log_info("Starting all-behavior detection...")

    try:
        combined_mod.run_detection(
            cap,
            pose_estimator,
            hand_detector,
            object_detector,
            tracker,
            setup_bundle["student_map"],
            setup_bundle["baseline_yaw_map"],
            setup_bundle["assigned_students"],
            setup_bundle["student_lines"],
            video_path,
            CV_WINDOW_PORT_HINT,
            roi_polygon=setup_bundle["roi_polygon"],
        )
    finally:
        close_cv_window(window_name)
        cap.release()
        pose_estimator.close()
        hand_detector.close()
        object_detector.close()

    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
