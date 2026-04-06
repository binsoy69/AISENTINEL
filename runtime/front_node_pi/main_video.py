#!/usr/bin/env python3
"""Front-node runtime entrypoint for video-file processing."""

from __future__ import annotations

import argparse
import sys

import cv2

from runtime_config import (
    DEFAULT_VIDEO_CONFIG_PATH,
    load_runtime_config,
    resolve_cli_path,
)
from runtime_support import (
    apply_behavior_config,
    configure_runtime_paths,
    load_runtime_modules,
    require_detection_environment,
    resolve_video_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AISENTINEL front-node video runtime (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runtime/front_node_pi/main_video.py
  python runtime/front_node_pi/main_video.py --config runtime/front_node_pi/config_video.ini
  python runtime/front_node_pi/main_video.py --video test-videos/front.mp4
  python runtime/front_node_pi/main_video.py --object-model models/custom.hef
        """,
    )
    parser.add_argument("--config", default=None, help="Optional path to a video runtime INI config file.")
    parser.add_argument("--video", default=None, help="Optional video path. If omitted, the file dialog flow is used.")
    parser.add_argument("--pose-model", default=None, help="Override the pose HEF model path from config.")
    parser.add_argument("--hand-model", default=None, help="Override the hand HEF model path from config.")
    parser.add_argument("--object-model", default=None, help="Override the phone/cheat-sheet HEF model path from config.")
    parser.add_argument("--pose-confidence", type=float, default=None, help="Override the pose/person confidence threshold from config.")
    parser.add_argument("--object-confidence", type=float, default=None, help="Override the object confidence threshold from config.")
    parser.add_argument("--port", type=int, default=None, help="Override the Flask web-stream port from config.")
    parser.add_argument("--calibration-file", default=None, help="Optional saved setup JSON. If omitted, config/auto/manual flow is used.")
    parser.add_argument("--ignore-saved-calibration", action="store_true", help="Force manual setup even if a saved setup exists.")
    return parser


def _require_existing_file(path, label, color):
    if not path.exists():
        print(f"{color.RED}[ERROR] {label} not found: {path}{color.RESET}")
        sys.exit(1)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config, DEFAULT_VIDEO_CONFIG_PATH)
    modules = load_runtime_modules()
    configure_runtime_paths(modules, config)
    apply_behavior_config(modules, config)

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod
    hands_mod = modules.hands_mod
    obj_mod = modules.obj_mod
    pass_mod = modules.pass_mod

    pose_path = resolve_cli_path(args.pose_model) or config.pose_model
    hand_path = resolve_cli_path(args.hand_model) or config.hand_model
    object_path = resolve_cli_path(args.object_model) or config.object_model
    pose_confidence = args.pose_confidence if args.pose_confidence is not None else config.pose_confidence
    object_confidence = args.object_confidence if args.object_confidence is not None else config.object_confidence
    port = args.port if args.port is not None else config.port
    requested_video = resolve_cli_path(args.video)

    print()
    print("=" * 78)
    print("  AISENTINEL - Front-Node Video Runtime")
    print(f"  Config          : {config.config_path}")
    print("  Runtime logic   : runtime/front_node_pi/front_node_all_behavior_pi.py")
    print(f"  Pose model      : {pose_path}")
    print(f"  Hand model      : {hand_path}")
    print(f"  Object model    : {object_path}")
    print(f"  Evidence root   : {config.evidence_root}")
    print(f"  Setup profiles  : {config.setup_profile_dir}")
    print("  Flow            : video -> ROI -> assignment -> desk lines -> stream")
    print("=" * 78)
    print()

    try:
        require_detection_environment(modules)
    except RuntimeError as exc:
        print(f"{head_mod.TC.RED}[ERROR] {exc}{head_mod.TC.RESET}")
        sys.exit(1)

    _require_existing_file(pose_path, "Pose HEF model", head_mod.TC)
    _require_existing_file(hand_path, "Hand HEF model", head_mod.TC)
    _require_existing_file(object_path, "Object HEF model", head_mod.TC)

    video_path = resolve_video_path(requested_video, config, pass_mod, head_mod)
    if video_path is None:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not video_path.is_file():
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)
    head_mod.log_info(f"Selected: {video_path}")

    shared_vdevice = None
    pose_estimator = None
    hand_detector = None
    object_detector = None
    cap = None

    try:
        shared_vdevice = hands_mod.VDevice()
        head_mod.log_info("Hailo VDevice created (shared across all models).")

        pose_estimator = combined_mod.SharedHailoPoseEstimator(
            str(pose_path),
            conf_threshold=pose_confidence,
            vdevice=shared_vdevice,
        )
        hand_detector = hands_mod.HailoObjectDetector(
            str(hand_path),
            class_names=hands_mod.HAND_MODEL_CLASS_NAMES,
            conf_threshold=hands_mod.HAND_CONFIDENCE,
            vdevice=shared_vdevice,
        )
        object_detector = obj_mod.HailoObjectDetector(
            str(object_path),
            conf_threshold=object_confidence,
            vdevice=shared_vdevice,
        )

        print(f"\n{head_mod.TC.BOLD}Object model classes:{head_mod.TC.RESET}")
        for idx, name in obj_mod.CLASS_NAMES.items():
            role = "  << ALERT" if name in obj_mod.ALERT_CLASSES else "  << IGNORED"
            thresh = obj_mod.CONFIDENCE_THRESHOLDS.get(name, "-")
            print(f"  [{idx}] {name} (thresh={thresh}){role}")

        print(f"\n{head_mod.TC.BOLD}Hand model classes:{head_mod.TC.RESET}")
        for idx, name in hands_mod.HAND_MODEL_CLASS_NAMES.items():
            role = "  << USED" if name == hands_mod.CLASS_HAND else "  << IGNORED"
            print(f"  [{idx}] {name}{role}")
        print()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
            sys.exit(1)

        ret, first_frame = cap.read()
        if not ret:
            print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
        head_mod.log_info(f"Video resolution: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        explicit_calibration = resolve_cli_path(args.calibration_file)
        calibration_path = None
        if explicit_calibration is not None:
            calibration_path = explicit_calibration
            if not calibration_path.exists():
                print(f"{head_mod.TC.RED}[ERROR] Setup file not found: {calibration_path}{head_mod.TC.RESET}")
                sys.exit(1)
        elif not args.ignore_saved_calibration:
            if config.video_source.default_setup_profile is not None:
                if config.video_source.default_setup_profile.exists():
                    calibration_path = config.video_source.default_setup_profile
                else:
                    head_mod.log_info("Configured setup profile not found. Falling back to auto/manual setup.")
            if calibration_path is None and config.video_source.auto_use_saved_setup:
                auto_calibration = setup_io.default_setup_profile_path(video_path)
                if auto_calibration.exists():
                    calibration_path = auto_calibration

        setup_bundle = None
        tracker = combined_mod.ReacquiringLockedIoUTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_lost=config.tracking.max_lost,
        )

        if calibration_path is not None:
            try:
                head_mod.log_info(f"Loading saved setup: {calibration_path}")
                setup_bundle = combined_mod.load_setup_from_profile(
                    calibration_path,
                    first_frame,
                    pose_estimator,
                    tracker,
                )
            except Exception as exc:
                if explicit_calibration is not None:
                    print(f"{head_mod.TC.RED}[ERROR] Failed to load setup file: {calibration_path}{head_mod.TC.RESET}")
                    print(str(exc))
                    sys.exit(1)
                head_mod.log_info(f"Saved setup could not be used ({exc}). Falling back to manual setup.")
                tracker = combined_mod.ReacquiringLockedIoUTracker(
                    iou_threshold=config.tracking.iou_threshold,
                    max_lost=config.tracking.max_lost,
                )

        if setup_bundle is None:
            tracker = combined_mod.ReacquiringLockedIoUTracker(
                iou_threshold=config.tracking.iou_threshold,
                max_lost=config.tracking.max_lost,
            )
            setup_bundle = combined_mod.run_manual_setup(
                first_frame,
                pose_estimator,
                tracker,
                disp_scale,
                hand_detector=hand_detector,
                object_detector=object_detector,
            )
            if setup_bundle is None:
                sys.exit(0)

        combined_mod.start_web_server(port)
        head_mod.log_info(f"Web stream at http://{combined_mod.get_local_ip()}:{port}")
        head_mod.log_info("Starting all-behavior detection...")

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
            str(video_path),
            port,
            roi_polygon=setup_bundle["roi_polygon"],
            source_mode="video",
        )
    finally:
        if cap is not None:
            cap.release()
        if hasattr(pose_estimator, "close"):
            pose_estimator.close()
        if hasattr(hand_detector, "close"):
            hand_detector.close()
        if hasattr(object_detector, "close"):
            object_detector.close()

    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
