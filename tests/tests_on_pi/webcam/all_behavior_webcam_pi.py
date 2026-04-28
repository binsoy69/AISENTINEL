#!/usr/bin/env python3
"""Webcam launcher for the Pi all-behavior test."""

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
    webcam_source_label,
)

import front_node_all_behavior_pi as combined_mod
from front_node_test_config import (
    DEFAULT_WEBCAM_CONFIG_PATH,
    add_config_arg,
    apply_all_behavior_config,
    cli_or_config,
    load_test_config,
    path_arg,
)


head_mod = combined_mod.head_mod
hands_mod = combined_mod.hands_mod
obj_mod = combined_mod.obj_mod
setup_io = combined_mod.setup_io


def _require_all_hailo() -> None:
    if (
        not head_mod.HAILO_AVAILABLE
        or not hands_mod.HAILO_AVAILABLE
        or not obj_mod.HAILO_AVAILABLE
    ):
        print(f"{head_mod.TC.RED}[ERROR] hailo_platform is required.{head_mod.TC.RESET}")
        print("Install: sudo apt install hailo-all")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AISENTINEL - All Behavior Detection (Pi webcam + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/tests_on_pi/webcam/all_behavior_webcam_pi.py
  python3 tests/tests_on_pi/webcam/all_behavior_webcam_pi.py --camera 1 --port 9090
  python3 tests/tests_on_pi/webcam/all_behavior_webcam_pi.py --object-confidence 0.4
        """,
    )
    add_webcam_args(parser)
    add_config_arg(parser, DEFAULT_WEBCAM_CONFIG_PATH)
    parser.add_argument("--pose-model", default=None, help=f"Path to pose HEF model (default: {combined_mod.POSE_MODEL_PATH})")
    parser.add_argument("--hand-model", default=None, help=f"Path to hand HEF model (default: {combined_mod.HAND_MODEL_PATH})")
    parser.add_argument("--object-model", "--model", dest="object_model", default=None, help=f"Path to object HEF model (default: {combined_mod.OBJECT_MODEL_PATH})")
    parser.add_argument("--pose-confidence", type=float, default=None, help="Pose/person confidence threshold (default: config)")
    parser.add_argument("--object-confidence", "--confidence", dest="object_confidence", type=float, default=None, help="Base object confidence threshold (default: config)")
    parser.add_argument("--port", type=int, default=None, help="Flask web server port (default: config)")
    parser.add_argument("--calibration-file", default=None, help="Path to a saved ROI/assignment/desk-line setup JSON")
    parser.add_argument("--ignore-saved-calibration", action="store_true", help="Force manual setup even if a saved setup JSON exists")
    args = parser.parse_args()
    config = load_test_config(args.config, DEFAULT_WEBCAM_CONFIG_PATH)
    apply_all_behavior_config(combined_mod, config)
    camera_index = cli_or_config(args.camera, config.webcam_source.camera_index)
    capture_width = cli_or_config(args.width, config.webcam_source.capture_width)
    capture_height = cli_or_config(args.height, config.webcam_source.capture_height)
    requested_fps = cli_or_config(args.fps, config.webcam_source.capture_fps)
    warmup_frames = cli_or_config(args.warmup_frames, config.webcam_source.warmup_frames)
    port = cli_or_config(args.port, config.port)
    pose_model_arg_value = cli_or_config(args.pose_model, path_arg(config.pose_model))
    hand_model_arg_value = cli_or_config(args.hand_model, path_arg(config.hand_model))
    object_model_arg_value = cli_or_config(args.object_model, path_arg(config.object_model))
    pose_confidence = cli_or_config(args.pose_confidence, config.pose_confidence)
    object_confidence = cli_or_config(args.object_confidence, config.object_confidence)

    print()
    print("=" * 78)
    print("  AISENTINEL - All Behavior Detection (Pi webcam + Hailo)")
    print("  Detects: head tilt | shoulder turn | passing papers")
    print("           hands under table | phone | cheat_sheet")
    print("=" * 78)
    print()

    pose_model_arg = combined_mod.pi_ui.select_pose_model(pose_model_arg_value)
    if not pose_model_arg:
        head_mod.log_info("No pose model selected. Exiting.")
        return
    hand_model_arg = combined_mod.pi_ui.select_hand_model(hand_model_arg_value)
    if not hand_model_arg:
        head_mod.log_info("No hand model selected. Exiting.")
        return
    object_model_arg = combined_mod.pi_ui.select_object_model(object_model_arg_value)
    if not object_model_arg:
        head_mod.log_info("No object model selected. Exiting.")
        return

    _require_all_hailo()
    require_flask(combined_mod.FLASK_AVAILABLE, head_mod.TC)
    pose_path = Path(pose_model_arg)
    hand_path = Path(hand_model_arg)
    object_path = Path(object_model_arg)
    require_file(pose_path, "Pose HEF model", head_mod.TC)
    require_file(hand_path, "Hand HEF model", head_mod.TC)
    require_file(object_path, "Object HEF model", head_mod.TC)

    cap, opened_as = open_webcam(camera_index, capture_width, capture_height, requested_fps, use_mjpg=not args.no_mjpg)
    source_label = webcam_source_label(camera_index)
    actual_fps = capture_fps(cap, requested_fps)
    first_frame = read_warmup_frame(cap, warmup_frames)
    if first_frame is None:
        cap.release()
        raise SystemExit(f"{head_mod.TC.RED}[ERROR] Cannot read a calibration frame from the webcam.{head_mod.TC.RESET}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(f"Opened webcam {opened_as}: {width}x{height} @ {actual_fps:.1f} FPS")

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

    calibration_path = None
    explicit_calibration = bool(args.calibration_file)
    if explicit_calibration:
        calibration_path = Path(args.calibration_file)
        if not calibration_path.exists():
            cap.release()
            print(f"{head_mod.TC.RED}[ERROR] Setup file not found: {calibration_path}{head_mod.TC.RESET}")
            raise SystemExit(1)
    elif not args.ignore_saved_calibration:
        candidates = []
        if config.webcam_source.default_setup_profile is not None:
            candidates.append(config.webcam_source.default_setup_profile)
        candidates.append(setup_io.default_setup_profile_path(source_label))
        for auto_calibration in candidates:
            if auto_calibration.exists():
                calibration_path = auto_calibration
                break

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
            if explicit_calibration:
                cap.release()
                print(f"{head_mod.TC.RED}[ERROR] Failed to load setup file: {calibration_path}{head_mod.TC.RESET}")
                print(str(exc))
                raise SystemExit(1) from exc
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
            cap.release()
            head_mod.log_info("Setup cancelled. Exiting.")
            return

    combined_mod.start_web_server(port)
    head_mod.log_info(f"Web stream at http://{combined_mod.get_local_ip()}:{port}")
    head_mod.log_info("Starting all-behavior webcam detection...")
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
        source_label,
        port,
        roi_polygon=setup_bundle["roi_polygon"],
        source_mode="webcam",
        source_fps=actual_fps,
    )
    cap.release()
    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
