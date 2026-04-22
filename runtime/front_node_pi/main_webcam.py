#!/usr/bin/env python3
"""Front-node runtime entrypoint for webcam processing."""

from __future__ import annotations

import argparse
import sys
import threading

from runtime_config import (
    DEFAULT_WEBCAM_CONFIG_PATH,
    load_runtime_config,
    resolve_cli_path,
)
from runtime_support import (
    apply_behavior_config,
    configure_runtime_paths,
    get_webcam_source_label,
    load_runtime_modules,
    open_webcam_capture,
    read_webcam_frame,
    require_detection_environment,
)
from sound_monitor import SoundMonitorService


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AISENTINEL front-node webcam runtime (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runtime/front_node_pi/main_webcam.py
  python runtime/front_node_pi/main_webcam.py --config runtime/front_node_pi/config_webcam.ini
  python runtime/front_node_pi/main_webcam.py --calibration-file runtime/front_node_pi/data/setup_profiles/front_webcam_all_behavior_setup.json
        """,
    )
    parser.add_argument("--config", default=None, help="Optional path to a webcam runtime INI config file.")
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


def _resolve_dashboard_calibration_path(args, session_details, config, setup_io, source_label, head_mod):
    explicit_calibration = (
        resolve_cli_path(session_details.get("setup_profile_override"))
        or resolve_cli_path(args.calibration_file)
    )
    calibration_path = None
    if explicit_calibration is not None:
        calibration_path = explicit_calibration
        if not calibration_path.exists():
            raise RuntimeError(f"Setup file not found: {calibration_path}")
    elif not args.ignore_saved_calibration:
        if config.webcam_source.default_setup_profile is not None:
            if config.webcam_source.default_setup_profile.exists():
                calibration_path = config.webcam_source.default_setup_profile
            else:
                head_mod.log_info(
                    "Configured webcam setup profile not found. Falling back to auto/manual setup."
                )
        if calibration_path is None and config.webcam_source.auto_use_saved_setup:
            auto_calibration = setup_io.default_setup_profile_path(source_label)
            if auto_calibration.exists():
                calibration_path = auto_calibration
    return explicit_calibration, calibration_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config, DEFAULT_WEBCAM_CONFIG_PATH)
    modules = load_runtime_modules()
    configure_runtime_paths(modules, config)
    apply_behavior_config(modules, config)

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod
    hands_mod = modules.hands_mod
    obj_mod = modules.obj_mod

    pose_path = resolve_cli_path(args.pose_model) or config.pose_model
    hand_path = resolve_cli_path(args.hand_model) or config.hand_model
    object_path = resolve_cli_path(args.object_model) or config.object_model
    pose_confidence = args.pose_confidence if args.pose_confidence is not None else config.pose_confidence
    object_confidence = args.object_confidence if args.object_confidence is not None else config.object_confidence
    port = args.port if args.port is not None else config.port
    source_label = get_webcam_source_label(config)
    setup_profile_hint = resolve_cli_path(args.calibration_file) or config.webcam_source.default_setup_profile

    print()
    print("=" * 78)
    print("  AISENTINEL - Front-Node Webcam Runtime")
    print(f"  Config          : {config.config_path}")
    print("  Runtime logic   : runtime/front_node_pi/front_node_all_behavior_pi.py")
    print(f"  Webcam source   : {source_label} (index {config.webcam_source.camera_index})")
    print(f"  Pose model      : {pose_path}")
    print(f"  Hand model      : {hand_path}")
    print(f"  Object model    : {object_path}")
    print(f"  Evidence root   : {config.evidence_root}")
    print(f"  Setup profiles  : {config.setup_profile_dir}")
    print("  Flow            : web login -> session setup -> webcam stream")
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

    worker_lock = threading.Lock()
    worker_thread = None

    def run_monitoring_session(session_details):
        shared_vdevice = None
        pose_estimator = None
        hand_detector = None
        object_detector = None
        cap = None
        sound_service = None

        try:
            combined_mod.set_dashboard_status(
                "starting",
                "Opening webcam and loading AI models...",
                monitoring_active=False,
                system_state="starting",
            )

            cap = open_webcam_capture(config, head_mod)
            first_frame = read_webcam_frame(cap)
            if first_frame is None:
                raise RuntimeError("Cannot read webcam frame.")

            width = int(cap.get(3))
            height = int(cap.get(4))
            actual_fps = cap.get(5) or 0.0
            if actual_fps <= 0 or actual_fps > 120:
                actual_fps = config.webcam_source.capture_fps
            disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
            head_mod.log_info(f"Webcam resolution: {width}x{height} @ {actual_fps:.1f} FPS")

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

            explicit_calibration, calibration_path = _resolve_dashboard_calibration_path(
                args,
                session_details,
                config,
                setup_io,
                source_label,
                head_mod,
            )

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
                        raise RuntimeError(f"Failed to load setup file: {calibration_path}\n{exc}") from exc
                    head_mod.log_info(
                        f"Saved setup could not be used ({exc}). Falling back to manual setup."
                    )
                    tracker = combined_mod.ReacquiringLockedIoUTracker(
                        iou_threshold=config.tracking.iou_threshold,
                        max_lost=config.tracking.max_lost,
                    )

            if setup_bundle is None:
                combined_mod.set_dashboard_status(
                    "manual_setup",
                    "Manual calibration is required on the device screen.",
                    monitoring_active=False,
                    system_state="starting",
                )
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
                    combined_mod.set_dashboard_status(
                        "idle",
                        "Monitoring setup was cancelled.",
                        monitoring_active=False,
                        system_state="idle",
                    )
                    return

            combined_mod.set_dashboard_status(
                "running",
                "Monitoring live classroom session.",
                monitoring_active=True,
                system_state="active",
            )
            head_mod.log_info("Starting all-behavior detection...")

            def handle_sound_telemetry(payload: dict) -> None:
                combined_mod.update_dashboard_metrics(
                    sound_db=payload.get("current_db"),
                    sound_threshold_db=payload.get("threshold_db"),
                    sound_status=payload.get("status", "disabled"),
                    sound_over_threshold=bool(payload.get("over_threshold")),
                    sound_updated_at=payload.get("updated_at", ""),
                )

            def handle_sound_incident(payload: dict) -> None:
                combined_mod.record_dashboard_noise_incident(
                    estimated_db=float(payload["estimated_db"]),
                    threshold_db=float(payload["threshold_db"]),
                    source_label=source_label,
                    session_details=session_details,
                )

            sound_service = SoundMonitorService(
                config.sound_sensor,
                on_telemetry=handle_sound_telemetry,
                on_threshold_cross=handle_sound_incident,
                log_fn=head_mod.log_info,
            )
            sound_service.start()

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

            combined_mod.set_dashboard_status(
                "completed",
                "Monitoring session ended.",
                monitoring_active=False,
                system_state="idle",
            )
        except Exception as exc:
            combined_mod.set_dashboard_status(
                "error",
                str(exc),
                monitoring_active=False,
                system_state="error",
                error_message=str(exc),
            )
            print(f"{head_mod.TC.RED}[ERROR] {exc}{head_mod.TC.RESET}")
        finally:
            if cap is not None:
                cap.release()
            if sound_service is not None:
                sound_service.stop()
            if hasattr(pose_estimator, "close"):
                pose_estimator.close()
            if hasattr(hand_detector, "close"):
                hand_detector.close()
            if hasattr(object_detector, "close"):
                object_detector.close()

    def start_monitoring_from_dashboard(session_details):
        nonlocal worker_thread
        with worker_lock:
            if worker_thread is not None and worker_thread.is_alive():
                raise RuntimeError("Monitoring is already running.")
            worker_thread = threading.Thread(
                target=run_monitoring_session,
                args=(session_details,),
                daemon=True,
            )
            worker_thread.start()

    combined_mod.configure_web_dashboard(
        auth_config=config.web_dashboard,
        runtime_mode="webcam",
        source_label=source_label,
        config_path=config.config_path,
        evidence_root=config.evidence_root,
        setup_profile_path=setup_profile_hint,
        start_monitoring_callback=start_monitoring_from_dashboard,
        require_session_setup=True,
    )
    combined_mod.update_dashboard_metrics(
        sound_db=None,
        sound_threshold_db=config.sound_sensor.alert_threshold_db,
        sound_status="idle" if config.sound_sensor.enabled else "disabled",
        sound_over_threshold=False,
        sound_updated_at="",
    )

    server_thread = combined_mod.start_web_server(port)
    head_mod.log_info(f"Dashboard at http://{combined_mod.get_local_ip()}:{port}")
    head_mod.log_info("Waiting for login and session start from the web dashboard...")

    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        head_mod.log_info("Interrupted by user.")


if __name__ == "__main__":
    main()
