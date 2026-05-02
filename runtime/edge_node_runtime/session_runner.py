"""Adapter that runs the real front-node Hailo runtime inside the node agent."""

from __future__ import annotations

from dataclasses import replace
import threading

import cv2

from central_dashboard.shared.dto import SessionSpec
from . import runtime_config as front_runtime_config
from . import runtime_support as front_runtime_support
from .behaviors import run_all_behavior_detection
from .capture import default_setup_profile, open_capture_bundle, resolve_calibration_path
from .evidence import (
    build_noise_incident_evidence,
    normalize_front_runtime_detected_incident,
    normalize_front_runtime_incident,
)
from .inference import create_hailo_detectors, require_hailo_runtime
from .preview import CENTRAL_AGENT_PUBLISHES_LOCAL_PREVIEW
from .sound_monitor import SoundMonitorService
from .tracking import create_tracker


def run_front_runtime_session(runtime, session: SessionSpec) -> None:
    """Run the real front-node detector loop for one central-dashboard session."""

    node_config = runtime.config
    runtime_cfg, modules = load_front_runtime_context(node_config)
    require_hailo_runtime(modules)
    runtime.update_sound_telemetry(
        {
            "enabled": runtime_cfg.sound_sensor.enabled,
            "threshold_db": runtime_cfg.sound_sensor.alert_threshold_db,
            "status": "idle" if runtime_cfg.sound_sensor.enabled else "disabled",
            "over_threshold": False,
            "last_error": "",
        }
    )

    combined_mod = modules.combined_mod
    setup_io = modules.setup_io
    head_mod = modules.head_mod
    hands_mod = modules.hands_mod
    obj_mod = modules.obj_mod

    source_label = node_config.camera_label
    setup_profile = default_setup_profile(runtime_cfg, node_config, setup_io)
    combined_mod.configure_web_dashboard(
        auth_config=runtime_cfg.web_dashboard,
        runtime_mode=node_config.source_mode,
        source_label=source_label,
        config_path=runtime_cfg.config_path,
        evidence_root=runtime_cfg.evidence_root,
        setup_profile_path=setup_profile,
        require_session_setup=False,
        session_form_defaults={
            "subject_code": session.subject_code,
            "professor": session.professor,
            "session_date": session.session_date,
            "start_time": session.start_time,
            "end_time": session.end_time,
        },
    )
    combined_mod.begin_dashboard_session(
        {
            "subject_code": session.subject_code,
            "professor": session.professor,
            "session_date": session.session_date,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "video_path": str(node_config.video_path or ""),
            "setup_profile_override": str(setup_profile or ""),
        }
    )

    shared_vdevice = None
    pose_estimator = None
    hand_detector = None
    object_detector = None
    cap = None
    sound_service = None
    latest_raw_frame = {"frame": None}
    latest_raw_frame_lock = threading.Lock()

    try:
        capture_bundle = open_capture_bundle(
            node_config,
            runtime_cfg,
            modules,
        )
        cap = capture_bundle["capture"]
        first_frame = capture_bundle["first_frame"]
        source_display_label = capture_bundle["source_label"]
        actual_fps = capture_bundle["fps"]
        with latest_raw_frame_lock:
            latest_raw_frame["frame"] = first_frame.copy()
        runtime.publish_preview_frames(first_frame, first_frame)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first_frame.shape[1])
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0

        (
            shared_vdevice,
            pose_estimator,
            hand_detector,
            object_detector,
        ) = create_hailo_detectors(
            runtime_cfg,
            modules,
        )
        tracker = create_tracker(runtime_cfg, combined_mod)
        calibration_path = resolve_calibration_path(
            node_config,
            runtime_cfg,
            setup_io,
            source_display_label,
            head_mod,
        )

        setup_bundle = None
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
                head_mod.log_info(
                    f"Saved setup could not be used ({exc}). Falling back to manual setup."
                )
                tracker = create_tracker(runtime_cfg, combined_mod)

        if setup_bundle is None:
            setup_bundle = combined_mod.run_manual_setup(
                first_frame,
                pose_estimator,
                tracker,
                disp_scale,
                hand_detector=hand_detector,
                object_detector=object_detector,
            )
            if setup_bundle is None:
                raise RuntimeError("Monitoring setup was cancelled.")

        runtime.mark_session_running()

        def sound_telemetry_callback(payload: dict) -> None:
            runtime.update_sound_telemetry(payload)

        def sound_incident_callback(payload: dict) -> None:
            with latest_raw_frame_lock:
                noise_frame = latest_raw_frame.get("frame")
            incident, assets = build_noise_incident_evidence(
                node_config=node_config,
                session=session,
                source_label=source_display_label,
                estimated_db=float(payload["estimated_db"]),
                threshold_db=float(payload["threshold_db"]),
                frame=noise_frame,
            )
            if not assets:
                head_mod.log_info(
                    "Noise threshold incident has no available camera frame for poster evidence."
                )
            runtime.record_finalized_incident(incident, assets)

        sound_service = SoundMonitorService(
            runtime_cfg.sound_sensor,
            on_telemetry=sound_telemetry_callback,
            on_threshold_cross=sound_incident_callback,
            log_fn=head_mod.log_info,
        )
        sound_service.start()

        def frame_publish_callback(
            raw_frame,
            annotated_frame,
            metrics: dict,
            *,
            debug_frame=None,
        ) -> None:
            with latest_raw_frame_lock:
                latest_raw_frame["frame"] = raw_frame
            runtime.publish_detector_frames(
                raw_frame,
                annotated_frame,
                processing_fps=metrics.get("processing_fps"),
                debug_frame=debug_frame,
            )

        def incident_finalize_callback(front_manifest: dict) -> None:
            incident, assets = normalize_front_runtime_incident(
                node_config=node_config,
                session=session,
                evidence_root=runtime_cfg.evidence_root,
                front_manifest=front_manifest,
            )
            runtime.record_finalized_incident(incident, assets)

        def incident_detected_callback(front_manifest: dict) -> None:
            incident = normalize_front_runtime_detected_incident(
                node_config=node_config,
                session=session,
                front_manifest=front_manifest,
            )
            runtime.record_detected_incident(incident)

        run_all_behavior_detection(
            combined_mod,
            cap,
            pose_estimator,
            hand_detector,
            object_detector,
            tracker,
            setup_bundle["student_map"],
            setup_bundle["baseline_yaw_map"],
            setup_bundle["assigned_students"],
            setup_bundle["student_lines"],
            source_display_label,
            runtime_cfg.port,
            roi_polygon=setup_bundle["roi_polygon"],
            source_mode=node_config.source_mode,
            source_fps=actual_fps,
            frame_publish_callback=frame_publish_callback,
            incident_finalize_callback=incident_finalize_callback,
            incident_detected_callback=incident_detected_callback,
            should_stop_callback=runtime.should_stop_requested,
            publish_local_preview=CENTRAL_AGENT_PUBLISHES_LOCAL_PREVIEW,
        )
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


def _apply_node_capture_overrides(node_config, runtime_cfg):
    if node_config.source_mode == "webcam":
        webcam_source = replace(
            runtime_cfg.webcam_source,
            camera_index=node_config.camera_index,
        )
        return replace(runtime_cfg, webcam_source=webcam_source)

    video_source = runtime_cfg.video_source
    if node_config.video_path is not None:
        video_source = replace(video_source, default_video=node_config.video_path)
    return replace(
        runtime_cfg,
        default_video=video_source.default_video,
        default_setup_profile=video_source.default_setup_profile,
        auto_use_saved_setup=video_source.auto_use_saved_setup,
        video_source=video_source,
    )


def load_front_runtime_context(node_config):
    if node_config.runtime_config_path is None:
        raise RuntimeError(
            "detector.runtime_config_path is required when detector.mode=front_runtime."
        )

    runtime_cfg = front_runtime_config.load_runtime_config(
        str(node_config.runtime_config_path)
    )
    runtime_cfg = _apply_node_capture_overrides(node_config, runtime_cfg)
    modules = front_runtime_support.load_runtime_modules()
    front_runtime_support.configure_runtime_paths(modules, runtime_cfg)
    front_runtime_support.apply_behavior_config(modules, runtime_cfg)
    return runtime_cfg, modules
