"""Adapter that runs the real front-node Hailo runtime inside the node agent."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
import sys
import threading

import cv2

from central_dashboard.shared.dto import IncidentManifest, SessionSpec, make_id, utc_now_iso


FRONT_NODE_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "front_node_pi"
if str(FRONT_NODE_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(FRONT_NODE_RUNTIME_DIR))

import runtime_config as front_runtime_config  # type: ignore
import runtime_support as front_runtime_support  # type: ignore
from sound_monitor import SoundMonitorService, build_noise_summary


def run_front_runtime_session(runtime, session: SessionSpec) -> None:
    """Run the real front-node detector loop for one central-dashboard session."""

    node_config = runtime.config
    runtime_cfg, modules = load_front_runtime_context(node_config)
    _require_hailo_runtime(modules)
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
    default_setup_profile = _default_setup_profile(runtime_cfg, node_config, setup_io)
    combined_mod.configure_web_dashboard(
        auth_config=runtime_cfg.web_dashboard,
        runtime_mode=node_config.source_mode,
        source_label=source_label,
        config_path=runtime_cfg.config_path,
        evidence_root=runtime_cfg.evidence_root,
        setup_profile_path=default_setup_profile,
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
            "setup_profile_override": str(default_setup_profile or ""),
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
        capture_bundle = _open_capture_bundle(
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

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first_frame.shape[1])
        disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0

        shared_vdevice = hands_mod.VDevice()
        head_mod.log_info("Hailo VDevice created (shared across all models).")

        pose_estimator = combined_mod.SharedHailoPoseEstimator(
            str(runtime_cfg.pose_model),
            conf_threshold=runtime_cfg.pose_confidence,
            vdevice=shared_vdevice,
        )
        hand_detector = hands_mod.HailoObjectDetector(
            str(runtime_cfg.hand_model),
            class_names=hands_mod.HAND_MODEL_CLASS_NAMES,
            conf_threshold=hands_mod.HAND_CONFIDENCE,
            vdevice=shared_vdevice,
        )
        object_detector = obj_mod.HailoObjectDetector(
            str(runtime_cfg.object_model),
            conf_threshold=runtime_cfg.object_confidence,
            vdevice=shared_vdevice,
        )

        tracker = combined_mod.ReacquiringLockedIoUTracker(
            iou_threshold=runtime_cfg.tracking.iou_threshold,
            max_lost=runtime_cfg.tracking.max_lost,
        )
        calibration_path = _resolve_calibration_path(
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
                tracker = combined_mod.ReacquiringLockedIoUTracker(
                    iou_threshold=runtime_cfg.tracking.iou_threshold,
                    max_lost=runtime_cfg.tracking.max_lost,
                )

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
            incident, assets = _build_noise_incident_evidence(
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

        def frame_publish_callback(raw_frame, annotated_frame, metrics: dict) -> None:
            with latest_raw_frame_lock:
                latest_raw_frame["frame"] = raw_frame
            runtime.publish_detector_frames(
                raw_frame,
                annotated_frame,
                processing_fps=metrics.get("processing_fps"),
            )

        def incident_finalize_callback(front_manifest: dict) -> None:
            incident, assets = _normalize_front_runtime_incident(
                node_config=node_config,
                session=session,
                evidence_root=runtime_cfg.evidence_root,
                front_manifest=front_manifest,
            )
            runtime.record_finalized_incident(incident, assets)

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
            source_display_label,
            runtime_cfg.port,
            roi_polygon=setup_bundle["roi_polygon"],
            source_mode=node_config.source_mode,
            source_fps=actual_fps,
            frame_publish_callback=frame_publish_callback,
            incident_finalize_callback=incident_finalize_callback,
            should_stop_callback=runtime.should_stop_requested,
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


def _require_hailo_runtime(modules) -> None:
    if (
        not modules.head_mod.HAILO_AVAILABLE
        or not modules.hands_mod.HAILO_AVAILABLE
        or not modules.obj_mod.HAILO_AVAILABLE
    ):
        raise RuntimeError(
            "hailo_platform is required for detector.mode=front_runtime. "
            "Install: sudo apt install hailo-all"
        )


def _default_setup_profile(runtime_cfg, node_config, setup_io):
    if node_config.source_mode == "webcam":
        return runtime_cfg.webcam_source.default_setup_profile
    video_path = runtime_cfg.video_source.default_video
    if video_path is not None and runtime_cfg.video_source.auto_use_saved_setup:
        auto_calibration = setup_io.default_setup_profile_path(video_path)
        if auto_calibration.exists():
            return auto_calibration
    return runtime_cfg.video_source.default_setup_profile


def _open_capture_bundle(node_config, runtime_cfg, modules) -> dict:
    head_mod = modules.head_mod

    if node_config.source_mode == "video":
        video_path = runtime_cfg.video_source.default_video
        if video_path is None:
            raise RuntimeError(
                "Video mode requires capture.video_path in the node INI or "
                "video_source.default_video in the detector runtime INI."
            )
        if not video_path.exists():
            raise RuntimeError(f"Configured video source not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            raise RuntimeError(f"Cannot read the first frame from: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        head_mod.log_info(
            f"Video resolution: {first_frame.shape[1]}x{first_frame.shape[0]} @ {fps:.1f} FPS"
        )
        return {
            "capture": cap,
            "first_frame": first_frame,
            "fps": fps,
            "source_label": str(video_path),
        }

    cap = front_runtime_support.open_webcam_capture(runtime_cfg, head_mod)
    first_frame = front_runtime_support.read_webcam_frame(cap)
    if first_frame is None:
        raise RuntimeError("Cannot read webcam frame.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 120:
        fps = runtime_cfg.webcam_source.capture_fps
    source_label = front_runtime_support.get_webcam_source_label(runtime_cfg)
    head_mod.log_info(
        f"Webcam resolution: {first_frame.shape[1]}x{first_frame.shape[0]} @ {fps:.1f} FPS"
    )
    return {
        "capture": cap,
        "first_frame": first_frame,
        "fps": fps,
        "source_label": source_label,
    }


def _resolve_calibration_path(node_config, runtime_cfg, setup_io, source_label: str, head_mod):
    calibration_path = None

    if node_config.source_mode == "video":
        if runtime_cfg.video_source.default_setup_profile is not None:
            if runtime_cfg.video_source.default_setup_profile.exists():
                calibration_path = runtime_cfg.video_source.default_setup_profile
            else:
                head_mod.log_info(
                    "Configured video setup profile not found. Falling back to auto/manual setup."
                )
        if calibration_path is None and runtime_cfg.video_source.auto_use_saved_setup:
            video_path = runtime_cfg.video_source.default_video
            if video_path is not None:
                auto_calibration = setup_io.default_setup_profile_path(video_path)
                if auto_calibration.exists():
                    calibration_path = auto_calibration
        return calibration_path

    if runtime_cfg.webcam_source.default_setup_profile is not None:
        if runtime_cfg.webcam_source.default_setup_profile.exists():
            calibration_path = runtime_cfg.webcam_source.default_setup_profile
        else:
            head_mod.log_info(
                "Configured webcam setup profile not found. Falling back to auto/manual setup."
            )
    if calibration_path is None and runtime_cfg.webcam_source.auto_use_saved_setup:
        auto_calibration = setup_io.default_setup_profile_path(source_label)
        if auto_calibration.exists():
            calibration_path = auto_calibration
    return calibration_path


def _normalize_front_runtime_incident(
    *,
    node_config,
    session: SessionSpec,
    evidence_root: Path,
    front_manifest: dict,
) -> tuple[IncidentManifest, list[dict]]:
    incident_root = PurePosixPath(
        str(front_manifest.get("manifest_relpath") or "manifest.json")
    ).parent
    poster_relpath = str(front_manifest.get("poster_relpath") or "").strip()
    gif_relpath = str(front_manifest.get("gif_relpath") or "").strip()
    frame_relpaths = [str(value) for value in (front_manifest.get("frame_relpaths") or [])]

    assets = []
    asset_names = []

    if poster_relpath:
        asset_names.append(_incident_asset_name(incident_root, poster_relpath))
        assets.append(
            {
                "asset_type": "poster",
                "file_path": _local_evidence_path(evidence_root, poster_relpath),
                "filename": asset_names[-1],
            }
        )

    if gif_relpath:
        asset_names.append(_incident_asset_name(incident_root, gif_relpath))
        assets.append(
            {
                "asset_type": "gif",
                "file_path": _local_evidence_path(evidence_root, gif_relpath),
                "filename": asset_names[-1],
            }
        )

    for relpath in frame_relpaths:
        if relpath == poster_relpath:
            continue
        asset_names.append(_incident_asset_name(incident_root, relpath))
        assets.append(
            {
                "asset_type": "frame",
                "file_path": _local_evidence_path(evidence_root, relpath),
                "filename": asset_names[-1],
            }
        )

    manifest = IncidentManifest(
        incident_id=str(front_manifest.get("id", "")).strip(),
        session_id=session.session_id,
        node_id=node_config.node_id,
        camera_label=node_config.camera_label,
        behavior_type=str(front_manifest.get("behavior_type", "")).strip() or "object",
        type_label=str(front_manifest.get("type_label", "")).strip() or "Incident",
        student_numbers=[
            int(value) for value in (front_manifest.get("student_numbers") or [])
        ],
        created_at=str(front_manifest.get("created_at") or utc_now_iso()),
        display_time=str(front_manifest.get("display_time", "")).strip(),
        review_status=str(front_manifest.get("review_status", "unverified")).strip()
        or "unverified",
        poster_path="",
        gif_path="",
        frame_count=int(front_manifest.get("frame_count") or len(frame_relpaths)),
        summary=str(front_manifest.get("summary", "")).strip(),
        sync_status="queued",
        sync_attempts=0,
        asset_names=asset_names,
    )
    return manifest, assets


def _build_noise_incident_evidence(
    *,
    node_config,
    session: SessionSpec,
    source_label: str,
    estimated_db: float,
    threshold_db: float,
    frame,
) -> tuple[IncidentManifest, list[dict]]:
    incident_id = make_id("noise")
    assets = []
    asset_names = []
    frame_count = 0

    if frame is not None:
        incident_dir = node_config.evidence_root / session.session_id / incident_id
        incident_dir.mkdir(parents=True, exist_ok=True)
        poster_path = incident_dir / "poster.jpg"
        if cv2.imwrite(str(poster_path), frame):
            asset_names.append("poster.jpg")
            assets.append(
                {
                    "asset_type": "poster",
                    "file_path": poster_path,
                    "filename": "poster.jpg",
                }
            )
            frame_count = 1

    manifest = IncidentManifest(
        incident_id=incident_id,
        session_id=session.session_id,
        node_id=node_config.node_id,
        camera_label=node_config.camera_label or source_label,
        behavior_type="noise",
        type_label="Noise Threshold Exceeded",
        student_numbers=[],
        created_at=utc_now_iso(),
        display_time=datetime.now().strftime("%I:%M %p").lstrip("0"),
        review_status="unverified",
        poster_path="",
        gif_path="",
        frame_count=frame_count,
        summary=build_noise_summary(estimated_db, threshold_db),
        sync_status="queued",
        sync_attempts=0,
        asset_names=asset_names,
    )
    return manifest, assets


def _incident_asset_name(incident_root: PurePosixPath, relpath: str) -> str:
    asset_path = PurePosixPath(relpath)
    try:
        return asset_path.relative_to(incident_root).as_posix()
    except ValueError:
        return asset_path.name


def _local_evidence_path(evidence_root: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath)
    return evidence_root.joinpath(*rel.parts)
