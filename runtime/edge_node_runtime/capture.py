"""Capture and calibration helpers for edge-node sessions."""

from __future__ import annotations

import cv2

from . import runtime_support


def default_setup_profile(runtime_cfg, node_config, setup_io):
    if node_config.source_mode == "webcam":
        return runtime_cfg.webcam_source.default_setup_profile
    video_path = runtime_cfg.video_source.default_video
    if video_path is not None and runtime_cfg.video_source.auto_use_saved_setup:
        auto_calibration = setup_io.default_setup_profile_path(video_path)
        if auto_calibration.exists():
            return auto_calibration
    return runtime_cfg.video_source.default_setup_profile


def open_capture_bundle(node_config, runtime_cfg, modules) -> dict:
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

    cap = runtime_support.open_webcam_capture(runtime_cfg, head_mod)
    first_frame = runtime_support.read_webcam_frame(cap)
    if first_frame is None:
        raise RuntimeError("Cannot read webcam frame.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 120:
        fps = runtime_cfg.webcam_source.capture_fps
    source_label = runtime_support.get_webcam_source_label(runtime_cfg)
    head_mod.log_info(
        f"Webcam resolution: {first_frame.shape[1]}x{first_frame.shape[0]} @ {fps:.1f} FPS"
    )
    return {
        "capture": cap,
        "first_frame": first_frame,
        "fps": fps,
        "source_label": source_label,
    }


def resolve_calibration_path(node_config, runtime_cfg, setup_io, source_label: str, head_mod):
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
