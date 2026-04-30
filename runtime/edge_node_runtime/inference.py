"""Hailo inference construction helpers."""

from __future__ import annotations


def require_hailo_runtime(modules) -> None:
    if (
        not modules.head_mod.HAILO_AVAILABLE
        or not modules.hands_mod.HAILO_AVAILABLE
        or not modules.obj_mod.HAILO_AVAILABLE
    ):
        raise RuntimeError(
            "hailo_platform is required for detector.mode=front_runtime. "
            "Install: sudo apt install hailo-all"
        )


def create_hailo_detectors(runtime_cfg, modules):
    combined_mod = modules.combined_mod
    head_mod = modules.head_mod
    hands_mod = modules.hands_mod
    obj_mod = modules.obj_mod

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
    return shared_vdevice, pose_estimator, hand_detector, object_detector

