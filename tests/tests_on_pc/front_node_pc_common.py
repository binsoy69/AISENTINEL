#!/usr/bin/env python3
"""PC adapters for running the updated Pi front-node test logic without Hailo."""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PI_TEST_DIR = SCRIPT_DIR.parent / "tests_on_pi"

POSE_MODEL_CANDIDATES = (
    REPO_ROOT / "models" / "archive" / "yolo26s-pose.pt",
    REPO_ROOT / "models" / "archive" / "yolo11n-pose.pt",
    REPO_ROOT / "tests" / "yolo26s-pose.pt",
    REPO_ROOT / "yolo26s-pose.pt",
)

SENTINEL_MODEL_CANDIDATES = (
    REPO_ROOT / "models" / "yolov11n-sentinel-new" / "sentinel_new.pt",
    REPO_ROOT / "models" / "yolov11n-sentinel-new" / "sentinel-new.pt",
    REPO_ROOT / "models" / "yolov11n-sentinel-new" / "train" / "weights" / "best.pt",
    REPO_ROOT / "models" / "archive" / "yolov11n-sentinel-new" / "sentinel_new.pt",
    REPO_ROOT / "models" / "archive" / "yolov11n-sentinel-new" / "train" / "weights" / "best.pt",
    REPO_ROOT / "models" / "archive" / "sentinel-yolov11n" / "my_model2.pt",
    REPO_ROOT / "models" / "archive" / "sentinel-yolov11n" / "train2" / "weights" / "best.pt",
)

FRONT_NODE_OBJECT_MODEL_CANDIDATES = (
    REPO_ROOT / "models" / "front_node" / "my_model.pt",
    REPO_ROOT / "models" / "front_node" / "train" / "weights" / "best.pt",
    REPO_ROOT / "models" / "archive" / "front_node" / "my_model.pt",
    REPO_ROOT / "models" / "archive" / "front_node" / "train" / "weights" / "best.pt",
)

# Final/combined front-node PC flow uses three model roles:
#   pose   -> person/keypoint tracking
#   hand   -> hand-only detection for hands-under-table
#   object -> phone/cheat_sheet detection
HAND_MODEL_CANDIDATES = FRONT_NODE_OBJECT_MODEL_CANDIDATES + SENTINEL_MODEL_CANDIDATES
OBJECT_MODEL_CANDIDATES = SENTINEL_MODEL_CANDIDATES
CV_WINDOW_PORT_HINT = " press Q/Esc to stop"


def is_readable_checkpoint(path):
    """Reject truncated Ultralytics checkpoints before YOLO emits a low-level zip error."""
    if path.suffix.lower() != ".pt":
        return True
    return zipfile.is_zipfile(path)


def first_existing(candidates):
    """Return the first existing path from a candidate list, or None."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and is_readable_checkpoint(path):
            return path
    return None


def resolve_model_path(value, candidates=(), fallback_name=None):
    """Resolve a model path, allowing Ultralytics built-in model names as fallback."""
    if value:
        return str(value)

    found = first_existing(candidates)
    if found is not None:
        return str(found)

    if fallback_name:
        return fallback_name

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No model file found. Searched:\n  {searched}")


class CvWindowFrameLock:
    """Display the Pi module's latest annotated frame from the detection loop."""

    def __init__(self, module, base_lock, window_name, max_width=1280, max_height=720):
        self.module = module
        self.base_lock = base_lock
        self.window_name = window_name
        self.max_width = max_width
        self.max_height = max_height

    def __enter__(self):
        return self.base_lock.__enter__()

    def __exit__(self, exc_type, exc, tb):
        release_result = self.base_lock.__exit__(exc_type, exc, tb)
        if exc_type is None:
            self._show_latest_frame()
        return release_result

    def _show_latest_frame(self):
        frame = getattr(self.module, "_latest_frame", None)
        if frame is None:
            return

        cv2.imshow(self.window_name, resize_for_cv_window(frame, self.max_width, self.max_height))
        key = cv2.waitKey(1) & 0xFF
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error as exc:
            raise KeyboardInterrupt from exc
        if key in (27, ord("q"), ord("Q")) or visible < 1:
            raise KeyboardInterrupt


def resize_for_cv_window(frame, max_width=1280, max_height=720):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    scale = min(1.0, max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def enable_cv_window_stream(module, window_name, max_width=1280, max_height=720):
    """Route the reused Pi detection loop to an OpenCV display window on PC."""
    base_lock = getattr(module, "_frame_lock", None)
    if isinstance(base_lock, CvWindowFrameLock):
        base_lock.window_name = window_name
        return window_name
    if base_lock is None:
        raise AttributeError(f"{module.__name__} does not expose _frame_lock")

    module._latest_frame = None
    module._frame_lock = CvWindowFrameLock(
        module,
        base_lock,
        window_name,
        max_width=max_width,
        max_height=max_height,
    )
    if hasattr(module, "get_local_ip"):
        module.get_local_ip = lambda: window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    return window_name


def close_cv_window(window_name):
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def load_pi_module(module_name: str):
    """Import a tests_on_pi module while hiding the expected Hailo warning on PC."""
    if str(PI_TEST_DIR) not in sys.path:
        sys.path.insert(0, str(PI_TEST_DIR))

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        module = importlib.import_module(module_name)

    for line in stdout.getvalue().splitlines():
        if "hailo_platform not found" in line:
            continue
        print(line)

    patch_module_for_pc(module)
    return module


def patch_module_for_pc(module):
    """Make reused Pi UI/log text identify this as a PC + Ultralytics run."""
    replacements = {
        "  Web stream          : http://": "  OpenCV window       : ",
        "  Web stream     : http://": "  OpenCV window  : ",
        "  Web stream   : http://": "  OpenCV window: ",
        "  Stream   : http://": "  OpenCV window: ",
        "Web stream starts at http://<pi-ip>:8080 with live annotations": (
            "OpenCV window shows live annotations"
        ),
        "streaming annotated frames via Flask.": "showing annotated frames in an OpenCV window.",
        "streaming annotated frames via Flask": "showing annotated frames in an OpenCV window",
        "Web stream": "OpenCV window",
        "Raspberry Pi 5 + Hailo AI HAT": "PC + Ultralytics",
        "Pi + Hailo": "PC + Ultralytics",
        "(Pi + Hailo)": "(PC + Ultralytics)",
        "Pose inference on Hailo": "Pose inference with Ultralytics",
        "Pose model on Hailo NPU": "Pose model with Ultralytics",
        "Shared sentinel model on Hailo NPU": "Shared sentinel model with Ultralytics",
        "Hailo NPU": "Ultralytics",
    }

    if hasattr(module, "HTML_PAGE"):
        module.HTML_PAGE = _replace_strings(module.HTML_PAGE, replacements)

    for value in list(module.__dict__.values()):
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            value.__code__ = _replace_code_strings(value.__code__, replacements)


def _replace_strings(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _replace_code_strings(code, replacements: dict[str, str]):
    new_consts = []
    changed = False

    for const in code.co_consts:
        if isinstance(const, str):
            new_const = _replace_strings(const, replacements)
            changed = changed or new_const != const
            new_consts.append(new_const)
        elif isinstance(const, types.CodeType):
            new_const = _replace_code_strings(const, replacements)
            changed = changed or new_const is not const
            new_consts.append(new_const)
        else:
            new_consts.append(const)

    if not changed:
        return code
    return code.replace(co_consts=tuple(new_consts))


def canonical_label(label: str) -> str:
    """Normalize common model label variants to the Pi test label vocabulary."""
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cellphone": "phone",
        "cell_phone": "phone",
        "mobile": "phone",
        "mobile_phone": "phone",
        "cheatsheet": "cheat_sheet",
        "cheat_sheet": "cheat_sheet",
        "paper_note": "cheat_sheet",
    }
    return aliases.get(key, key)


class UltralyticsPoseEstimator:
    """Ultralytics pose model adapter with the same surface as the Pi estimator."""

    def __init__(
        self,
        model_path,
        conf_threshold=0.5,
        iou_threshold=0.45,
        imgsz=640,
        device=None,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(self.model_path)

    def detect_pose(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._result_to_pose_detections(results[0], frame.shape)

    def detect_persons(self, frame):
        return self.detect_pose(frame)

    def close(self):
        return None

    @staticmethod
    def _result_to_pose_detections(result, frame_shape):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        keypoints = np.zeros((len(xyxy), 17, 3), dtype=np.float32)
        result_keypoints = getattr(result, "keypoints", None)
        if result_keypoints is not None and getattr(result_keypoints, "data", None) is not None:
            kp_data = result_keypoints.data.detach().cpu().numpy()
            if kp_data.ndim == 3 and kp_data.shape[0] == len(xyxy):
                count = min(kp_data.shape[1], 17)
                keypoints[:, :count, : min(kp_data.shape[2], 3)] = kp_data[
                    :, :count, : min(kp_data.shape[2], 3)
                ]
                if kp_data.shape[2] == 2:
                    keypoints[:, :count, 2] = 1.0

        h, w = frame_shape[:2]
        detections = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = _clip_box(box, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confs[idx]),
                    "keypoints": keypoints[idx].astype(np.float32),
                }
            )
        return detections


class UltralyticsObjectDetector:
    """Ultralytics object detector adapter matching the Pi Hailo detector API."""

    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        imgsz=640,
        device=None,
        class_aliases=None,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_aliases = class_aliases or {}
        self.model = YOLO(self.model_path)

    @property
    def names(self):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return names
        return {idx: name for idx, name in enumerate(names)}

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._result_to_object_detections(results[0], frame.shape)

    def close(self):
        return None

    def _result_to_object_detections(self, result, frame_shape):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        names = self.names
        h, w = frame_shape[:2]

        detections = []
        for box, conf, class_id in zip(xyxy, confs, class_ids):
            x1, y1, x2, y2 = _clip_box(box, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = names.get(int(class_id), f"class_{int(class_id)}")
            class_name = self.class_aliases.get(raw_name, canonical_label(raw_name))
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(class_id),
                    "class_name": class_name,
                }
            )
        return detections


def _clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2
