#!/usr/bin/env python3
"""
Cellphone / Cheat Sheet Detection - Raspberry Pi + Hailo AI HAT
================================================================
Pi counterpart of front_node_cellphone_cheat_pc.py.

Runs the cheat-sheet_phone_model.hef object detection model on the Hailo-8 NPU,
detecting phone and cheat_sheet objects associated with tracked students.
The model also contains a hand class, but this script ignores it.

Workflow:
  1. File dialog opens (tkinter) to select a video file
  2. First frame shown via OpenCV - click student bboxes to assign numbers
  3. Web stream starts at http://<pi-ip>:8080 with live annotations
  4. Console alerts + evidence screenshots saved to ./evidence_obj/

Inference runs on the Hailo-8 NPU using HailoRT Python API with
cheat-sheet_phone_model.hef (no Ultralytics / no GStreamer dependency at runtime).

A simple IoU tracker maintains student identity across frames so that
the student assignments from the first frame persist throughout the video.

Detected objects (phone, cheat_sheet) are associated to the
nearest assigned student by bbox overlap.

Controls (Assignment phase - local OpenCV window):
    Left-click on person  -> select person (highlighted in cyan)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start detection (need >= 1 assignment)
    ESC                   -> quit

Requirements:
    pip install opencv-python numpy flask
    System: hailo-all (provides hailo_platform)
"""

import sys
import os
import time
import threading
import socket
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

from front_node_pi_model_paths import OBJECT_MODEL_PATH, POSE_MODEL_PATH
import front_node_pi_interactive as pi_ui

OBJ_MODEL_PATH = OBJECT_MODEL_PATH
EVIDENCE_DIR = SCRIPT_DIR / "evidence_obj"

# ── Class mapping (from exported sentinel_new.onnx metadata) ─
# Embedded names found in the sibling ONNX export:
#   {0: 'cheat_sheet', 1: 'hand', 2: 'phone'}
OBJECT_UPDATED_CLASS_NAMES = {
    0: "cheat_sheet",
    1: "phone",
}

LEGACY_OBJECT_CLASS_NAMES = {
    0: "cheat_sheet",
    1: "hand",
    2: "phone",
}

CLASS_NAMES = OBJECT_UPDATED_CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)


def object_class_names_for_model(model_path):
    """Return the class map matching the selected phone/cheat-sheet HEF."""
    if Path(model_path).name == "object-updated.hef":
        return dict(OBJECT_UPDATED_CLASS_NAMES)
    return dict(LEGACY_OBJECT_CLASS_NAMES)

# ── Object classes to monitor from the model output ──────────
OBJECT_CLASSES = {"phone", "cheat_sheet"}

# ── Alert classes (trigger alerts + evidence) ────────────────
ALERT_CLASSES = {"phone", "cheat_sheet"}

PERSON_CONFIDENCE = 0.5

CONFIDENCE_THRESHOLDS = {
    "phone": 0.25,
    "cheat_sheet": 0.3,
}

# ── Alert cooldown per (student, class) pair ─────────────────
EVENT_COOLDOWN_SEC = 10.0

# ── Object-to-student association ────────────────────────────
# An object is associated to a student if the IoU between the object
# bbox and the student bbox is above this threshold, OR the object
# center is inside the student bbox.
ASSOC_IOU_THRESH = 0.05

# ── Colors (BGR) ─────────────────────────────────────────────
COL_PHONE = (0, 0, 255)           # red
COL_CHEAT_SHEET = (0, 165, 255)   # orange
COL_UNASSIGNED = (128, 128, 128)  # gray
COL_SELECTED = (255, 255, 0)      # cyan
COL_FLAGGED = (0, 0, 255)         # red
COL_HUD = (0, 255, 0)             # green

CLASS_COLORS = {
    "phone": COL_PHONE,
    "cheat_sheet": COL_CHEAT_SHEET,
}

# ── Flask globals ────────────────────────────────────────────
_latest_frame = None
_frame_lock = threading.Lock()

# ── Try imports ──────────────────────────────────────────────
try:
    from flask import Flask, Response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[WARN] flask not found. Install with: pip install flask")

try:
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[WARN] hailo_platform not found. Install: sudo apt install hailo-all")


# ── Terminal helpers ─────────────────────────────────────────
class TC:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - total) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def log_alert(label: str, student_num: int, conf: float, ts_sec: float):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.RED}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{TC.RED}{label.upper()} detected near Student #{student_num} "
        f"(conf={conf:.0%}){TC.RESET}"
    )


def log_info(msg: str):
    print(f"{TC.CYAN}[INFO]{TC.RESET} {msg}")


# ── Drawing helpers ──────────────────────────────────────────
def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)


def save_evidence(frame, student_num, label, conf, ts_sec):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"student{student_num}_{label}_{conf:.0f}pct_{ts_str}.jpg"
    path = EVIDENCE_DIR / fname
    cv2.imwrite(str(path), frame)
    log_info(f"Evidence saved: {fname}")


# ═══════════════════════════════════════════════════════════════
#  HAILO OBJECT DETECTOR
# ═══════════════════════════════════════════════════════════════

def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _decode_dfl(box_logits, reg_max=16):
    box_logits = box_logits.reshape(-1, 4, reg_max)
    box_logits = box_logits - np.max(box_logits, axis=2, keepdims=True)
    probs = np.exp(box_logits)
    probs /= np.sum(probs, axis=2, keepdims=True) + 1e-9
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max)
    return np.sum(probs * bins, axis=2)


def _nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _xywh_to_xyxy(boxes):
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


class HailoPoseEstimator:
    """HailoRT wrapper for YOLOv8-pose. Used here only for person bbox detection."""

    def __init__(self, hef_path, conf_threshold=0.5, iou_threshold=0.45, vdevice=None):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._infer_ctx = None
        self._infer_pipeline = None

        log_info(f"Loading pose HEF model: {hef_path}")
        self.hef = HEF(str(hef_path))

        self.vdevice = vdevice or VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self.input_shape = self.input_vstream_info[0].shape
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        log_info(f"Pose model input: {self.input_shape}")
        log_info("Pose model ready (person detection).")

    def _ensure_infer_pipeline(self):
        """Create the Hailo infer pipeline once and reuse it."""
        if self._infer_pipeline is not None:
            return

        infer_ctx = None
        try:
            infer_ctx = InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            )
            infer_pipeline = infer_ctx.__enter__()
        except Exception as exc:
            if infer_ctx is not None:
                infer_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

        self._infer_ctx = infer_ctx
        self._infer_pipeline = infer_pipeline

    def close(self):
        """Release persistent Hailo resources."""
        if self._infer_ctx is not None:
            self._infer_ctx.__exit__(None, None, None)
            self._infer_ctx = None
            self._infer_pipeline = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def detect_persons(self, frame):
        """Run pose estimation, return person bboxes only.

        Returns list of dicts: [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        img_h, img_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        input_dict = {self.input_vstream_info[0].name: input_data}
        self._ensure_infer_pipeline()
        with self.network_group.activate(self.network_group_params):
            results = self._infer_pipeline.infer(input_dict)

        return self._postprocess(results, img_w, img_h)

    def _decode_multiscale_heads(self, raw_output):
        if not isinstance(raw_output, dict):
            return None

        groups = {}
        for _, arr in raw_output.items():
            a = np.array(arr, dtype=np.float32, copy=False)
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 2:
                a = a[:, :, None]
            if a.ndim != 3:
                continue
            if a.shape[-1] in (64, 51, 1):
                h, w, c = a.shape
                hwc = a
            elif a.shape[0] in (64, 51, 1):
                c, h, w = a.shape
                hwc = np.transpose(a, (1, 2, 0))
            else:
                continue
            group = groups.setdefault((h, w), {})
            group[c] = hwc

        decoded_scales = []
        for (h, w), group in sorted(groups.items(), key=lambda x: x[0][0],
                                     reverse=True):
            if 64 not in group or 1 not in group or 51 not in group:
                continue
            box_logits = group[64].reshape(-1, 64)
            obj_logits = group[1].reshape(-1)
            kpt_logits = group[51].reshape(-1, 51)
            stride_x = self.input_w / float(w)
            stride_y = self.input_h / float(h)
            gy, gx = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32), indexing="ij")
            anchor_x = gx.reshape(-1) + 0.5
            anchor_y = gy.reshape(-1) + 0.5
            ltrb = _decode_dfl(box_logits, reg_max=16)
            x1 = (anchor_x - ltrb[:, 0]) * stride_x
            y1 = (anchor_y - ltrb[:, 1]) * stride_y
            x2 = (anchor_x + ltrb[:, 2]) * stride_x
            y2 = (anchor_y + ltrb[:, 3]) * stride_y
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bw = x2 - x1
            bh = y2 - y1
            boxes_xywh = np.stack([cx, cy, bw, bh], axis=1)
            confidences = _sigmoid(obj_logits).reshape(-1, 1)
            keypoints_flat = kpt_logits.reshape(-1, 51)
            decoded_scales.append(
                np.concatenate([boxes_xywh, confidences, keypoints_flat], axis=1))
        if not decoded_scales:
            return None
        return np.concatenate(decoded_scales, axis=0)

    def _postprocess(self, raw_output, img_w, img_h):
        if isinstance(raw_output, dict):
            decoded = self._decode_multiscale_heads(raw_output)
            if decoded is not None:
                output = decoded
            else:
                arrays = list(raw_output.values())
                if len(arrays) == 1:
                    output = arrays[0]
                else:
                    flat = []
                    for a in arrays:
                        a = np.squeeze(a)
                        if a.ndim == 3:
                            a = a.reshape(-1, a.shape[-1])
                        elif a.ndim == 1:
                            continue
                        flat.append(a)
                    if not flat:
                        return []
                    if all(x.shape[1] == flat[0].shape[1] for x in flat):
                        output = np.concatenate(flat, axis=0)
                    else:
                        output = max(flat, key=lambda x: x.shape[0] * x.shape[1])
        else:
            output = raw_output

        output = np.squeeze(output)
        if output.ndim == 2:
            if output.shape[0] == 56 and output.shape[1] > 56:
                output = output.T
        elif output.ndim == 3:
            output = output.reshape(-1, output.shape[-1])
            if output.shape[0] == 56 and output.shape[1] > 56:
                output = output.T

        num_cols = output.shape[1]
        if num_cols == 55:
            has_conf = False
        elif num_cols == 56:
            has_conf = True
        else:
            if output.shape[0] in (55, 56):
                output = output.T
                num_cols = output.shape[1]
            if num_cols not in (55, 56):
                return []
            has_conf = (num_cols == 56)

        boxes_xywh = output[:, :4]
        if has_conf:
            confidences = output[:, 4]
        else:
            keypoints_raw = output[:, 4:]
            kpt_confs = keypoints_raw[:, 2::3]
            confidences = np.mean(kpt_confs, axis=1)

        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

        scale_x = img_w / self.input_w
        scale_y = img_h / self.input_h
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        keep = _nms(boxes_xyxy, confidences, self.iou_threshold)

        results = []
        for idx in keep:
            results.append({
                'bbox': boxes_xyxy[idx].astype(int).tolist(),
                'confidence': float(confidences[idx]),
            })
        return results


class HailoObjectDetector:
    """HailoRT Python API wrapper for YOLOv11 object detection on the Hailo NPU."""

    def __init__(self, hef_path, conf_threshold=0.25, iou_threshold=0.45, vdevice=None):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = object_class_names_for_model(hef_path)
        self.num_classes = len(self.class_names)
        self._infer_ctx = None
        self._infer_pipeline = None

        log_info(f"Loading HEF model: {hef_path}")
        self.hef = HEF(str(hef_path))

        self.vdevice = vdevice or VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self.input_shape = self.input_vstream_info[0].shape
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        log_info(f"Model input shape : {self.input_shape}")
        log_info(f"Model classes     : {self.num_classes} -> {self.class_names}")
        for out_info in self.output_vstream_info:
            log_info(f"Model output layer: {out_info.name} -> {out_info.shape}")
        log_info("Hailo device ready.")

    def _ensure_infer_pipeline(self):
        """Create the Hailo infer pipeline once and reuse it."""
        if self._infer_pipeline is not None:
            return

        infer_ctx = None
        try:
            infer_ctx = InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            )
            infer_pipeline = infer_ctx.__enter__()
        except Exception as exc:
            if infer_ctx is not None:
                infer_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

        self._infer_ctx = infer_ctx
        self._infer_pipeline = infer_pipeline

    def close(self):
        """Release persistent Hailo resources."""
        if self._infer_ctx is not None:
            self._infer_ctx.__exit__(None, None, None)
            self._infer_ctx = None
            self._infer_pipeline = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def detect(self, frame):
        """Run object detection on a BGR frame.

        Returns list of dicts:
            [{'bbox': [x1,y1,x2,y2], 'confidence': float,
              'class_id': int, 'class_name': str}, ...]
        """
        img_h, img_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        input_dict = {self.input_vstream_info[0].name: input_data}
        self._ensure_infer_pipeline()
        with self.network_group.activate(self.network_group_params):
            results = self._infer_pipeline.infer(input_dict)

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            if isinstance(results, dict):
                for name, arr in results.items():
                    try:
                        a = np.array(arr, dtype=np.float32, copy=False)
                        log_info(f"Output '{name}': shape={a.shape}, "
                                 f"min={a.min():.4f}, max={a.max():.4f}")
                    except ValueError:
                        log_info(f"Output '{name}': inhomogeneous (NMS output)")

        return self._postprocess(results, img_w, img_h)

    def _postprocess(self, raw_output, img_w, img_h):
        """Try NMS output first, then fall back to multiscale head decode."""
        if not isinstance(raw_output, dict):
            return []

        # Strategy 1: HailoRT NMS postprocessed output
        nms_results = self._try_decode_nms(raw_output, img_w, img_h)
        if nms_results is not None:
            return nms_results

        # Strategy 2: Split multiscale heads (64ch box + nc cls per scale)
        multiscale = self._try_decode_multiscale(raw_output)
        if multiscale is not None:
            return self._filter_decoded(multiscale, img_w, img_h)

        # Strategy 3: Concatenated [N, 4+nc]
        concat = self._try_decode_concatenated(raw_output)
        if concat is not None:
            return self._filter_decoded(concat, img_w, img_h)

        return []

    def _try_decode_nms(self, raw_output, img_w, img_h):
        """Decode HailoRT NMS (post-processed) output.

        NMS layer outputs a nested structure:
          result[layer_name][0] = classes_list (length = num_classes)
            classes_list[cls_id] = np.array of shape (N, 5)
              each row = [y1, x1, y2, x2, score] (normalized 0..1)
        """
        results = []

        for name, arr in raw_output.items():
            try:
                batch_0 = arr[0]
                num_classes_found = len(batch_0)

                for cls_id in range(num_classes_found):
                    class_dets = batch_0[cls_id]
                    try:
                        ca = np.array(class_dets, dtype=np.float32)
                    except ValueError:
                        continue

                    if ca.size == 0:
                        continue

                    if ca.ndim == 1 and ca.shape[0] == 5:
                        ca = ca.reshape(1, 5)
                    elif ca.ndim == 3:
                        ca = ca.reshape(-1, ca.shape[-1])
                    elif ca.ndim != 2:
                        continue

                    if ca.shape[1] < 5:
                        continue

                    for row in ca:
                        score = float(row[4])
                        if score > self.conf_threshold:
                            y1n, x1n, y2n, x2n = row[0], row[1], row[2], row[3]
                            results.append({
                                'bbox': [
                                    int(x1n * img_w), int(y1n * img_h),
                                    int(x2n * img_w), int(y2n * img_h),
                                ],
                                'confidence': score,
                                'class_id': int(cls_id),
                                'class_name': self.class_names.get(cls_id, f"cls_{cls_id}"),
                            })
            except (IndexError, TypeError):
                return None

        return results if results else []

    def _try_decode_multiscale(self, raw_output):
        """Decode split YOLO detection heads (64ch box + nc cls per scale)."""
        nc = self.num_classes
        groups = {}
        for _, arr in raw_output.items():
            try:
                a = np.array(arr, dtype=np.float32, copy=False)
            except ValueError:
                continue
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            if a.ndim != 3:
                continue

            h, w, c = a.shape
            if c in (64, nc):
                group = groups.setdefault((h, w), {})
                group.setdefault(c, []).append(a)
                continue

            c2, h2, w2 = a.shape
            if c2 in (64, nc):
                hwc = np.transpose(a, (1, 2, 0))
                group = groups.setdefault((h2, w2), {})
                group.setdefault(c2, []).append(hwc)

        decoded_scales = []
        for (gh, gw), group in sorted(groups.items(), key=lambda x: x[0][0],
                                       reverse=True):
            if 64 not in group or nc not in group:
                continue

            box_hwc = group[64][0]
            cls_hwc = group[nc][0]

            box_logits = box_hwc.reshape(-1, 64)
            cls_logits = cls_hwc.reshape(-1, nc)

            stride_x = self.input_w / float(gw)
            stride_y = self.input_h / float(gh)

            gy, gx = np.meshgrid(
                np.arange(gh, dtype=np.float32),
                np.arange(gw, dtype=np.float32),
                indexing="ij",
            )
            ax = gx.reshape(-1) + 0.5
            ay = gy.reshape(-1) + 0.5

            ltrb = _decode_dfl(box_logits, reg_max=16)
            x1 = (ax - ltrb[:, 0]) * stride_x
            y1 = (ay - ltrb[:, 1]) * stride_y
            x2 = (ax + ltrb[:, 2]) * stride_x
            y2 = (ay + ltrb[:, 3]) * stride_y

            boxes = np.stack([x1, y1, x2, y2], axis=1)
            cls_scores = _sigmoid(cls_logits)

            decoded_scales.append(np.concatenate([boxes, cls_scores], axis=1))

        if not decoded_scales:
            return None
        return np.concatenate(decoded_scales, axis=0)

    def _try_decode_concatenated(self, raw_output):
        """Try to decode a single concatenated output [N, 4+nc] or [4+nc, N]."""
        nc = self.num_classes
        expected = 4 + nc

        for arr in raw_output.values():
            try:
                a = np.array(arr, dtype=np.float32, copy=False)
            except ValueError:
                continue
            a = np.squeeze(a)
            if a.ndim != 2:
                continue
            if a.shape[1] == expected:
                return a
            if a.shape[0] == expected:
                return a.T

        return None

    def _filter_decoded(self, output, img_w, img_h):
        """Apply confidence filter + NMS on decoded [N, 4+nc] array."""
        nc = self.num_classes
        boxes = output[:, :4]
        cls_scores = output[:, 4:]

        class_ids = np.argmax(cls_scores, axis=1)
        confidences = np.max(cls_scores, axis=1)

        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        sx = img_w / self.input_w
        sy = img_h / self.input_h
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        results = []
        for cid in range(nc):
            cmask = class_ids == cid
            if not np.any(cmask):
                continue
            cb = boxes[cmask]
            cc = confidences[cmask]
            keep = _nms(cb, cc, self.iou_threshold)
            for idx in keep:
                results.append({
                    'bbox': cb[idx].astype(int).tolist(),
                    'confidence': float(cc[idx]),
                    'class_id': int(cid),
                    'class_name': self.class_names.get(cid, f"cls_{cid}"),
                })

        return results


# ═══════════════════════════════════════════════════════════════
#  SIMPLE IoU TRACKER
# ═══════════════════════════════════════════════════════════════

class IoUTracker:
    """Lightweight frame-to-frame IoU tracker for seated students.

    After keep_only() is called the tracker enters *locked* mode:
      - Only the assigned track IDs are kept.
      - Assigned tracks are NEVER deleted, even after many lost frames.
        Their last-known bbox is preserved so that a student who
        temporarily disappears (occlusion, missed detection) can be
        re-matched as soon as the detector picks them up again.
      - A lower IoU threshold is used for re-acquiring lost tracks
        (the student may have shifted slightly).
      - Unmatched detections that don't match any assigned track
        are returned with track_id = -1.
    """

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self._next_id = 1
        self._tracks = {}       # track_id -> {'bbox': [x1,y1,x2,y2], 'lost': int}
        self.iou_threshold = iou_threshold
        self.reacquire_iou_threshold = 0.15  # lower bar for re-acquiring lost tracks
        self.max_lost = max_lost
        self._locked = False
        self._assigned_tids = set()

    def keep_only(self, track_ids_to_keep):
        """Remove all tracks except those in the given set."""
        to_remove = [tid for tid in self._tracks if tid not in track_ids_to_keep]
        for tid in to_remove:
            del self._tracks[tid]
        self._locked = True
        self._assigned_tids = set(track_ids_to_keep)

    def update(self, detections):
        """Match detections to existing tracks.

        Args:
            detections: list of dicts with 'bbox' key [x1,y1,x2,y2]

        Returns list of track_ids aligned with detections.
        Unmatched detections get track_id = -1 when locked.
        """
        if not detections:
            for t in self._tracks.values():
                t['lost'] += 1
            if not self._locked:
                to_remove = [tid for tid, t in self._tracks.items()
                             if t['lost'] > self.max_lost]
                for tid in to_remove:
                    del self._tracks[tid]
            return []

        det_boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        n_det = len(det_boxes)

        if not self._tracks:
            if self._locked:
                return [-1] * n_det
            ids = []
            for d in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {'bbox': d['bbox'], 'lost': 0}
                ids.append(tid)
            return ids

        track_ids = list(self._tracks.keys())
        track_boxes = np.array(
            [self._tracks[tid]['bbox'] for tid in track_ids], dtype=np.float32
        )

        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        assigned_det = set()
        assigned_track = set()
        matches = {}

        # Pass 1: match at normal IoU threshold
        pairs = []
        for ti in range(len(track_ids)):
            for di in range(n_det):
                if iou_matrix[ti, di] > self.iou_threshold:
                    pairs.append((iou_matrix[ti, di], ti, di))
        pairs.sort(reverse=True)

        for _, ti, di in pairs:
            if ti in assigned_track or di in assigned_det:
                continue
            matches[di] = track_ids[ti]
            assigned_track.add(ti)
            assigned_det.add(di)

        # Pass 2 (locked mode only): try re-acquiring lost assigned tracks
        # at a lower IoU threshold — the student may have shifted slightly
        if self._locked:
            lost_pairs = []
            for ti in range(len(track_ids)):
                if ti in assigned_track:
                    continue
                tid = track_ids[ti]
                if tid not in self._assigned_tids:
                    continue
                # Only try re-acquire if this track has been lost
                if self._tracks[tid]['lost'] == 0:
                    continue
                for di in range(n_det):
                    if di in assigned_det:
                        continue
                    if iou_matrix[ti, di] > self.reacquire_iou_threshold:
                        lost_pairs.append((iou_matrix[ti, di], ti, di))
            lost_pairs.sort(reverse=True)

            for _, ti, di in lost_pairs:
                if ti in assigned_track or di in assigned_det:
                    continue
                matches[di] = track_ids[ti]
                assigned_track.add(ti)
                assigned_det.add(di)

        result_ids = []
        for di in range(n_det):
            if di in matches:
                tid = matches[di]
                self._tracks[tid]['bbox'] = detections[di]['bbox']
                self._tracks[tid]['lost'] = 0
                result_ids.append(tid)
            elif self._locked:
                result_ids.append(-1)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    'bbox': detections[di]['bbox'], 'lost': 0
                }
                result_ids.append(tid)

        for ti, tid in enumerate(track_ids):
            if ti not in assigned_track:
                self._tracks[tid]['lost'] += 1
                # Never delete assigned tracks — keep last-known bbox
                if not self._locked and self._tracks[tid]['lost'] > self.max_lost:
                    del self._tracks[tid]

        return result_ids

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        m, n = len(boxes_a), len(boxes_b)
        iou = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            xa1, ya1, xa2, ya2 = boxes_a[i]
            area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
            for j in range(n):
                xb1, yb1, xb2, yb2 = boxes_b[j]
                area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
                ix1 = max(xa1, xb1)
                iy1 = max(ya1, yb1)
                ix2 = min(xa2, xb2)
                iy2 = min(ya2, yb2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0.0
        return iou


# ═══════════════════════════════════════════════════════════════
#  OBJECT-TO-STUDENT ASSOCIATION
# ═══════════════════════════════════════════════════════════════

def _bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_inside_bbox(obj_bbox, student_bbox):
    """Check if the center of obj_bbox is inside student_bbox."""
    cx = (obj_bbox[0] + obj_bbox[2]) / 2
    cy = (obj_bbox[1] + obj_bbox[3]) / 2
    return (student_bbox[0] <= cx <= student_bbox[2] and
            student_bbox[1] <= cy <= student_bbox[3])


def associate_objects_to_students(obj_dets, student_bboxes_by_tid):
    """Associate each object detection to the nearest assigned student.

    Args:
        obj_dets: list of detection dicts (phone/cheat_sheet)
        student_bboxes_by_tid: {track_id: [x1,y1,x2,y2]} for assigned students

    Returns:
        list of (det, track_id) pairs. track_id is -1 if no match.
    """
    results = []
    for det in obj_dets:
        obj_box = det['bbox']
        best_tid = -1
        best_iou = 0.0

        for tid, s_box in student_bboxes_by_tid.items():
            iou = _bbox_iou(obj_box, s_box)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

            # Also match if object center is inside student bbox
            if best_tid == -1 and _center_inside_bbox(obj_box, s_box):
                best_tid = tid

        if best_iou >= ASSOC_IOU_THRESH:
            results.append((det, best_tid))
        elif best_tid != -1:
            # Matched by center-inside
            results.append((det, best_tid))
        else:
            results.append((det, -1))

    return results


# ═══════════════════════════════════════════════════════════════
#  FLASK WEB STREAM
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Cellphone / Cheat Sheet Detection (Pi + Hailo)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
            display: flex; flex-direction: column; align-items: center;
        }
        h1 { color: #0ff; text-shadow: 0 0 10px rgba(0,255,255,0.5); margin-bottom: 10px; }
        .info { color: #aaa; margin-bottom: 20px; text-align: center; }
        .stream-container {
            border: 2px solid #0ff; border-radius: 8px;
            box-shadow: 0 0 20px rgba(0,255,255,0.3);
            overflow: hidden; max-width: 90vw;
        }
        .stream-container img { display: block; width: 100%; height: auto; }
        .footer { margin-top: 20px; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>AISENTINEL - Cellphone / Cheat Sheet Detection</h1>
    <p class="info">Raspberry Pi 5 + Hailo AI HAT | Cellphone &amp; Cheat Sheet</p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <p class="footer">Stream: MJPEG | Press Ctrl+C in terminal to stop</p>
</body>
</html>
"""


def create_flask_app():
    app = Flask(__name__)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route('/')
    def index():
        return render_template_string(HTML_PAGE)

    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                with _frame_lock:
                    frame = _latest_frame
                if frame is not None:
                    _, jpeg = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + jpeg.tobytes() + b'\r\n')
                else:
                    time.sleep(0.05)
                time.sleep(0.03)

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def start_web_server(port=8080):
    app = create_flask_app()
    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, threaded=True),
        daemon=True,
    )
    thread.start()
    return thread


# ═══════════════════════════════════════════════════════════════
#  ROI CALIBRATION
# ═══════════════════════════════════════════════════════════════

def calibrate_roi(frame, disp_scale):
    """Let the user draw a polygon ROI on the first frame.

    Controls:
        Left-click   -> add vertex
        Right-click  -> close polygon (need >= 3 vertices)
        BACKSPACE    -> undo last vertex
        S            -> skip (no ROI, track entire frame)
        ESC          -> cancel / quit

    Returns:
        np.array of shape (N, 2) with polygon vertices, None to skip,
        or "CANCEL" string to abort.
    """
    fh, fw = frame.shape[:2]
    win_name = "AISENTINEL - Draw ROI Boundary"
    vertices = []

    def on_mouse(event, mx, my, flags, param):
        ox = int(mx / disp_scale)
        oy = int(my / disp_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            vertices.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(vertices) >= 3:
                vertices.append("CLOSE")

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    instructions = [
        "LEFT-CLICK: add vertex | RIGHT-CLICK: close polygon (>= 3 pts)",
        "BACKSPACE: undo | S: skip ROI (use full frame) | ESC: quit",
    ]

    closed = False

    while True:
        if vertices and vertices[-1] == "CLOSE":
            vertices.pop()
            if len(vertices) >= 3:
                closed = True

        display = frame.copy()

        for i in range(len(vertices)):
            cv2.circle(display, vertices[i], 5, (0, 255, 255), -1, cv2.LINE_AA)
            if i > 0:
                cv2.line(display, vertices[i - 1], vertices[i],
                         (0, 255, 255), 2, cv2.LINE_AA)

        if closed and len(vertices) >= 3:
            pts = np.array(vertices, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0, 40))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        status = f"Vertices: {len(vertices)}"
        if closed:
            status += " [CLOSED - press ENTER to confirm, BACKSPACE to edit]"
        cv2.putText(display, status, (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if closed else (0, 255, 255), 2)

        if disp_scale < 1.0:
            show = cv2.resize(display, (int(fw * disp_scale), int(fh * disp_scale)))
        else:
            show = display

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win_name)
            return "CANCEL"

        elif key in (ord("s"), ord("S")):
            cv2.destroyWindow(win_name)
            log_info("ROI skipped — tracking entire frame.")
            return None

        elif key == 13 and closed:  # ENTER to confirm
            cv2.destroyWindow(win_name)
            roi = np.array(vertices, dtype=np.int32)
            log_info(f"ROI set with {len(vertices)} vertices.")
            return roi

        elif key == 8:  # BACKSPACE
            if vertices:
                vertices.pop()
                closed = False

    cv2.destroyWindow(win_name)
    return None


def filter_detections_by_roi(detections, roi_polygon):
    """Filter detections to only those whose bbox center is inside the ROI.

    Args:
        detections: list of dicts with 'bbox' key [x1, y1, x2, y2]
        roi_polygon: np.array of shape (N, 2), or None (no filtering)

    Returns:
        Filtered list of detections.
    """
    if roi_polygon is None:
        return detections

    contour = roi_polygon.reshape(-1, 1, 2)
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
            filtered.append(det)
    return filtered


# ═══════════════════════════════════════════════════════════════
#  FILE DIALOG
# ═══════════════════════════════════════════════════════════════

def select_video_dialog():
    """Open a tkinter file dialog to select a video file."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="AISENTINEL - Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ("All files", "*.*"),
        ]
    )
    root.destroy()
    return path if path else None


# ═══════════════════════════════════════════════════════════════
#  ASSIGNMENT PHASE (OpenCV window on local display)
# ═══════════════════════════════════════════════════════════════

def run_assignment_phase(first_frame, student_dets, track_ids, disp_scale):
    """Interactive student number assignment on the first frame.

    Args:
        first_frame: BGR image
        student_dets: list of person detection dicts with 'bbox' and 'confidence'
        track_ids: list of int track IDs aligned with student_dets

    Returns:
        student_map: {track_id: student_number} or None if cancelled.
    """
    if not student_dets:
        log_info("No students detected in the first frame.")
        log_info("Press any key to proceed without assignments (or ESC to quit).")
        cv2.imshow("AISENTINEL - Assign Students", first_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AISENTINEL - Assign Students")
        if key == 27:
            return None
        return {}

    persons = []
    for i, det in enumerate(student_dets):
        persons.append({
            "track_id": track_ids[i],
            "bbox": tuple(det['bbox']),
            "confidence": det['confidence'],
        })

    student_map = {}
    selected_idx = -1
    input_buffer = ""

    fh, fw = first_frame.shape[:2]
    win_name = "AISENTINEL - Assign Students"

    def on_mouse(event, mx, my, flags, param):
        nonlocal selected_idx, input_buffer
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox = int(mx / disp_scale)
        oy = int(my / disp_scale)
        for i, p in enumerate(persons):
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                selected_idx = i
                input_buffer = ""
                return
        selected_idx = -1
        input_buffer = ""

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    instructions = [
        "Click a student -> type student # -> ENTER to assign",
        "Press S to START detection | ESC to quit",
    ]

    while True:
        display = first_frame.copy()

        for i, p in enumerate(persons):
            tid = p["track_id"]
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]

            if i == selected_idx:
                color = COL_SELECTED
                thickness = 3
            elif tid in student_map:
                color = COL_HUD
                thickness = 2
            else:
                color = COL_UNASSIGNED
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            if tid in student_map:
                draw_label(display, f"Student #{student_map[tid]}", x1, y1 - 2,
                           COL_HUD)
            else:
                draw_label(display, f"[unassigned] (ID:{tid}) {p['confidence']:.0%}",
                           x1, y1 - 2, COL_UNASSIGNED)

            if i == selected_idx and input_buffer:
                draw_label(display, f"Typing: {input_buffer}_", x1, y2 + 18,
                           COL_SELECTED, (0, 0, 0))
            elif i == selected_idx:
                draw_label(display, "Selected - type a number", x1, y2 + 18,
                           COL_SELECTED, (0, 0, 0))

        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

        assigned = len(student_map)
        total = len(persons)
        status = f"Assigned: {assigned}/{total} students"
        cv2.putText(display, status, (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_HUD if assigned > 0 else COL_UNASSIGNED, 2)

        if disp_scale < 1.0:
            show = cv2.resize(display, (int(fw * disp_scale), int(fh * disp_scale)))
        else:
            show = display

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(win_name)
            return None

        elif key in (ord("s"), ord("S")):
            if len(student_map) == 0:
                log_info("Assign at least one student before starting.")
                continue
            break

        elif key == 13:  # ENTER
            if selected_idx >= 0 and input_buffer:
                num = int(input_buffer)
                tid = persons[selected_idx]["track_id"]
                dup_tid = None
                for t, n in student_map.items():
                    if n == num and t != tid:
                        dup_tid = t
                        break
                if dup_tid is not None:
                    log_info(f"Warning: Student #{num} was already assigned "
                             f"to track {dup_tid}. Reassigning to track {tid}.")
                    del student_map[dup_tid]
                student_map[tid] = num
                log_info(f"Assigned Student #{num} to person (track ID: {tid})")
                selected_idx = -1
                input_buffer = ""

        elif key == 8:  # BACKSPACE
            if input_buffer:
                input_buffer = input_buffer[:-1]

        elif ord("0") <= key <= ord("9"):
            if selected_idx >= 0:
                input_buffer += chr(key)

    cv2.destroyWindow(win_name)

    log_info(f"Assignment complete: {len(student_map)} students assigned.")
    for tid, num in sorted(student_map.items(), key=lambda x: x[1]):
        log_info(f"  Student #{num} -> Track ID {tid}")
    return student_map


# ═══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_detection(cap, person_detector, detector, tracker, student_map, video_path,
                   port, roi_polygon=None, source_mode="video", source_fps=None):
    """Run detection loop, streaming annotated frames via Flask."""
    global _latest_frame

    source_label = str(video_path)
    video_name = Path(source_label).stem
    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_mode == "video" else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

    roi_str = (f"Yes ({len(roi_polygon)} vertices)"
               if roi_polygon is not None else "No (full frame)")

    print()
    print("=" * 70)
    print(f"  AISENTINEL - Cellphone / Cheat Sheet Detection (Pi + Hailo)")
    source_heading = "Video" if source_mode == "video" else "Webcam"
    print(f"  {source_heading:8s}: {Path(source_label).name}")
    if total_frames > 0:
        print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    else:
        print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Live source")
    print(f"  Students : {len(student_map)} assigned")
    print(f"  ROI      : {roi_str}")
    print(f"  Detecting: cellphone | cheat_sheet")
    print(f"  Alerting : cellphone | cheat_sheet")
    print(f"  Cooldown : {EVENT_COOLDOWN_SEC}s between repeated flags")
    print(f"  Evidence : {EVIDENCE_DIR}")
    local_ip = get_local_ip()
    print(f"  Stream   : http://{local_ip}:{port}")
    print("=" * 70)
    print()

    assigned_tids = set(student_map.keys())

    # Per-(student, class) cooldown: (track_id, class_name) -> last_alert_time
    last_alert_time = defaultdict(lambda: -999.0)

    frame_idx = 1  # frame 1 already processed in assignment
    stats = defaultdict(int)
    total_alerts = 0
    t_start = time.perf_counter()
    source_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source_mode == "video":
                    log_info("End of video reached.")
                else:
                    log_info("Webcam stream ended.")
                break
            frame_idx += 1
            ts_sec = frame_idx / fps if source_mode == "video" else time.perf_counter() - source_start

            # ── Inference ─────────────────────────────────────
            t0 = time.perf_counter()

            # Person detection via pose model
            person_dets = person_detector.detect_persons(frame)
            person_dets = filter_detections_by_roi(person_dets, roi_polygon)

            # Object detection via sentinel model
            all_dets = detector.detect(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # ── Filter by ROI ─────────────────────────────────
            all_dets = filter_detections_by_roi(all_dets, roi_polygon)

            # ── Filter object detections by confidence ────────
            object_dets = []
            for det in all_dets:
                cls_name = det['class_name']
                min_conf = CONFIDENCE_THRESHOLDS.get(cls_name, 0.25)
                if det['confidence'] < min_conf:
                    continue
                if cls_name in OBJECT_CLASSES:
                    object_dets.append(det)

            # ── Track students (person detections) ────────────
            track_ids = tracker.update(person_dets)

            # Build bbox map for assigned students
            student_bboxes = {}  # track_id -> bbox
            for i, det in enumerate(person_dets):
                tid = track_ids[i]
                if tid in assigned_tids:
                    student_bboxes[tid] = det['bbox']

            # ── Associate objects to students ─────────────────
            obj_associations = associate_objects_to_students(
                object_dets, student_bboxes
            )

            # ── Annotate frame ────────────────────────────────
            annotated = frame.copy()
            frame_events = []

            # Draw ROI boundary
            if roi_polygon is not None:
                cv2.polylines(annotated, [roi_polygon], isClosed=True,
                              color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Draw assigned student bboxes
            for i, det in enumerate(person_dets):
                tid = track_ids[i]
                if tid not in assigned_tids:
                    continue
                x1, y1, x2, y2 = det['bbox']
                snum = student_map[tid]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_HUD, 2)
                draw_label(annotated, f"Student #{snum}", x1, y1 - 2, COL_HUD)

            # Draw objects and generate alerts
            for det, assoc_tid in obj_associations:
                cls_name = det['class_name']
                conf = det['confidence']
                x1, y1, x2, y2 = det['bbox']
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))

                # Draw bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if assoc_tid != -1 and assoc_tid in student_map:
                    snum = student_map[assoc_tid]
                    draw_label(annotated, f"{cls_name} {conf:.0%} [S#{snum}]",
                               x1, y1 - 2, color)
                else:
                    draw_label(annotated, f"{cls_name} {conf:.0%}",
                               x1, y1 - 2, color)

                stats[cls_name] += 1

                # Alert only for alert classes, only for assigned students
                if (cls_name in ALERT_CLASSES and
                        assoc_tid != -1 and assoc_tid in student_map):
                    snum = student_map[assoc_tid]
                    cooldown_key = (assoc_tid, cls_name)
                    if (ts_sec - last_alert_time[cooldown_key]) >= EVENT_COOLDOWN_SEC:
                        total_alerts += 1
                        log_alert(cls_name, snum, conf, ts_sec)
                        save_evidence(annotated, snum, cls_name, conf, ts_sec)
                        last_alert_time[cooldown_key] = ts_sec
                        frame_events.append((cls_name, snum))

            # ── HUD ───────────────────────────────────────────
            ts_text = fmt_ts(ts_sec)
            elapsed_wall = time.perf_counter() - t_start
            actual_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0
            n_tracked = len(student_bboxes)

            frame_label = (
                f"Frame: {frame_idx}/{total_frames}"
                if total_frames > 0 else f"Frame: {frame_idx}"
            )
            hud_lines = [
                f"{frame_label} | Time: {ts_text}",
                f"Tracked: {n_tracked}/{len(student_map)} | "
                f"Alerts: {total_alerts} | "
                f"Inf: {inference_ms:.0f}ms | FPS: {actual_fps:.1f}",
            ]
            for i, line in enumerate(hud_lines):
                y_pos = 25 + i * 28
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COL_FLAGGED if total_alerts else COL_HUD,
                            2, cv2.LINE_AA)

            # Alert banner
            if frame_events:
                banner_y = h - 40
                for cls_name, snum in frame_events:
                    txt = f"ALERT: Student #{snum} - {cls_name.upper()}"
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                                4, cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_FLAGGED,
                                2, cv2.LINE_AA)
                    banner_y -= 35

            # Timestamp watermark bottom-right
            (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Push to web stream ────────────────────────────
            with _frame_lock:
                _latest_frame = annotated

            # Progress
            if frame_idx % 500 == 0:
                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames}) | "
                             f"FPS: {actual_fps:.1f}")
                else:
                    log_info(f"Live progress: {frame_idx} frames | {ts_text} | "
                             f"FPS: {actual_fps:.1f}")

    except KeyboardInterrupt:
        log_info("Stopped by user.")

    # ── Summary ───────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 70)
    print(f"  Summary: {Path(source_label).name}")
    print("-" * 70)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Average FPS      : {frame_idx / elapsed:.1f}" if elapsed > 0 else "")
    print(f"  Students tracked : {len(student_map)}")
    print(f"  Total alerts     : {total_alerts}")
    for cls_name, count in sorted(stats.items()):
        marker = " (ALERT)" if cls_name in ALERT_CLASSES else " (drawn only)"
        print(f"    {cls_name:20s}: {count} detections{marker}")
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print(f"  No alerts triggered.")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Cellphone / Cheat Sheet Detection (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_cellphone_cheat_pi.py
  python3 front_node_cellphone_cheat_pi.py --model /path/to/model.hef
  python3 front_node_cellphone_cheat_pi.py --port 9090
  python3 front_node_cellphone_cheat_pi.py --confidence 0.4
        """,
    )
    parser.add_argument("--video", default=None,
                        help="Optional path to a video file")
    parser.add_argument("--model", default=None,
                        help=f"Path to detection HEF model (default: {OBJ_MODEL_PATH})")
    parser.add_argument("--pose-model", default=None,
                        help=f"Path to pose HEF model for person detection (default: {POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080,
                        help="Flask web server port (default: 8080)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Base detection confidence (default: 0.25)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AISENTINEL - Cellphone / Cheat Sheet Detection (Pi + Hailo)")
    print("  Detects: phone | cheat_sheet")
    print("=" * 60)
    print()

    video_path = pi_ui.select_video(args.video, select_video_dialog)
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)

    pose_model_arg = pi_ui.select_pose_model(args.pose_model)
    if not pose_model_arg:
        log_info("No pose model selected. Exiting.")
        sys.exit(0)

    model_arg = pi_ui.select_object_model(args.model)
    if not model_arg:
        log_info("No object model selected. Exiting.")
        sys.exit(0)

    # ── Validate Hailo ──────────────────────────────────────
    if not HAILO_AVAILABLE:
        print(f"{TC.RED}[ERROR] hailo_platform is required.{TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    model_path = Path(model_arg)
    if not model_path.exists():
        print(f"{TC.RED}[ERROR] HEF model not found: {model_path}{TC.RESET}")
        sys.exit(1)

    pose_path = Path(pose_model_arg)
    if not pose_path.exists():
        print(f"{TC.RED}[ERROR] Pose HEF model not found: {pose_path}{TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    # ── Select video via file dialog ────────────────────────
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Load Hailo models (shared VDevice) ─────────────────
    shared_vdevice = VDevice()
    log_info("Hailo VDevice created (shared between both models).")

    # Pose model for person detection & tracking
    person_detector = HailoPoseEstimator(
        str(pose_path),
        conf_threshold=PERSON_CONFIDENCE,
        vdevice=shared_vdevice,
    )

    # Object detection model for phone / cheat_sheet.
    # The model also outputs 'hand', but this script ignores that class.
    detector = HailoObjectDetector(
        str(model_path),
        conf_threshold=args.confidence,
        vdevice=shared_vdevice,
    )

    # Show class info
    print(f"\n{TC.BOLD}Model classes:{TC.RESET}")
    for idx, name in detector.class_names.items():
        role = "  << ALERT" if name in ALERT_CLASSES else "  << IGNORED"
        thresh = CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){role}")
    print()

    # ── Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{TC.RED}[ERROR] Cannot open video: {video_path}{TC.RESET}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{TC.RED}[ERROR] Cannot read first frame.{TC.RESET}")
        sys.exit(1)

    log_info(f"Video resolution: {w}x{h}")

    # ── ROI calibration ────────────────────────────────────
    log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        log_info("ROI calibration cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    log_info("Running person detection on first frame for student assignment...")

    # ── Detect persons on first frame (pose model) ──────────
    first_student_dets = person_detector.detect_persons(first_frame)
    first_student_dets = filter_detections_by_roi(first_student_dets, roi_polygon)

    # Create tracker and seed with person detections
    tracker = IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_student_dets)

    roi_label = " (within ROI)" if roi_polygon is not None else ""
    log_info(f"Detected {len(first_student_dets)} students on first frame{roi_label}.")

    # Also show object detections on first frame for context
    first_obj_dets = [d for d in detector.detect(first_frame)
                      if d['class_name'] in OBJECT_CLASSES]
    if first_obj_dets:
        log_info(f"Also detected: " + ", ".join(
            f"{d['class_name']}({d['confidence']:.0%})" for d in first_obj_dets))

    print()
    print(f"  {TC.BOLD}Instructions:{TC.RESET}")
    print(f"    1. Click on a student bbox to select them (cyan highlight)")
    print(f"    2. Type the student number (digits)")
    print(f"    3. Press ENTER to assign")
    print(f"    4. Repeat for each student you want to monitor")
    print(f"    5. Press S to start detection")
    print()

    # ── Assignment phase (local OpenCV window) ──────────────
    student_map = run_assignment_phase(
        first_frame, first_student_dets, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        log_info("No students assigned. Exiting.")
        sys.exit(0)

    # ── Lock tracker to only assigned students ──────────────
    tracker.keep_only(set(student_map.keys()))
    log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    # ── Start Flask web server ──────────────────────────────
    if not FLASK_AVAILABLE:
        print(f"{TC.RED}[ERROR] Flask is required for web streaming.{TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(args.port)
    local_ip = get_local_ip()
    log_info(f"Web stream at http://{local_ip}:{args.port}")

    # ── Run detection ───────────────────────────────────────
    log_info("Starting detection...")
    run_detection(cap, person_detector, detector, tracker, student_map, video_path,
                  args.port, roi_polygon=roi_polygon)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
