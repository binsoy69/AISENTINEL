#!/usr/bin/env python3
"""
Hands Under Table Detection - Raspberry Pi + Hailo AI HAT
==========================================================
Pi counterpart of front_node_hands_under_table_pc.py.

Detects when a student's hands disappear near a calibrated student-side
table-edge line and then stay missing for a sustained period, suggesting
hands are hidden under the table.

Inference runs on the Hailo-8 NPU using two models:
  - Pose model (yolov8s_pose.hef) for person detection & tracking
  - Shared sentinel detection model (sentinel-yolo11n-min.hef); this
    script uses only its 'hand' class

Algorithm:
  1. Detect persons via pose model, IoU-track for persistent IDs
  2. Detect hands via the shared sentinel model (only 'hand' class used)
  3. Associate each detected hand to the nearest tracked student
  4. Define one student-side table-edge trigger line per assigned student
  5. If the last visible hand was near that line, arm an under-table candidate
  6. Majority-vote over a sliding window to smooth missed detections
  7. Flag only when the edge-disappearance candidate stays missing >= threshold

Workflow:
  1. File dialog opens to select a video
  2. ROI calibration: draw a polygon boundary (limits tracking area)
  3. First frame: click detected persons to assign student numbers
  4. Table-edge calibration: draw one 2-point line per assigned student
  5. Web stream starts at http://<pi-ip>:8080 with live annotations
  6. Console alerts + evidence screenshots saved to ./evidence_hands/

ROI Drawing Controls:
    Left-click      — add vertex
    Right-click     — close polygon (>= 3 pts)
    BACKSPACE       — undo last vertex
    S               — skip (no ROI, track entire frame)
    ESC             — cancel

Student Assignment Controls:
    Left-click on person  -> select (cyan highlight)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start (need >= 1 assignment)
    ESC                   -> quit

Table Edge Line Controls:
    Left-click      — place point 1 / point 2
    ENTER / SPACE   — confirm current student's line
    S               — skip current student
    Z               — undo point / go back to previous student
    ESC             — cancel

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
from collections import defaultdict, deque

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

POSE_MODEL_PATH = REPO_ROOT / "models" / "yolov8s_pose.hef"
HAND_MODEL_PATH = REPO_ROOT / "models" / "sentinel-yolo11n-min.hef"
EVIDENCE_DIR = SCRIPT_DIR / "data" / "evidence_hands"

# ── Detection classes ────────────────────────────────────────
# Shared sentinel model order for models/sentinel-yolo11n-min.hef:
#   {0: 'calculator', 1: 'cellphone', 2: 'cheat_sheet',
#    3: 'hand', 4: 'paper', 5: 'student'}
HAND_MODEL_CLASS_NAMES = {
    0: "calculator",
    1: "cellphone",
    2: "cheat_sheet",
    3: "hand",
    4: "paper",
    5: "student",
}
CLASS_HAND = "hand"

HAND_CONFIDENCE = 0.3
PERSON_CONFIDENCE = 0.5

# ── Behavior Thresholds ─────────────────────────────────────
HANDS_MISSING_SUSTAIN_SEC = 3.0     # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0           # cooldown between repeated flags

# ── Tracking & Smoothing ────────────────────────────────────
MIN_VISIBLE_HANDS = 2               # require both hands visible to count as "present"
                                    # 1 visible hand -> warning, 0 visible hands -> alert
HAND_ASSOC_MARGIN_PX = 60           # max pixel distance from student bbox to claim a hand
SMOOTH_WINDOW_FRAMES = 12           # sliding window size for majority vote
SMOOTH_MISSING_RATIO = 0.6          # fraction of window that must be "missing" to confirm
STUDENT_ABSENT_RESET_SEC = 2.0      # reset the line monitor if student undetected this long
TABLE_EDGE_NEAR_PX = 35             # hand center must be this close to the table-edge line
EDGE_DISAPPEAR_ARM_SEC = 0.75       # last visible hand must disappear this soon after edge contact

# ── Colors (BGR) ─────────────────────────────────────────────
COL_STUDENT = (0, 255, 0)           # green
COL_UNASSIGNED = (128, 128, 128)    # gray
COL_SELECTED = (255, 255, 0)        # cyan
COL_HAND = (255, 200, 0)            # cyan-ish
COL_ALERT = (0, 0, 255)            # red
COL_WARNING = (0, 165, 255)        # orange
COL_HUD = (0, 255, 0)              # green
COL_ASSOC_LINE = (200, 200, 0)     # teal — hand-to-student association line
COL_EDGE_LINE = (0, 255, 255)      # yellow — student-side trigger line

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


def log_alert(student_id: int, line_idx: int, ts_sec: float, detail: str = ""):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.RED}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{TC.RED}HANDS UNDER TABLE - Student #{student_id} at Line #{line_idx + 1}{TC.RESET}"
        + (f" | {detail}" if detail else "")
    )


def log_warning(student_id: int, line_idx: int, ts_sec: float, detail: str = ""):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.YELLOW}{TC.BOLD}[WARNING @ {ts}]{TC.RESET} "
        f"{TC.YELLOW}HANDS UNDER TABLE WARNING - Student #{student_id} at Line #{line_idx + 1}{TC.RESET}"
        + (f" | {detail}" if detail else "")
    )


def log_info(msg: str):
    print(f"{TC.CYAN}[INFO]{TC.RESET} {msg}")


# ── Drawing helpers ──────────────────────────────────────────
def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)


def save_evidence(annotated_frame, raw_frame, video_name, line_idx, student_id, ts_sec):
    """Save both annotated and raw evidence frames."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")

    fname_ann = f"{video_name}_line{line_idx + 1}_sid{student_id}_{ts_str}_annotated.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname_ann), annotated_frame)

    fname_raw = f"{video_name}_line{line_idx + 1}_sid{student_id}_{ts_str}_raw.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname_raw), raw_frame)

    log_info(f"Evidence saved: {fname_ann} + raw")


# ── File Dialog ──────────────────────────────────────────────
def select_video_dialog():
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


# ══════════════════════════════════════════════════════════════
#  HAILO OBJECT DETECTOR
# ══════════════════════════════════════════════════════════════

class HailoObjectDetector:
    """HailoRT Python API wrapper for YOLO object detection on the Hailo NPU.

    Expects a HEF model with built-in NMS postprocessing (yolov8_nms_postprocess).
    Output format: nested list — batch[0][cls_id] = np.array (N, 5) with
    [y1, x1, y2, x2, score] in normalized [0..1] coordinates.
    """

    def __init__(self, hef_path, class_names, conf_threshold=0.5, vdevice=None):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.class_names = class_names  # {class_id: class_name}
        self.num_classes = len(class_names)
        self.conf_threshold = conf_threshold
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
            raw = self._infer_pipeline.infer(input_dict)

        return self._postprocess_nms(raw, img_w, img_h)

    def _postprocess_nms(self, raw_output, img_w, img_h):
        """Parse Hailo NMS postprocessed output.

        The yolov8_nms_postprocess layer outputs a nested structure:
          raw_output[layer_name] = batch_list
            batch_list[0] = classes_list  (length = num_classes)
              classes_list[cls_id] = np.array shape (N_detections, 5)
                each row = [y1, x1, y2, x2, score]  (normalized 0..1)

        N_detections varies per class (inhomogeneous), so we navigate
        with Python indexing — never np.array() on the whole structure.
        """
        if not isinstance(raw_output, dict):
            return []

        results = []

        for _name, arr in raw_output.items():
            try:
                batch_0 = arr[0]
            except (IndexError, TypeError):
                continue

            for cls_id in range(len(batch_0)):
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

                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                thresh = self.conf_threshold

                for row in ca:
                    score = float(row[4])
                    if score < thresh:
                        continue
                    # Hailo NMS: [y1, x1, y2, x2, score] normalized 0..1
                    y1n, x1n, y2n, x2n = row[0], row[1], row[2], row[3]
                    results.append({
                        'bbox': [
                            int(x1n * img_w), int(y1n * img_h),
                            int(x2n * img_w), int(y2n * img_h),
                        ],
                        'confidence': score,
                        'class_id': int(cls_id),
                        'class_name': cls_name,
                    })

        return results


# ══════════════════════════════════════════════════════════════
#  HAILO POSE ESTIMATOR (for person detection)
# ══════════════════════════════════════════════════════════════

def _xywh_to_xyxy(boxes):
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


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
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


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


# ══════════════════════════════════════════════════════════════
#  SIMPLE IoU TRACKER
# ══════════════════════════════════════════════════════════════

class IoUTracker:
    """Lightweight frame-to-frame IoU tracker."""

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self._next_id = 1
        self._tracks = {}       # track_id -> {'bbox': [x1,y1,x2,y2], 'lost': int}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._locked = False

    def keep_only(self, track_ids_to_keep):
        """Remove all tracks except those in the given set.
        After calling, unmatched detections get track_id = -1."""
        to_remove = [tid for tid in self._tracks if tid not in track_ids_to_keep]
        for tid in to_remove:
            del self._tracks[tid]
        self._locked = True

    def update(self, detections):
        """Match detections to existing tracks.

        Returns list of track_ids aligned with detections.
        Unmatched detections get -1 when locked.
        """
        if not detections:
            to_remove = []
            for tid, t in self._tracks.items():
                t['lost'] += 1
                if t['lost'] > self.max_lost:
                    to_remove.append(tid)
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
                if self._tracks[tid]['lost'] > self.max_lost:
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


# ══════════════════════════════════════════════════════════════
#  ROI CALIBRATION (tracking area boundary)
# ══════════════════════════════════════════════════════════════

def calibrate_roi(frame, disp_scale):
    """Draw a polygon ROI to limit tracking area.
    Returns np.array (N,2), None (skip), or "CANCEL"."""
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
            cv2.polylines(display, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
        status = f"Vertices: {len(vertices)}"
        if closed:
            status += " [CLOSED - press ENTER to confirm, BACKSPACE to edit]"
        cv2.putText(display, status, (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if closed else (0, 255, 255), 2)
        if disp_scale < 1.0:
            show = cv2.resize(display, (int(fw * disp_scale), int(fh * disp_scale)))
        else:
            show = display
        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(win_name)
            return "CANCEL"
        elif key in (ord("s"), ord("S")):
            cv2.destroyWindow(win_name)
            log_info("ROI skipped — tracking entire frame.")
            return None
        elif key == 13 and closed:
            cv2.destroyWindow(win_name)
            roi = np.array(vertices, dtype=np.int32)
            log_info(f"ROI set with {len(vertices)} vertices.")
            return roi
        elif key == 8:
            if vertices:
                vertices.pop()
                closed = False

    return None


def filter_detections_by_roi(detections, roi_polygon):
    """Filter detections to only those whose bbox center is inside the ROI."""
    if roi_polygon is None:
        return detections
    contour = roi_polygon.reshape(-1, 1, 2)
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
            filtered.append(det)
    return filtered


# ══════════════════════════════════════════════════════════════
#  ASSIGNMENT PHASE
# ══════════════════════════════════════════════════════════════

def run_assignment_phase(first_frame, detections, track_ids, disp_scale):
    """Interactive student number assignment. Returns {track_id: student_num} or None."""
    if not detections:
        log_info("No persons detected in the first frame.")
        log_info("Press any key to proceed without assignments (or ESC to quit).")
        cv2.imshow("AISENTINEL - Assign Students", first_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AISENTINEL - Assign Students")
        return None if key == 27 else {}

    persons = []
    for i, det in enumerate(detections):
        persons.append({
            "track_id": track_ids[i],
            "bbox": tuple(det['bbox']),
        })

    student_map = {}
    selected_idx = -1
    input_buffer = ""
    fh, fw = first_frame.shape[:2]
    win_name = "AISENTINEL - Assign Students (Hands Under Table)"

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
        "Click a person -> type student # -> ENTER to assign",
        "Press S to START (need >= 1) | ESC to quit",
    ]

    while True:
        display = first_frame.copy()
        for i, p in enumerate(persons):
            tid = p["track_id"]
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            if i == selected_idx:
                color, thickness = COL_SELECTED, 3
            elif tid in student_map:
                color, thickness = COL_STUDENT, 2
            else:
                color, thickness = COL_UNASSIGNED, 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            if tid in student_map:
                draw_label(display, f"Student #{student_map[tid]}", x1, y1 - 2,
                           COL_STUDENT)
            else:
                draw_label(display, f"[unassigned] (ID:{tid})", x1, y1 - 2,
                           COL_UNASSIGNED)
            if i == selected_idx and input_buffer:
                draw_label(display, f"Typing: {input_buffer}_", x1, y2 + 18,
                           COL_SELECTED, (0, 0, 0))
            elif i == selected_idx:
                draw_label(display, "Selected - type a number", x1, y2 + 18,
                           COL_SELECTED, (0, 0, 0))

        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        assigned = len(student_map)
        cv2.putText(display, f"Assigned: {assigned}/{len(persons)} persons",
                    (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_STUDENT if assigned > 0 else COL_UNASSIGNED, 2)
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
        elif key == 13:
            if selected_idx >= 0 and input_buffer:
                num = int(input_buffer)
                tid = persons[selected_idx]["track_id"]
                dup_tid = None
                for t, n in student_map.items():
                    if n == num and t != tid:
                        dup_tid = t
                        break
                if dup_tid is not None:
                    log_info(f"Warning: Student #{num} reassigned from track {dup_tid} to {tid}.")
                    del student_map[dup_tid]
                student_map[tid] = num
                log_info(f"Assigned Student #{num} to track ID {tid}")
                selected_idx = -1
                input_buffer = ""
        elif key == 8:
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


def build_assigned_student_list(detections, track_ids, student_map):
    """Build a stable, student-number-sorted list for per-student line calibration."""
    assigned_students = []
    for i, det in enumerate(detections):
        tid = track_ids[i]
        if tid not in student_map:
            continue
        assigned_students.append({
            "track_id": tid,
            "student_num": student_map[tid],
            "bbox": tuple(det["bbox"]),
        })

    assigned_students.sort(key=lambda item: item["student_num"])
    return assigned_students


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def bbox_center(bbox):
    """Return (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_to_segment_distance(point, segment):
    """Return (distance_px, closest_point) from a point to a 2-point segment."""
    px, py = point
    a = np.array(segment[0], dtype=np.float32)
    b = np.array(segment[1], dtype=np.float32)
    p = np.array([px, py], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom <= 1e-6:
        closest = a
    else:
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        closest = a + t * ab

    dist = float(np.linalg.norm(p - closest))
    return dist, (float(closest[0]), float(closest[1]))


def hand_distance_to_bbox(hand_center, student_bbox):
    """
    Compute the signed distance from a hand center to the nearest edge
    of the student bounding box. Returns 0 if inside, positive if outside.
    """
    hx, hy = hand_center
    sx1, sy1, sx2, sy2 = student_bbox

    cx = max(sx1, min(hx, sx2))
    cy = max(sy1, min(hy, sy2))

    dx = hx - cx
    dy = hy - cy
    return (dx * dx + dy * dy) ** 0.5


# ══════════════════════════════════════════════════════════════
#  PER-STUDENT LINE TRACKING STATE (with temporal smoothing)
# ══════════════════════════════════════════════════════════════

class LineMonitorState:
    """
    Tracks the hands-missing state for a single calibrated line tied to one
    assigned student.
    """

    def __init__(self, line_idx: int, track_id: int):
        self.line_idx = line_idx
        self.assigned_student_id = track_id
        self.history = deque(maxlen=SMOOTH_WINDOW_FRAMES)
        self.hands_missing_start = -1.0
        self.edge_disappear_start = -1.0
        self.last_flagged_at = -999.0
        self.total_alerts = 0
        self.total_warnings = 0
        self.last_student_seen_at = -1.0
        self.last_hand_seen_at = -1.0
        self.last_visible_was_near_edge = False
        self.last_edge_hand_point = None
        self.last_visible_hands = 0

    def can_flag(self, now: float) -> bool:
        return (now - self.last_flagged_at) > EVENT_COOLDOWN_SEC

    def push_observation(self, hands_present: bool):
        self.history.append(hands_present)

    def majority_says_missing(self) -> bool:
        if len(self.history) < SMOOTH_WINDOW_FRAMES // 2:
            return False
        missing_count = sum(1 for v in self.history if not v)
        return missing_count / len(self.history) >= SMOOTH_MISSING_RATIO

    def reset(self):
        self.history.clear()
        self.hands_missing_start = -1.0
        self.edge_disappear_start = -1.0
        self.last_student_seen_at = -1.0
        self.last_hand_seen_at = -1.0
        self.last_visible_was_near_edge = False
        self.last_edge_hand_point = None
        self.last_visible_hands = 0

    def note_visible_hands(self, now: float, nearest_edge_point=None):
        self.last_hand_seen_at = now
        self.last_visible_was_near_edge = nearest_edge_point is not None
        self.last_edge_hand_point = nearest_edge_point
        self.edge_disappear_start = -1.0

    def maybe_arm_edge_disappearance(self, now: float) -> bool:
        if self.edge_disappear_start >= 0:
            return True

        recently_visible = (
            self.last_hand_seen_at > 0
            and (now - self.last_hand_seen_at) <= EDGE_DISAPPEAR_ARM_SEC
        )
        if recently_visible and self.last_visible_was_near_edge:
            self.edge_disappear_start = now
            return True
        return False


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_table_edge_lines(img, student_lines, line_states, assigned_students):
    """Draw the calibrated trigger line for each assigned student."""
    for i, line in enumerate(student_lines):
        state = line_states[i]
        student_num = assigned_students[i]["student_num"]

        if line is not None:
            color = COL_WARNING if state.edge_disappear_start > 0 else COL_EDGE_LINE
            pt1 = tuple(int(v) for v in line[0])
            pt2 = tuple(int(v) for v in line[1])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)

            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.circle(img, pt1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt2, 4, color, -1, cv2.LINE_AA)
            draw_label(img, f"L{i + 1} S#{student_num}", mid_x + 6, mid_y - 6, color,
                       (0, 0, 0))

            if state.last_edge_hand_point is not None:
                px, py = [int(v) for v in state.last_edge_hand_point]
                cv2.circle(img, (px, py), 5, color, 2, cv2.LINE_AA)
        else:
            x1, y1, _, _ = assigned_students[i]["bbox"]
            draw_label(img, f"L{i + 1} S#{student_num} [skipped]", x1, y1 - 22,
                       COL_UNASSIGNED)


def calibrate_table_edge_lines(frame, assigned_students):
    """
    Draw one student-side table-edge line per assigned student.

    Returns a list aligned with assigned_students.
    Each item is either np.array([[x1, y1], [x2, y2]], dtype=np.int32) or None.
    """
    fh, fw = frame.shape[:2]
    scale = min(1.0, 1280 / fw)
    lines = []
    current_points = []
    student_idx = 0

    def on_mouse(event, mx, my, flags, param):
        ox = int(mx / scale)
        oy = int(my / scale)

        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 2:
            current_points.append((ox, oy))

    win = "AISENTINEL - Table Edge Line Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while student_idx < len(assigned_students):
        display = frame.copy()
        current_student = assigned_students[student_idx]

        for i, student in enumerate(assigned_students):
            x1, y1, x2, y2 = [int(v) for v in student["bbox"]]
            is_current = i == student_idx
            color = COL_WARNING if is_current else COL_STUDENT
            thickness = 3 if is_current else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            draw_label(display, f"Student #{student['student_num']}", x1, y1 - 2,
                       color, (0, 0, 0) if is_current else (255, 255, 255))

        for i, line in enumerate(lines):
            if line is None:
                continue
            pt1 = tuple(int(v) for v in line[0])
            pt2 = tuple(int(v) for v in line[1])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            cv2.line(display, pt1, pt2, COL_EDGE_LINE, 2, cv2.LINE_AA)
            cv2.circle(display, pt1, 4, COL_EDGE_LINE, -1, cv2.LINE_AA)
            cv2.circle(display, pt2, 4, COL_EDGE_LINE, -1, cv2.LINE_AA)
            draw_label(display,
                       f"L{i + 1} S#{assigned_students[i]['student_num']}",
                       mid_x + 6, mid_y - 6, COL_EDGE_LINE, (0, 0, 0))

        if len(current_points) == 1:
            cv2.circle(display, current_points[0], 5, COL_EDGE_LINE, -1, cv2.LINE_AA)
        elif len(current_points) == 2:
            cv2.line(display, current_points[0], current_points[1],
                     COL_EDGE_LINE, 2, cv2.LINE_AA)
            cv2.circle(display, current_points[0], 5, COL_EDGE_LINE, -1, cv2.LINE_AA)
            cv2.circle(display, current_points[1], 5, COL_EDGE_LINE, -1, cv2.LINE_AA)

        instructions = [
            f"Student #{current_student['student_num']} ({student_idx + 1}/{len(assigned_students)}): click 2 points on the student-side table edge",
            "ENTER/SPACE: confirm line | S: skip this student | Z: undo/back | ESC: cancel",
            f"Monitoring starts only if a hand disappears within {TABLE_EDGE_NEAR_PX}px of this line",
        ]
        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if scale < 1.0:
            show = cv2.resize(display, (int(fw * scale), int(fh * scale)))
        else:
            show = display

        cv2.imshow(win, show)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):
            if len(current_points) != 2:
                log_info("Place exactly 2 points for the current student's edge line, or press S to skip.")
                continue
            lines.append(np.array(current_points, dtype=np.int32))
            current_points.clear()
            student_idx += 1
        elif key in (ord("s"), ord("S")):
            lines.append(None)
            current_points.clear()
            student_idx += 1
        elif key in (ord("z"), ord("Z")):
            if current_points:
                current_points.pop()
            elif student_idx > 0:
                student_idx -= 1
                prev = lines.pop()
                current_points = prev.tolist() if prev is not None else []
        elif key == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    configured = sum(1 for line in lines if line is not None)
    log_info(
        f"Table-edge calibration complete: {configured}/{len(lines)} student lines configured."
    )
    return lines


# ══════════════════════════════════════════════════════════════
#  FLASK WEB SERVER
# ══════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Hands Under Table Detection (Pi + Hailo)</title>
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
    <h1>AISENTINEL - Hands Under Table Detection</h1>
    <p class="info">Raspberry Pi 5 + Hailo AI HAT | Student &amp; Hand Detection</p>
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


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════

def run_detection(cap, person_detector, hand_detector, tracker, student_map,
                  assigned_students, student_lines, video_path, port, roi_polygon=None):
    """Run detection loop with dual models: pose for persons, shared sentinel model for hands."""
    global _latest_frame

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assigned_tids = {student["track_id"] for student in assigned_students}
    configured_lines = sum(1 for line in student_lines if line is not None)
    duration = total_frames / fps if fps > 0 else 0

    line_states = [
        LineMonitorState(i, student["track_id"])
        for i, student in enumerate(assigned_students)
    ]

    print()
    print("=" * 60)
    local_ip = get_local_ip()
    print(f"  AISENTINEL - Hands Under Table Detection (Pi + Hailo)")
    print(f"  Video        : {Path(video_path).name}")
    print(f"  Resolution   : {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames : {total_frames}")
    print(f"  Students     : {len(assigned_students)} monitored")
    print(f"  Line config  : {configured_lines}/{len(student_lines)}")
    print(f"  ROI          : {'Yes (' + str(len(roi_polygon)) + ' vertices)' if roi_polygon is not None else 'No (full frame)'}")
    print(f"  Threshold    : sustained missing for {HANDS_MISSING_SUSTAIN_SEC:.1f}s")
    print("  Severity     : 1 visible hand -> warning | 0 visible hands -> alert")
    print(f"  Cooldown     : {EVENT_COOLDOWN_SEC}s")
    print(f"  Smoothing    : {SMOOTH_WINDOW_FRAMES} frames, "
          f">= {SMOOTH_MISSING_RATIO:.0%} missing to confirm")
    print(f"  Hand margin  : {HAND_ASSOC_MARGIN_PX}px from student bbox")
    print(f"  Edge zone    : {TABLE_EDGE_NEAR_PX}px | Arm window: {EDGE_DISAPPEAR_ARM_SEC:.2f}s")
    print(f"  Evidence     : {EVIDENCE_DIR}")
    print(f"  Web stream   : http://{local_ip}:{port}")
    print("=" * 60)
    print()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    total_alerts = 0
    total_warnings = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_info("End of video reached.")
                break
            frame_idx += 1
            ts_sec = frame_idx / fps
            raw_frame = frame.copy()

            # ──────────────────────────────────────────────────────
            #  1a. PERSON DETECTION — Pose model on Hailo NPU
            # ──────────────────────────────────────────────────────
            t0 = time.perf_counter()
            person_dets = person_detector.detect_persons(frame)
            person_dets = filter_detections_by_roi(person_dets, roi_polygon)

            # ──────────────────────────────────────────────────────
            #  1b. HAND DETECTION — Shared sentinel model on Hailo NPU
            # ──────────────────────────────────────────────────────
            hand_raw = hand_detector.detect(frame)
            hand_dets = [d for d in hand_raw if d['class_name'] == CLASS_HAND]
            inference_ms = (time.perf_counter() - t0) * 1000

            annotated = frame.copy()

            if roi_polygon is not None:
                cv2.polylines(annotated, [roi_polygon], True,
                              (0, 255, 255), 1, cv2.LINE_AA)

            # ──────────────────────────────────────────────────────
            #  2. IoU TRACK PERSONS — only assigned students
            # ──────────────────────────────────────────────────────
            track_ids = tracker.update(person_dets)

            student_tracks = {}  # track_id -> bbox
            for i, det in enumerate(person_dets):
                tid = track_ids[i]
                if tid == -1 or tid not in assigned_tids:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_UNASSIGNED, 1)
                    continue

                student_tracks[tid] = det['bbox']
                x1, y1, x2, y2 = det['bbox']
                snum = student_map[tid]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_STUDENT, 2)
                draw_label(annotated, f"#{snum} {det['confidence']:.0%}",
                           x1, y1 - 2, COL_STUDENT)

            hand_detections = []
            for det in hand_dets:
                hand_detections.append(det['bbox'])
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_HAND, 2)
                draw_label(annotated, f"hand {det['confidence']:.0%}",
                           x1, y1 - 2, COL_HAND)

            # ──────────────────────────────────────────────────────
            #  3. ASSOCIATE HANDS WITH STUDENTS
            # ──────────────────────────────────────────────────────
            student_hands = defaultdict(list)

            for h_bbox in hand_detections:
                hx, hy = bbox_center(h_bbox)
                best_sid = None
                best_dist = float("inf")

                for track_id, s_bbox in student_tracks.items():
                    dist = hand_distance_to_bbox((hx, hy), s_bbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_sid = track_id

                if best_sid is not None and best_dist <= HAND_ASSOC_MARGIN_PX:
                    student_hands[best_sid].append(h_bbox)

                    sx1, sy1, sx2, sy2 = student_tracks[best_sid]
                    s_cx, s_cy = int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)
                    cv2.line(annotated, (int(hx), int(hy)), (s_cx, s_cy),
                             COL_ASSOC_LINE, 1, cv2.LINE_AA)

            # ──────────────────────────────────────────────────────
            #  4. LINE-LEVEL HAND PRESENCE CHECK
            # ──────────────────────────────────────────────────────
            frame_alerts = []
            frame_warnings = []

            for i, state in enumerate(line_states):
                sid = state.assigned_student_id
                edge_line = student_lines[i] if i < len(student_lines) else None

                if sid in student_tracks:
                    state.last_student_seen_at = ts_sec
                elif (
                    state.last_student_seen_at > 0
                    and (ts_sec - state.last_student_seen_at) > STUDENT_ABSENT_RESET_SEC
                ):
                    state.reset()

                if edge_line is None or sid not in student_tracks:
                    state.push_observation(True)
                    state.hands_missing_start = -1.0
                    state.edge_disappear_start = -1.0
                    state.last_edge_hand_point = None
                    state.last_visible_hands = 0
                    continue

                hands = student_hands.get(sid, [])
                visible_hands = len(hands)
                near_line_hand_count = 0
                nearest_edge_point = None
                nearest_edge_dist = float("inf")

                for h_bbox in hands:
                    hx, hy = bbox_center(h_bbox)
                    edge_dist, edge_point = point_to_segment_distance((hx, hy), edge_line)
                    if edge_dist <= TABLE_EDGE_NEAR_PX:
                        near_line_hand_count += 1
                        if edge_dist < nearest_edge_dist:
                            nearest_edge_dist = edge_dist
                            nearest_edge_point = edge_point

                hands_present = visible_hands >= MIN_VISIBLE_HANDS
                state.last_visible_hands = visible_hands

                if near_line_hand_count > 0:
                    state.note_visible_hands(ts_sec, nearest_edge_point=nearest_edge_point)
                elif not hands_present:
                    state.maybe_arm_edge_disappearance(ts_sec)
                else:
                    state.edge_disappear_start = -1.0
                    state.last_edge_hand_point = None

                state.push_observation(hands_present)
                smoothed_missing = state.majority_says_missing()
                suspicious_missing = smoothed_missing and (state.edge_disappear_start >= 0)

                if suspicious_missing:
                    if state.hands_missing_start < 0:
                        state.hands_missing_start = ts_sec
                    elapsed = ts_sec - state.hands_missing_start

                    if elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.can_flag(ts_sec):
                        state.last_flagged_at = ts_sec
                        snum = student_map.get(sid, sid)

                        if visible_hands == 1:
                            state.total_warnings += 1
                            total_warnings += 1
                            log_warning(
                                snum,
                                i,
                                ts_sec,
                                f"line-disappear, only 1 hand visible for {elapsed:.1f}s",
                            )
                            frame_warnings.append(i)
                        else:
                            state.total_alerts += 1
                            total_alerts += 1
                            log_alert(
                                snum,
                                i,
                                ts_sec,
                                f"line-disappear, 0 hands visible for {elapsed:.1f}s "
                                f"({sum(1 for v in state.history if not v)}/{len(state.history)} frames)",
                            )
                            frame_alerts.append(i)
                else:
                    state.hands_missing_start = -1.0

            draw_table_edge_lines(annotated, student_lines, line_states, assigned_students)

            for i, state in enumerate(line_states):
                if state.hands_missing_start > 0:
                    line = student_lines[i] if i < len(student_lines) else None
                    if line is None:
                        continue

                    elapsed = ts_sec - state.hands_missing_start
                    pt1 = line[0]
                    pt2 = line[1]
                    label_x = int((pt1[0] + pt2[0]) / 2) + 8
                    label_y = int((pt1[1] + pt2[1]) / 2) + 22

                    if elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.last_visible_hands == 0:
                        txt = f"ALERT! 0 hands ({elapsed:.1f}s)"
                        color = COL_ALERT
                    elif elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.last_visible_hands == 1:
                        txt = f"WARNING! 1 hand ({elapsed:.1f}s)"
                        color = COL_WARNING
                    else:
                        txt = f"Watching line ({elapsed:.1f}s)"
                        color = COL_WARNING

                    cv2.putText(annotated, txt, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(annotated, txt, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                                cv2.LINE_AA)

            for line_idx in frame_alerts:
                tid = line_states[line_idx].assigned_student_id
                snum = student_map.get(tid, tid)
                save_evidence(annotated, raw_frame, video_name, line_idx, snum, ts_sec)

            ts_text = fmt_ts(ts_sec)
            hud1 = f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}"
            tracked_count = len(student_tracks)
            hud2 = (f"Monitored: {tracked_count}/{len(assigned_students)} | Hands: {len(hand_detections)} "
                    f"| Lines: {configured_lines} | Alerts: {total_alerts} | Warn: {total_warnings} "
                    f"| Inf: {inference_ms:.0f}ms")

            cv2.putText(annotated, hud1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_HUD, 2, cv2.LINE_AA)

            hud_color = COL_ALERT if total_alerts > 0 else (
                COL_WARNING if total_warnings > 0 else COL_HUD
            )
            cv2.putText(annotated, hud2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)

            if frame_alerts or frame_warnings:
                banner_y = h - 30
                for line_idx in frame_alerts:
                    tid = line_states[line_idx].assigned_student_id
                    snum = student_map.get(tid, tid)
                    txt = f"ALERT: Student #{snum} hands missing near Line #{line_idx + 1}"
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4,
                                cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_ALERT, 2,
                                cv2.LINE_AA)
                    banner_y -= 35
                for line_idx in frame_warnings:
                    tid = line_states[line_idx].assigned_student_id
                    snum = student_map.get(tid, tid)
                    txt = f"WARNING: Student #{snum} long hands-missing event near Line #{line_idx + 1}"
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4,
                                cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WARNING, 2,
                                cv2.LINE_AA)
                    banner_y -= 35

            (tw, th_), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

            with _frame_lock:
                _latest_frame = annotated

            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")

    except KeyboardInterrupt:
        log_info("Interrupted by user.")

    print()
    print("=" * 60)
    print(f"  Summary: {Path(video_path).name}")
    print("-" * 60)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Students         : {len(assigned_students)}")
    print(f"  Line config      : {configured_lines}/{len(student_lines)}")
    print(f"  Total alerts     : {total_alerts}")
    print(f"  Total warnings   : {total_warnings}")
    for i, state in enumerate(line_states):
        if state.total_alerts > 0 or state.total_warnings > 0:
            tid = state.assigned_student_id
            snum = student_map.get(tid, "?")
            print(
                f"    Line #{i + 1:2d} (Student #{snum}) : "
                f"{state.total_alerts} alerts, {state.total_warnings} warnings"
            )
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    elif total_warnings == 0:
        print("  No hands-under-table detected.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Hands Under Table Detection (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_hands_under_table_pi.py
  python3 front_node_hands_under_table_pi.py --pose-model /path/to/pose.hef
  python3 front_node_hands_under_table_pi.py --hand-model /path/to/sentinel-yolo11n-min.hef
  python3 front_node_hands_under_table_pi.py --port 9090
        """,
    )
    parser.add_argument("--pose-model", default=str(POSE_MODEL_PATH),
                        help=f"Path to pose HEF model for person detection (default: {POSE_MODEL_PATH})")
    parser.add_argument("--hand-model", default=str(HAND_MODEL_PATH),
                        help=f"Path to detection HEF model containing the hand class (default: {HAND_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080,
                        help="Flask web server port (default: 8080)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AISENTINEL - Hands Under Table Detection (Pi + Hailo)")
    print("  Person detection : pose model (IoU tracked)")
    print("  Hand detection   : shared sentinel model (hand class only)")
    print("  Trigger logic    : hands disappear near the calibrated table-edge line")
    print(f"  Sustained threshold: {HANDS_MISSING_SUSTAIN_SEC:.1f}s")
    print("  Severity         : 1 visible hand -> warning | 0 visible hands -> alert")
    print(f"  Smoothing window: {SMOOTH_WINDOW_FRAMES} frames "
          f"(missing ratio >= {SMOOTH_MISSING_RATIO:.0%})")
    print("=" * 60)
    print()

    # ── Validate Hailo ──────────────────────────────────────
    if not HAILO_AVAILABLE:
        print(f"{TC.RED}[ERROR] hailo_platform is required.{TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    pose_path = Path(args.pose_model)
    if not pose_path.exists():
        print(f"{TC.RED}[ERROR] Pose HEF model not found: {pose_path}{TC.RESET}")
        sys.exit(1)

    hand_path = Path(args.hand_model)
    if not hand_path.exists():
        print(f"{TC.RED}[ERROR] Hand HEF model not found: {hand_path}{TC.RESET}")
        sys.exit(1)

    # ── Select video via file dialog ────────────────────────
    log_info("Opening file dialog...")
    video_path = select_video_dialog()
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Load Hailo models (shared VDevice) ─────────────────
    shared_vdevice = VDevice()
    log_info("Hailo VDevice created (shared between both models).")

    # Pose model for person detection
    person_detector = HailoPoseEstimator(
        str(pose_path),
        conf_threshold=PERSON_CONFIDENCE,
        vdevice=shared_vdevice,
    )

    # Shared sentinel model; only the 'hand' class is used by this script.
    hand_detector = HailoObjectDetector(
        str(hand_path),
        class_names=HAND_MODEL_CLASS_NAMES,
        conf_threshold=HAND_CONFIDENCE,
        vdevice=shared_vdevice,
    )

    # ── Open video & read first frame ────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{TC.RED}[ERROR] Cannot open video: {video_path}{TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{TC.RED}[ERROR] Cannot read first frame.{TC.RESET}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0
    log_info(f"Video resolution: {w}x{h}")

    # ── ROI calibration (tracking area boundary) ─────────────
    log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = calibrate_roi(first_frame, disp_scale)
    if roi_result is not None and isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        log_info("Cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    # ── Detect persons on first frame ────────────────────────
    log_info("Running pose detection on first frame for student assignment...")
    first_detections = person_detector.detect_persons(first_frame)
    first_detections = filter_detections_by_roi(first_detections, roi_polygon)

    # ── Create tracker and assign initial IDs ────────────────
    tracker = IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)

    log_info(f"Detected {len(first_detections)} persons (within ROI).")
    print()
    print(f"  {TC.BOLD}Instructions:{TC.RESET}")
    print(f"    1. Click on a person to select them (cyan highlight)")
    print(f"    2. Type the student number (digits)")
    print(f"    3. Press ENTER to assign")
    print(f"    4. Repeat for each student you want to monitor")
    print(f"    5. Press S to start")
    print()

    # ── Assignment phase (local OpenCV window) ────────────────
    student_map = run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        log_info("No students assigned. Exiting.")
        sys.exit(0)

    # ── Lock tracker to only assigned students ────────────────
    tracker.keep_only(set(student_map.keys()))
    log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    assigned_students = build_assigned_student_list(
        first_detections, first_track_ids, student_map
    )
    if len(assigned_students) == 0:
        cap.release()
        log_info("No assigned students available for line calibration. Exiting.")
        sys.exit(0)

    log_info("Now draw one student-side table-edge line for each assigned student (or press S to skip a student).")
    student_lines = calibrate_table_edge_lines(first_frame, assigned_students)
    if student_lines is None:
        cap.release()
        log_info("Table-edge calibration cancelled. Exiting.")
        sys.exit(0)

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
    run_detection(cap, person_detector, hand_detector, tracker, student_map,
                  assigned_students, student_lines, video_path, args.port,
                  roi_polygon=roi_polygon)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
