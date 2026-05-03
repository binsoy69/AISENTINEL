#!/usr/bin/env python3
"""
Head Behavior Detection - Raspberry Pi + Hailo AI HAT
=====================================================
Pi counterpart of front_node_head_behavior_pc.py.

Detects two pose-based cheating behaviors from PROJECT.md:
  1. Head Tilting        - ear-to-ear roll angle OR nose yaw offset > threshold, sustained
  2. Shoulder Turn (OVERHEAD)- shoulder-line angle deviation from horizontal, sustained

Both behaviors must be sustained for 4 seconds (configurable) before
triggering an alert and saving an evidence screenshot.

Workflow:
  1. File dialog opens (tkinter) to select a video file
  2. First frame shown via OpenCV - click to assign student numbers
  3. Web stream starts at http://<pi-ip>:8080 with live annotations
  4. Console alerts + evidence screenshots saved to ./evidence/

Inference runs on the Hailo-8 NPU using HailoRT Python API with
yolo_pose_model.hef (no Ultralytics / no GStreamer dependency at runtime).

A simple IoU tracker maintains person identity across frames so that
the student assignments from the first frame persist throughout the video.

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
import math
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

from front_node_pi_model_paths import POSE_MODEL_PATH
import front_node_pi_interactive as pi_ui
from front_node_test_config import (
    DEFAULT_VIDEO_CONFIG_PATH,
    add_config_arg,
    apply_head_config,
    cli_or_config,
    load_test_config,
    path_arg,
)

EVIDENCE_DIR = SCRIPT_DIR / "evidence_head"

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# ── Behavior Thresholds ─────────────────────────────────────
HEAD_TILT_ANGLE_DEG = 30.0      # ear-to-ear roll angle threshold
HEAD_TURN_RATIO = 0.18          # nose offset / shoulder width threshold for yaw detection
SHOULDER_TURN_ANGLE_DEG = 20.0  # shoulder-line deviation from horizontal (overhead cam)
SUSTAINED_SEC = 2.5             # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0       # cooldown between repeated flags
KP_CONF_THRESH = 0.3            # minimum keypoint confidence
HEAD_FACE_CONF_THRESH = 0.45    # stricter face confidence for head-tilt signals
YAW_MIN_SHOULDER_WIDTH_PX = 20.0  # avoid yaw ratio spikes from tiny shoulder spans

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)
COL_UNASSIGNED = (128, 128, 128)
COL_SELECTED = (255, 255, 0)     # cyan
COL_HEAD_TILT = (0, 165, 255)    # orange
COL_SHOULDER_TURN = (255, 191, 0)  # deep sky blue (BGR)
COL_FLAGGED = (0, 0, 255)        # red

# ── Skeleton for drawing ─────────────────────────────────────
SKELETON = [
    (KP_NOSE, 1), (KP_NOSE, 2),       # nose -> eyes
    (1, KP_LEFT_EAR), (2, KP_RIGHT_EAR),  # eyes -> ears
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, 7), (KP_RIGHT_SHOULDER, 8),  # shoulders -> elbows
    (7, 9), (8, 10),                   # elbows -> wrists
    (KP_LEFT_SHOULDER, 11), (KP_RIGHT_SHOULDER, 12),  # shoulders -> hips
    (11, 12),
]

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
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - total) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def log_alert(behavior: str, student_num: int, ts_sec: float,
              detail: str = "", color: str = TC.RED):
    ts = fmt_ts(ts_sec)
    print(
        f"{color}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{color}{behavior} - Student #{student_num}{TC.RESET}"
        + (f" | {detail}" if detail else "")
    )


def log_info(msg: str):
    print(f"{TC.CYAN}[INFO]{TC.RESET} {msg}")


# ── Drawing helpers ──────────────────────────────────────────
def draw_skeleton(img, kp_xy, kp_conf, color=(255, 255, 0)):
    for i, j in SKELETON:
        if i < len(kp_conf) and j < len(kp_conf):
            if kp_conf[i] > KP_CONF_THRESH and kp_conf[j] > KP_CONF_THRESH:
                p1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                p2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    for k in range(min(len(kp_xy), 13)):
        if kp_conf[k] > KP_CONF_THRESH:
            cv2.circle(img, (int(kp_xy[k][0]), int(kp_xy[k][1])),
                       3, color, -1, cv2.LINE_AA)


def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def save_evidence(frame, student_num, behavior, ts_sec):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"student{student_num}_{behavior}_{ts_str}.jpg"
    path = EVIDENCE_DIR / fname
    cv2.imwrite(str(path), frame)
    log_info(f"Evidence saved: {fname}")


# ═══════════════════════════════════════════════════════════════
#  HAILO POSE ESTIMATOR
# ═══════════════════════════════════════════════════════════════

def _xywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
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
    """HailoRT Python API wrapper for YOLOv8-pose inference on the Hailo NPU."""

    def __init__(self, hef_path, conf_threshold=0.25, kpt_threshold=0.3,
                 iou_threshold=0.45):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.iou_threshold = iou_threshold

        log_info(f"Loading HEF model: {hef_path}")
        self.hef = HEF(str(hef_path))

        self.vdevice = VDevice()
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
        for out_info in self.output_vstream_info:
            log_info(f"Model output layer: {out_info.name} -> {out_info.shape}")
        log_info("Hailo device ready.")

    def detect_pose(self, frame):
        """Run pose estimation on a BGR frame.

        Returns list of dicts:
            [{'bbox': [x1,y1,x2,y2], 'confidence': float,
              'keypoints': np.array(17,3)}, ...]
        """
        img_h, img_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        input_dict = {self.input_vstream_info[0].name: input_data}

        with self.network_group.activate(self.network_group_params):
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                results = infer_pipeline.infer(input_dict)

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            if isinstance(results, dict):
                for name, arr in results.items():
                    log_info(f"Output '{name}': shape={np.array(arr).shape}")

        return self._postprocess(results, img_w, img_h)

    def _decode_multiscale_heads(self, raw_output):
        """Decode split YOLO pose heads (64/1/51 channels per scale)."""
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
                np.arange(w, dtype=np.float32),
                indexing="ij",
            )
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

            kpts = kpt_logits.reshape(-1, 17, 3)
            kpts[:, :, 0] = (kpts[:, :, 0] * 2.0 + (anchor_x[:, None] - 0.5)) * stride_x
            kpts[:, :, 1] = (kpts[:, :, 1] * 2.0 + (anchor_y[:, None] - 0.5)) * stride_y
            kpts[:, :, 2] = _sigmoid(kpts[:, :, 2])
            keypoints_flat = kpts.reshape(-1, 51)

            decoded_scales.append(
                np.concatenate([boxes_xywh, confidences, keypoints_flat], axis=1)
            )

        if not decoded_scales:
            return None
        return np.concatenate(decoded_scales, axis=0)

    def _postprocess(self, raw_output, img_w, img_h):
        """Parse YOLOv8-pose output into list of detections."""
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
            keypoints_raw = output[:, 5:]
        else:
            keypoints_raw = output[:, 4:]
            kpt_confs = keypoints_raw[:, 2::3]
            confidences = np.mean(kpt_confs, axis=1)

        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        keypoints_raw = keypoints_raw[mask]

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

        scale_x = img_w / self.input_w
        scale_y = img_h / self.input_h
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        keypoints_list = []
        for kpt_raw in keypoints_raw:
            kpts = kpt_raw.reshape(17, 3)
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y
            keypoints_list.append(kpts)

        keep = _nms(boxes_xyxy, confidences, self.iou_threshold)

        results = []
        for idx in keep:
            results.append({
                'bbox': boxes_xyxy[idx].astype(int).tolist(),
                'confidence': float(confidences[idx]),
                'keypoints': keypoints_list[idx],
            })
        return results


# ═══════════════════════════════════════════════════════════════
#  SIMPLE IoU TRACKER
# ═══════════════════════════════════════════════════════════════

class IoUTracker:
    """Lightweight frame-to-frame IoU tracker for seated students."""

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self._next_id = 1
        self._tracks = {}       # track_id -> {'bbox': [x1,y1,x2,y2], 'lost': int}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def keep_only(self, track_ids_to_keep):
        """Remove all tracks except those in the given set.

        Call after assignment phase to discard unassigned tracks so the
        tracker only maintains state for selected students.
        """
        to_remove = [tid for tid in self._tracks if tid not in track_ids_to_keep]
        for tid in to_remove:
            del self._tracks[tid]
        # Prevent creation of new tracks in subsequent updates
        self._locked = True

    def update(self, detections):
        """Match detections to existing tracks.

        Args:
            detections: list of dicts with 'bbox' key ([x1,y1,x2,y2])

        Returns:
            list of track_ids aligned with detections (same length/order).
            Unmatched detections get track_id = -1 when locked.
        """
        locked = getattr(self, '_locked', False)

        if not detections:
            # Age out lost tracks
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
            if locked:
                # No tracks left and locked — nothing to match
                return [-1] * n_det
            # First frame - create new tracks for all detections
            ids = []
            for d in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {'bbox': d['bbox'], 'lost': 0}
                ids.append(tid)
            return ids

        # Build IoU matrix between existing tracks and new detections
        track_ids = list(self._tracks.keys())
        track_boxes = np.array(
            [self._tracks[tid]['bbox'] for tid in track_ids], dtype=np.float32
        )

        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        # Greedy assignment
        assigned_det = set()
        assigned_track = set()
        matches = {}  # det_idx -> track_id

        # Sort all (track_idx, det_idx) pairs by IoU descending
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

        # Update matched tracks; unmatched detections get -1 when locked
        result_ids = []
        for di in range(n_det):
            if di in matches:
                tid = matches[di]
                self._tracks[tid]['bbox'] = detections[di]['bbox']
                self._tracks[tid]['lost'] = 0
                result_ids.append(tid)
            elif locked:
                # Don't create new tracks — mark as untracked
                result_ids.append(-1)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    'bbox': detections[di]['bbox'], 'lost': 0
                }
                result_ids.append(tid)

        # Age unmatched tracks
        for ti, tid in enumerate(track_ids):
            if ti not in assigned_track:
                self._tracks[tid]['lost'] += 1
                if self._tracks[tid]['lost'] > self.max_lost:
                    del self._tracks[tid]

        return result_ids

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        """Compute IoU between two sets of boxes. Returns (M, N) matrix."""
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
#  BEHAVIOR DETECTION
# ═══════════════════════════════════════════════════════════════

def _kp_clear(kp_conf, idx, threshold=HEAD_FACE_CONF_THRESH):
    return idx < len(kp_conf) and float(kp_conf[idx]) >= threshold


def has_clear_head_signal(kp_conf):
    """Return True when face keypoints are reliable enough for head tilt."""
    ears_clear = (
        _kp_clear(kp_conf, KP_LEFT_EAR)
        and _kp_clear(kp_conf, KP_RIGHT_EAR)
    )
    if ears_clear:
        return True

    nose_clear = _kp_clear(kp_conf, KP_NOSE)
    side_reference_clear = any(
        _kp_clear(kp_conf, idx)
        for idx in (
            KP_LEFT_EYE,
            KP_RIGHT_EYE,
            KP_LEFT_EAR,
            KP_RIGHT_EAR,
        )
    )
    return nose_clear and side_reference_clear


def compute_signed_yaw(kp_xy, kp_conf):
    """
    Compute the signed nose-offset / shoulder-width ratio.
    Positive = nose is to the right of shoulder center in image coords.
    Returns (valid, signed_ratio).
    """
    if (_kp_clear(kp_conf, KP_NOSE) and
            kp_conf[KP_LEFT_SHOULDER] >= KP_CONF_THRESH and
            kp_conf[KP_RIGHT_SHOULDER] >= KP_CONF_THRESH):
        nose_x = float(kp_xy[KP_NOSE][0])
        ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
        rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
        shoulder_width = abs(rs_x - ls_x)
        if shoulder_width >= YAW_MIN_SHOULDER_WIDTH_PX:
            shoulder_center_x = (ls_x + rs_x) / 2.0
            return True, (nose_x - shoulder_center_x) / shoulder_width
    return False, 0.0


def detect_head_tilt(kp_xy, kp_conf, baseline_yaw=0.0):
    """
    Detects head tilting via two complementary signals:

    1. Roll (sideways lean): ear-to-ear angle vs horizontal.
       Triggers when angle > HEAD_TILT_ANGLE_DEG.

    2. Yaw (turning left/right): nose offset from shoulder midpoint,
       normalized by shoulder width.  Triggers when ratio > HEAD_TURN_RATIO.
       The baseline_yaw is subtracted first to compensate for perspective
       distortion when the student sits at the side of the camera's FOV.

    Returns (is_tilted, score) where score is the higher of the two
    normalized signals (0.0 = neutral, 1.0 = at threshold, >1.0 = exceeded).
    """
    roll_score = 0.0
    yaw_score = 0.0

    if not has_clear_head_signal(kp_conf):
        return False, 0.0

    # ── Roll detection (ear-to-ear angle) ──────────────────────
    if (_kp_clear(kp_conf, KP_LEFT_EAR) and
            _kp_clear(kp_conf, KP_RIGHT_EAR)):
        le = kp_xy[KP_LEFT_EAR]
        re = kp_xy[KP_RIGHT_EAR]
        raw = abs(math.degrees(
            math.atan2(float(re[1]) - float(le[1]),
                       float(re[0]) - float(le[0]))
        ))
        angle = raw if raw <= 90 else 180 - raw
        roll_score = angle / HEAD_TILT_ANGLE_DEG if HEAD_TILT_ANGLE_DEG > 0 else 0.0

    # ── Yaw detection (nose offset from shoulder center) ───────
    # Subtract the student's baseline offset so that their natural
    # resting position (due to perspective at the edge of the FOV)
    # reads as ~0, and only actual head turns trigger the alert.
    yaw_face_clear = (
        _kp_clear(kp_conf, KP_NOSE)
        and any(
            _kp_clear(kp_conf, idx)
            for idx in (
                KP_LEFT_EYE,
                KP_RIGHT_EYE,
                KP_LEFT_EAR,
                KP_RIGHT_EAR,
            )
        )
    )
    yaw_valid, signed_yaw = compute_signed_yaw(kp_xy, kp_conf)
    if yaw_face_clear and yaw_valid:
        corrected_yaw = abs(signed_yaw - baseline_yaw)
        yaw_score = corrected_yaw / HEAD_TURN_RATIO if HEAD_TURN_RATIO > 0 else 0.0

    # No valid signal from either method
    if roll_score == 0.0 and yaw_score == 0.0:
        return False, 0.0

    score = max(roll_score, yaw_score)
    return score > 1.0, score


def detect_shoulder_turn(kp_xy, kp_conf):
    """
    Shoulder angle detection optimized for OVERHEAD camera.

    From a top-down view, a student sitting normally has their shoulder
    line roughly horizontal (parallel to the desk edge).  When they
    turn their head/torso to look at a neighbor, the shoulder line
    rotates noticeably.

    Algorithm:
        1. Get left & right shoulder keypoints.
        2. Compute the angle of the shoulder line vs. horizontal.
        3. If the angle exceeds SHOULDER_TURN_ANGLE_DEG, the student
           is turning to look sideways.

    Returns (is_turned, angle_degrees, direction).
    """
    if (kp_conf[KP_LEFT_SHOULDER] < KP_CONF_THRESH or
            kp_conf[KP_RIGHT_SHOULDER] < KP_CONF_THRESH):
        return False, 0.0, ""

    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    ls_y = float(kp_xy[KP_LEFT_SHOULDER][1])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
    rs_y = float(kp_xy[KP_RIGHT_SHOULDER][1])

    # Distance between shoulders — skip if too small / unreliable
    shoulder_dist = math.hypot(rs_x - ls_x, rs_y - ls_y)
    if shoulder_dist < 10:
        return False, 0.0, ""

    # Angle of shoulder line relative to horizontal.
    # atan2 gives 0° when perfectly horizontal.
    raw_angle = math.degrees(
        math.atan2(rs_y - ls_y, rs_x - ls_x)
    )
    # Normalize to 0-90 range (we only care about deviation magnitude)
    angle = abs(raw_angle) if abs(raw_angle) <= 90 else 180 - abs(raw_angle)

    # Direction: positive raw_angle → right shoulder is lower in image
    # (from overhead, this typically means turning RIGHT)
    if raw_angle > 0:
        direction = "RIGHT"
    else:
        direction = "LEFT"

    is_turned = angle > SHOULDER_TURN_ANGLE_DEG
    return is_turned, angle, direction


# ── Per-Student State ────────────────────────────────────────
@dataclass
class StudentState:
    track_id: int
    student_num: int

    # Baseline yaw offset captured during assignment phase.
    # Compensates for perspective distortion when student sits at
    # the side of the camera's field of view.
    baseline_yaw: float = 0.0

    # Head tilt tracking
    head_tilt_start: float = -1.0
    head_tilt_flagged_at: float = -999.0

    # Shoulder turn tracking (overhead camera)
    shoulder_turn_start: float = -1.0
    shoulder_turn_flagged_at: float = -999.0

    def can_flag(self, behavior: str, now: float) -> bool:
        last = {
            "head_tilt": self.head_tilt_flagged_at,
            "shoulder_turn": self.shoulder_turn_flagged_at,
        }.get(behavior, -999.0)
        return (now - last) > EVENT_COOLDOWN_SEC


# ═══════════════════════════════════════════════════════════════
#  FLASK WEB SERVER
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Head Behavior Detection (Pi + Hailo)</title>
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
    <h1>AISENTINEL - Head Behavior Detection</h1>
    <p class="info">Raspberry Pi 5 + Hailo AI HAT | Head Tilt &amp; Shoulder Turn</p>
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

def run_assignment_phase(first_frame, detections, track_ids, disp_scale):
    """Interactive student number assignment on the first frame.

    Args:
        first_frame: BGR image
        detections: list of pose dicts from HailoPoseEstimator
        track_ids: list of int track IDs aligned with detections

    Returns:
        (student_map, baseline_yaw_map) or (None, None) if cancelled.
        student_map: {track_id: student_number}
        baseline_yaw_map: {track_id: signed_yaw_ratio}
    """
    if not detections:
        log_info("No persons detected in the first frame.")
        log_info("Press any key to proceed without assignments (or ESC to quit).")
        cv2.imshow("AISENTINEL - Assign Students", first_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AISENTINEL - Assign Students")
        if key == 27:
            return None, None
        return {}, {}

    # Build person data
    persons = []
    for i, det in enumerate(detections):
        kp = det['keypoints']  # (17, 3)
        persons.append({
            "track_id": track_ids[i],
            "bbox": tuple(det['bbox']),
            "kp_xy": kp[:, :2],
            "kp_conf": kp[:, 2],
        })

    # State
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
        "Click a person -> type student # -> ENTER to assign",
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
                color = COL_NORMAL
                thickness = 2
            else:
                color = COL_UNASSIGNED
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            draw_skeleton(display, p["kp_xy"], p["kp_conf"],
                          color=COL_SELECTED if i == selected_idx else (255, 255, 0))

            if tid in student_map:
                draw_label(display, f"Student #{student_map[tid]}", x1, y1 - 2,
                           COL_NORMAL)
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
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

        assigned = len(student_map)
        total = len(persons)
        status = f"Assigned: {assigned}/{total} persons"
        cv2.putText(display, status, (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_NORMAL if assigned > 0 else COL_UNASSIGNED, 2)

        if disp_scale < 1.0:
            show = cv2.resize(display, (int(fw * disp_scale), int(fh * disp_scale)))
        else:
            show = display

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(win_name)
            return None, None

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

    # Compute baseline yaw offset for each assigned student from
    # their first-frame pose.  This compensates for perspective
    # distortion when a student sits at the side of the camera FOV.
    baseline_yaw_map = {}  # track_id -> signed yaw ratio
    for p in persons:
        tid = p["track_id"]
        if tid in student_map:
            valid, signed_yaw = compute_signed_yaw(p["kp_xy"], p["kp_conf"])
            baseline_yaw_map[tid] = signed_yaw if valid else 0.0

    log_info(f"Assignment complete: {len(student_map)} students assigned.")
    for tid, num in sorted(student_map.items(), key=lambda x: x[1]):
        by = baseline_yaw_map.get(tid, 0.0)
        log_info(f"  Student #{num} -> Track ID {tid}  (baseline yaw: {by:+.3f})")
    return student_map, baseline_yaw_map


# ═══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_detection(cap, estimator, tracker, student_map, video_path, port,
                   baseline_yaw_map=None, source_mode="video", source_fps=None):
    """Run detection loop, streaming annotated frames via Flask.

    Only assigned students are tracked to reduce computation.
    baseline_yaw_map: {track_id: signed_yaw_ratio} from assignment phase.
    """
    global _latest_frame

    source_label = str(video_path)
    video_name = Path(source_label).stem
    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_mode == "video" else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

    print()
    print("=" * 70)
    print(f"  AISENTINEL - Head Behavior Detection (Pi + Hailo)")
    source_heading = "Video" if source_mode == "video" else "Webcam"
    print(f"  {source_heading:8s}: {Path(source_label).name}")
    if total_frames > 0:
        print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    else:
        print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Live source")
    print(f"  Students : {len(student_map)} assigned")
    print(f"  Head tilt roll : >{HEAD_TILT_ANGLE_DEG:.0f} deg (ear-to-ear), sustained {SUSTAINED_SEC}s")
    print(f"  Head tilt yaw  : >{HEAD_TURN_RATIO:.0%} offset ratio (nose/shoulder), sustained {SUSTAINED_SEC}s")
    print(f"  Shoulder turn  : >{SHOULDER_TURN_ANGLE_DEG:.0f} deg (overhead), sustained {SUSTAINED_SEC}s")
    print(f"  Cooldown       : {EVENT_COOLDOWN_SEC}s between repeated flags")
    print(f"  Evidence dir   : {EVIDENCE_DIR}")
    local_ip = get_local_ip()
    print(f"  Web stream     : http://{local_ip}:{port}")
    print("=" * 70)
    print()

    if baseline_yaw_map is None:
        baseline_yaw_map = {}

    # Build student states
    students: dict[int, StudentState] = {}
    for tid, num in student_map.items():
        students[tid] = StudentState(
            track_id=tid, student_num=num,
            baseline_yaw=baseline_yaw_map.get(tid, 0.0),
        )

    # Store assigned track IDs so we only track those persons
    assigned_tids = set(student_map.keys())

    frame_idx = 1  # frame 1 already processed
    stats = defaultdict(int)
    total_alerts = 0
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

            # ── Pose inference on Hailo ─────────────────────────
            t0 = time.perf_counter()
            detections = estimator.detect_pose(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # ── Track only assigned students ──────────────────────
            # Feed all detections to tracker to get IDs, then filter
            # to only assigned students for behavior analysis.
            track_ids = tracker.update(detections)

            # Filter to only assigned students
            assigned_dets = []
            assigned_track_ids = []
            for i, det in enumerate(detections):
                if track_ids[i] in assigned_tids:
                    assigned_dets.append(det)
                    assigned_track_ids.append(track_ids[i])

            annotated = frame.copy()
            frame_events = []

            for i, det in enumerate(assigned_dets):
                tid = assigned_track_ids[i]
                bbox = det['bbox']
                kp = det['keypoints']  # (17, 3)
                kp_xy = kp[:, :2]
                kp_conf = kp[:, 2]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                state = students[tid]
                box_color = COL_NORMAL
                behavior_labels = []

                draw_skeleton(annotated, kp_xy, kp_conf)

                # ── 1. Head Tilt (roll + yaw) ────────────────
                is_tilted, tilt_score = detect_head_tilt(
                    kp_xy, kp_conf, baseline_yaw=state.baseline_yaw)

                if is_tilted:
                    if state.head_tilt_start < 0:
                        state.head_tilt_start = ts_sec
                    elapsed = ts_sec - state.head_tilt_start

                    if elapsed >= SUSTAINED_SEC and state.can_flag("head_tilt", ts_sec):
                        state.head_tilt_flagged_at = ts_sec
                        stats["head_tilt"] += 1
                        total_alerts += 1
                        log_alert("HEAD TILT", state.student_num, ts_sec,
                                  f"score={tilt_score:.2f}, "
                                  f"sustained {elapsed:.1f}s",
                                  TC.YELLOW)
                        frame_events.append(("head_tilt", state.student_num))

                    if elapsed >= 1.0:
                        behavior_labels.append(
                            f"HEAD TILT {tilt_score:.1f}x ({elapsed:.1f}s)")
                        if box_color == COL_NORMAL:
                            box_color = COL_HEAD_TILT
                        if elapsed >= SUSTAINED_SEC:
                            box_color = COL_FLAGGED
                else:
                    state.head_tilt_start = -1.0

                # ── 2. Shoulder Turn (overhead camera) ────────
                is_turned, shoulder_angle, turn_dir = detect_shoulder_turn(
                    kp_xy, kp_conf)

                # Always draw shoulder debug overlay when shoulders visible
                if (kp_conf[KP_LEFT_SHOULDER] > KP_CONF_THRESH and
                        kp_conf[KP_RIGHT_SHOULDER] > KP_CONF_THRESH):
                    ls_pt = (int(kp_xy[KP_LEFT_SHOULDER][0]), int(kp_xy[KP_LEFT_SHOULDER][1]))
                    rs_pt = (int(kp_xy[KP_RIGHT_SHOULDER][0]), int(kp_xy[KP_RIGHT_SHOULDER][1]))
                    shoulder_color = COL_SHOULDER_TURN if is_turned else (100, 200, 100)
                    cv2.line(annotated, ls_pt, rs_pt, shoulder_color, 3, cv2.LINE_AA)
                    mid_x = (ls_pt[0] + rs_pt[0]) // 2
                    mid_y = (ls_pt[1] + rs_pt[1]) // 2
                    angle_txt = f"S:{shoulder_angle:.0f}deg"
                    cv2.putText(annotated, angle_txt,
                                (mid_x + 5, mid_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                shoulder_color, 1, cv2.LINE_AA)

                if is_turned:
                    if state.shoulder_turn_start < 0:
                        state.shoulder_turn_start = ts_sec
                    elapsed = ts_sec - state.shoulder_turn_start

                    if elapsed >= SUSTAINED_SEC and state.can_flag("shoulder_turn", ts_sec):
                        state.shoulder_turn_flagged_at = ts_sec
                        stats["shoulder_turn"] += 1
                        total_alerts += 1
                        log_alert("SHOULDER TURN", state.student_num, ts_sec,
                                  f"direction={turn_dir}, angle={shoulder_angle:.1f}°, "
                                  f"sustained {elapsed:.1f}s",
                                  TC.CYAN)
                        frame_events.append(("shoulder_turn", state.student_num))

                    if elapsed >= 1.0:
                        behavior_labels.append(
                            f"SHOULDER {turn_dir} {shoulder_angle:.0f}° ({elapsed:.1f}s)")
                        if box_color == COL_NORMAL:
                            box_color = COL_SHOULDER_TURN
                        if elapsed >= SUSTAINED_SEC:
                            box_color = COL_FLAGGED
                else:
                    state.shoulder_turn_start = -1.0

                # ── Draw box + student label ────────────────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                draw_label(annotated, f"Student #{state.student_num}",
                           x1, y1 - 2, box_color)

                lbl_y = y1 + 18
                for bl in behavior_labels:
                    draw_label(annotated, bl, x1, lbl_y, box_color)
                    lbl_y += 18

            # ── HUD ─────────────────────────────────────────────
            n_tracked = len(assigned_dets)
            frame_label = (
                f"Frame: {frame_idx}/{total_frames}"
                if total_frames > 0 else f"Frame: {frame_idx}"
            )
            hud_lines = [
                f"{frame_label} | Time: {fmt_ts(ts_sec)}",
                f"Tracked: {n_tracked} | Assigned: {len(students)} | "
                f"Alerts: {total_alerts} | Inf: {inference_ms:.0f}ms",
            ]
            for i, line in enumerate(hud_lines):
                y_pos = 25 + i * 28
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COL_FLAGGED if total_alerts else COL_NORMAL,
                            2, cv2.LINE_AA)

            # ── Alert banner ────────────────────────────────────
            if frame_events:
                banner_y = h - 40
                for behavior, snum in frame_events:
                    txt = (f"ALERT: Student #{snum} - "
                           f"{behavior.replace('_', ' ').upper()}")
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                                4, cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_FLAGGED,
                                2, cv2.LINE_AA)
                    banner_y -= 35

            # ── Save evidence ───────────────────────────────────
            for behavior, snum in frame_events:
                save_evidence(annotated, snum, behavior, ts_sec)

            # ── Push to web stream ──────────────────────────────
            with _frame_lock:
                _latest_frame = annotated

            # ── Progress ────────────────────────────────────────
            if frame_idx % 500 == 0:
                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")
                else:
                    log_info(f"Live progress: {frame_idx} frames | {fmt_ts(ts_sec)}")

    except KeyboardInterrupt:
        log_info("Interrupted by user.")

    # ── Summary ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  Summary: {Path(source_label).name}")
    print("-" * 70)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Students tracked : {len(students)}")
    print(f"  Total alerts     : {total_alerts}")
    for beh, count in sorted(stats.items()):
        print(f"    {beh:25s}: {count}")
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
        description="AISENTINEL - Head Behavior Detection (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_head_behavior_pi.py
  python3 front_node_head_behavior_pi.py --model /path/to/yolo_pose_model.hef
  python3 front_node_head_behavior_pi.py --port 9090
        """,
    )
    add_config_arg(parser, DEFAULT_VIDEO_CONFIG_PATH)
    parser.add_argument("--video", default=None,
                        help="Optional path to a video file")
    parser.add_argument("--model", default=None,
                        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=None,
                        help="Flask web server port (default: config)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Person detection confidence (default: config)")
    args = parser.parse_args()
    config = load_test_config(args.config, DEFAULT_VIDEO_CONFIG_PATH)
    apply_head_config(sys.modules[__name__], config)
    video_arg = cli_or_config(args.video, path_arg(config.video_source.default_video))
    model_arg_value = cli_or_config(args.model, path_arg(config.pose_model))
    port = cli_or_config(args.port, config.port)
    confidence = cli_or_config(args.confidence, config.pose_confidence)

    print()
    print("=" * 60)
    print("  AISENTINEL - Head Behavior Detection (Pi + Hailo)")
    print("  Detects: Head Tilting | Shoulder Turn")
    print("=" * 60)
    print()

    video_path = pi_ui.select_video(video_arg, select_video_dialog)
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)

    model_arg = pi_ui.select_pose_model(model_arg_value)
    if not model_arg:
        log_info("No pose model selected. Exiting.")
        sys.exit(0)

    # ── Validate Hailo ──────────────────────────────────────
    if not HAILO_AVAILABLE:
        print(f"{TC.RED}[ERROR] hailo_platform is required.{TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    model_path = Path(model_arg)
    if not model_path.exists():
        print(f"{TC.RED}[ERROR] HEF model not found: {model_path}{TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    # ── Select video via file dialog ────────────────────────
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Load Hailo pose estimator ───────────────────────────
    estimator = HailoPoseEstimator(
        str(model_path),
        conf_threshold=confidence,
    )

    # ── Open video & read first frame ───────────────────────
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
    log_info("Running pose detection on first frame for student assignment...")

    # ── Detect persons on first frame ───────────────────────
    first_detections = estimator.detect_pose(first_frame)

    # ── Create tracker and assign initial IDs ───────────────
    tracker = IoUTracker(
        iou_threshold=config.tracking.iou_threshold,
        max_lost=config.tracking.max_lost,
    )
    first_track_ids = tracker.update(first_detections)

    log_info(f"Detected {len(first_detections)} persons.")
    print()
    print(f"  {TC.BOLD}Instructions:{TC.RESET}")
    print(f"    1. Click on a person to select them (cyan highlight)")
    print(f"    2. Type the student number (digits)")
    print(f"    3. Press ENTER to assign")
    print(f"    4. Repeat for each student you want to monitor")
    print(f"    5. Press S to start detection")
    print()

    # ── Assignment phase (local OpenCV window) ──────────────
    student_map, baseline_yaw_map = run_assignment_phase(
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

    # ── Start Flask web server ──────────────────────────────
    if not FLASK_AVAILABLE:
        print(f"{TC.RED}[ERROR] Flask is required for web streaming.{TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(port)
    local_ip = get_local_ip()
    log_info(f"Web stream at http://{local_ip}:{port}")

    # ── Run detection ───────────────────────────────────────
    log_info("Starting detection...")
    run_detection(cap, estimator, tracker, student_map, video_path, port,
                  baseline_yaw_map)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
