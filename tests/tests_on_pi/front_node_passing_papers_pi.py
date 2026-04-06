#!/usr/bin/env python3
"""
Passing Papers Detection - Raspberry Pi + Hailo AI HAT
======================================================
Pi counterpart of front_node_passing_papers_pc.py.

Detects when a student passes a paper/note to a side-by-side neighbor
by monitoring multi-signal hand interaction (arm extension + wrist
velocity + wrist proximity).

Algorithm (multi-signal interaction detection):
  1. Track students with IoU tracker (person bboxes + persistent IDs)
  2. For each tracked student, compute per-frame signals:
     a. Arm extension ratio: dist(shoulder,wrist) / dist(shoulder,hip)
     b. Wrist velocity vector (frame-to-frame displacement)
  3. For each same-row student pair, evaluate three interaction signals:
     a. ARM EXTENSION — either student extends arm toward the other
     b. WRIST APPROACH — wrist velocity points toward the neighbor
     c. WRIST PROXIMITY — wrist-to-wrist distance below threshold
  4. A passing event is triggered when >= 2 signals are active,
     proximity has been observed, and the interaction exceeds a
     minimum duration (0.4s default).
  5. All pixel thresholds (row tolerance, wrist proximity, wrist velocity)
     are scaled by person bbox height relative to REFERENCE_BBOX_HEIGHT
     to handle perspective distortion from elevated camera angles.
  6. Per-pair cooldown prevents duplicate alerts.

Workflow:
  1. File dialog opens to select a video file
  2. ROI calibration: draw a polygon boundary on the first frame
     (only persons inside this region will be tracked)
  3. First frame shown with detected persons - click to assign student numbers
  4. Web stream starts at http://<pi-ip>:8080 with live annotations
  5. Console alerts + evidence screenshots saved to ./evidence_passing/

Inference runs on the Hailo-8 NPU using HailoRT Python API with
yolov8m_pose.hef (no Ultralytics / no GStreamer dependency at runtime).

A simple IoU tracker maintains person identity across frames so that
the student assignments from the first frame persist throughout the video.

Controls (Assignment phase - local OpenCV window):
    Left-click on person  -> select person (highlighted in cyan)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start detection (need >= 2 assignments)
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
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

POSE_MODEL_PATH = REPO_ROOT / "models" / "yolov8s_pose.hef"
EVIDENCE_DIR = SCRIPT_DIR / "evidence_passing"

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

# ── Behavior Thresholds ─────────────────────────────────────
EVENT_COOLDOWN_SEC = 10.0       # cooldown between repeated flags for same student pair
KP_CONF_THRESH = 0.3            # minimum keypoint confidence
ROW_TOLERANCE_PX = 80           # max vertical (y-center) difference to be in same "row" (at reference scale)
REFERENCE_BBOX_HEIGHT = 300.0   # approx bbox height (px) for front-row student; thresholds scale relative to this

# ── Wrist Proximity Thresholds ───────────────────────────────
WRIST_PROXIMITY_PX = 160        # max distance between wrists of two students for proximity
MIN_INTERACTION_SEC = 0.03       # minimum proximity duration to trigger alert

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)        # green
COL_UNASSIGNED = (128, 128, 128)
COL_SELECTED = (255, 255, 0)    # cyan
COL_WARNING = (0, 165, 255)     # orange - wrist exiting
COL_FLAGGED = (0, 0, 255)       # red - confirmed passing
COL_WRIST = (255, 0, 255)       # magenta - wrist keypoints
COL_EXIT_LINE = (0, 0, 255)     # red - exit direction line
COL_NEIGHBOR_LINE = (255, 100, 0)  # blue - line to neighbor
COL_HUD = (0, 255, 0)

# ── Skeleton for drawing ────────────────────────────────────
SKELETON = [
    (KP_NOSE, 1), (KP_NOSE, 2),
    (1, KP_LEFT_EAR), (2, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, 7), (KP_RIGHT_SHOULDER, 8),
    (7, KP_LEFT_WRIST), (8, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, 11), (KP_RIGHT_SHOULDER, 12),
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


def log_alert(behavior: str, student_nums, ts_sec: float,
              detail: str = "", color: str = TC.RED):
    ts = fmt_ts(ts_sec)
    if isinstance(student_nums, (list, tuple)):
        students_str = " & ".join(f"#{n}" for n in student_nums)
    else:
        students_str = f"#{student_nums}"
    print(
        f"{color}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{color}{behavior} - Students {students_str}{TC.RESET}"
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


def save_evidence(frame, student_nums, ts_sec):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    nums_str = "_".join(str(n) for n in student_nums)
    fname = f"passing_s{nums_str}_{ts_str}.jpg"
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
        """Remove all tracks except those in the given set."""
        to_remove = [tid for tid in self._tracks if tid not in track_ids_to_keep]
        for tid in to_remove:
            del self._tracks[tid]
        self._locked = True

    def update(self, detections):
        """Match detections to existing tracks.

        Returns list of track_ids aligned with detections.
        Unmatched detections get track_id = -1 when locked.
        """
        locked = getattr(self, '_locked', False)

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
            if locked:
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
            elif locked:
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


# ═══════════════════════════════════════════════════════════════
#  PER-STUDENT & PAIR INTERACTION STATE
# ═══════════════════════════════════════════════════════════════

@dataclass
class StudentState:
    track_id: int
    student_num: int
    total_alerts: int = 0


@dataclass
class PairInteractionState:
    """Tracks wrist proximity interaction state between a pair of students."""
    tid_a: int
    tid_b: int

    # Temporal tracking
    interaction_start: float = -1.0
    last_proximity_time: float = -1.0
    peak_proximity_dist: float = 9999.0

    # Cooldown
    last_flagged_at: float = -999.0

    # Active this frame (reset each frame)
    frame_proximity: bool = False
    frame_proximity_dist: float = 9999.0

    def can_flag(self, now: float) -> bool:
        return (now - self.last_flagged_at) > EVENT_COOLDOWN_SEC

    def reset_interaction(self):
        self.interaction_start = -1.0
        self.last_proximity_time = -1.0
        self.peak_proximity_dist = 9999.0


# ═══════════════════════════════════════════════════════════════
#  GEOMETRY & SIGNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _perspective_scale(bbox_height):
    """Scale factor based on person bbox height relative to reference.
    Larger person (closer to camera) -> larger scale -> larger pixel thresholds.
    """
    return max(bbox_height, 50.0) / REFERENCE_BBOX_HEIGHT


def _kp_pos(kp_xy, kp_conf, idx):
    """Return (x, y) for a keypoint if confident, else None."""
    if idx < len(kp_conf) and kp_conf[idx] > KP_CONF_THRESH:
        return (float(kp_xy[idx][0]), float(kp_xy[idx][1]))
    return None


# -- Wrist-to-Wrist Proximity ------------------------------------
def compute_wrist_proximity(kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf):
    """Compute the minimum distance between wrist keypoints of two students."""
    min_dist = 9999.0

    a_rw = _kp_pos(kp_a_xy, kp_a_conf, KP_RIGHT_WRIST)
    b_lw = _kp_pos(kp_b_xy, kp_b_conf, KP_LEFT_WRIST)
    if a_rw and b_lw:
        min_dist = min(min_dist, _dist(a_rw, b_lw))

    a_lw = _kp_pos(kp_a_xy, kp_a_conf, KP_LEFT_WRIST)
    b_rw = _kp_pos(kp_b_xy, kp_b_conf, KP_RIGHT_WRIST)
    if a_lw and b_rw:
        min_dist = min(min_dist, _dist(a_lw, b_rw))

    return min_dist


# -- Neighbor finding (row-filtered, both directions) -------------
def find_row_neighbors(source_tid, all_student_bboxes, students):
    """Find all assigned students in the same row as source_tid."""
    if source_tid not in all_student_bboxes:
        return []
    sx1, sy1, sx2, sy2 = all_student_bboxes[source_tid]
    source_cy = (sy1 + sy2) / 2.0
    source_cx = (sx1 + sx2) / 2.0
    source_h = sy2 - sy1

    neighbors = []
    for tid, bbox in all_student_bboxes.items():
        if tid == source_tid or tid not in students:
            continue
        nx1, ny1, nx2, ny2 = bbox
        neighbor_cy = (ny1 + ny2) / 2.0
        neighbor_cx = (nx1 + nx2) / 2.0
        neighbor_h = ny2 - ny1
        # Scale row tolerance by average person size (perspective adaptation)
        avg_h = (source_h + neighbor_h) / 2.0
        scaled_row_tol = ROW_TOLERANCE_PX * _perspective_scale(avg_h)
        if abs(neighbor_cy - source_cy) > scaled_row_tol:
            continue
        direction = "LEFT" if neighbor_cx < source_cx else "RIGHT"
        neighbors.append((tid, direction))

    neighbors.sort(key=lambda x: abs(
        (all_student_bboxes[x[0]][0] + all_student_bboxes[x[0]][2]) / 2 - source_cx))
    return neighbors


# ═══════════════════════════════════════════════════════════════
#  FLASK WEB SERVER
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Passing Papers Detection (Pi + Hailo)</title>
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
    <h1>AISENTINEL - Passing Papers Detection</h1>
    <p class="info">Raspberry Pi 5 + Hailo AI HAT | Multi-Signal Interaction Detection</p>
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
        np.array of shape (N, 2) with polygon vertices, or None to skip.
    """
    fh, fw = frame.shape[:2]
    win_name = "AISENTINEL - Draw ROI Boundary"
    vertices = []

    def on_mouse(event, mx, my, flags, param):
        # Convert display coords back to frame coords
        ox = int(mx / disp_scale)
        oy = int(my / disp_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            vertices.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(vertices) >= 3:
                # Signal close by appending a sentinel; handled in loop
                vertices.append("CLOSE")

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    instructions = [
        "LEFT-CLICK: add vertex | RIGHT-CLICK: close polygon (>= 3 pts)",
        "BACKSPACE: undo | S: skip ROI (use full frame) | ESC: quit",
    ]

    closed = False

    while True:
        # Check for close sentinel
        if vertices and vertices[-1] == "CLOSE":
            vertices.pop()
            if len(vertices) >= 3:
                closed = True

        display = frame.copy()

        # Draw existing edges
        for i in range(len(vertices)):
            cv2.circle(display, vertices[i], 5, (0, 255, 255), -1, cv2.LINE_AA)
            if i > 0:
                cv2.line(display, vertices[i - 1], vertices[i],
                         (0, 255, 255), 2, cv2.LINE_AA)

        if closed and len(vertices) >= 3:
            # Draw closed polygon
            pts = np.array(vertices, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0, 40))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
        elif len(vertices) >= 2:
            # Draw lines between vertices (not yet closed)
            pass  # already drawn above

        # Draw instructions
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
            # Skip ROI
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
#  ASSIGNMENT PHASE (OpenCV window on local display)
# ═══════════════════════════════════════════════════════════════

def run_assignment_phase(first_frame, detections, track_ids, disp_scale):
    """Interactive student number assignment on the first frame.

    Returns student_map: {track_id: student_number} or None if cancelled.
    """
    if not detections:
        log_info("No persons detected in the first frame.")
        log_info("Press any key to proceed without assignments (or ESC to quit).")
        cv2.imshow("AISENTINEL - Assign Students", first_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AISENTINEL - Assign Students")
        if key == 27:
            return None
        return {}

    persons = []
    for i, det in enumerate(detections):
        kp = det['keypoints']  # (17, 3)
        persons.append({
            "track_id": track_ids[i],
            "bbox": tuple(det['bbox']),
            "kp_xy": kp[:, :2],
            "kp_conf": kp[:, 2],
        })

    student_map = {}
    selected_idx = -1
    input_buffer = ""

    fh, fw = first_frame.shape[:2]
    win_name = "AISENTINEL - Assign Students (Passing Papers)"

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
        "Press S to START detection (need >= 2) | ESC to quit",
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

            # Draw wrist keypoints prominently
            for kp_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
                if kp_idx < len(p["kp_conf"]) and p["kp_conf"][kp_idx] > KP_CONF_THRESH:
                    wx, wy = int(p["kp_xy"][kp_idx][0]), int(p["kp_xy"][kp_idx][1])
                    cv2.circle(display, (wx, wy), 6, COL_WRIST, -1, cv2.LINE_AA)
                    cv2.circle(display, (wx, wy), 8, COL_WRIST, 1, cv2.LINE_AA)

            if tid in student_map:
                label = f"Student #{student_map[tid]}"
                draw_label(display, label, x1, y1 - 2, COL_NORMAL)
            else:
                label = f"[unassigned] (ID:{tid})"
                draw_label(display, label, x1, y1 - 2, COL_UNASSIGNED)

            if i == selected_idx and input_buffer:
                draw_label(display, f"Typing: {input_buffer}_",
                           x1, y2 + 18, COL_SELECTED, (0, 0, 0))
            elif i == selected_idx:
                draw_label(display, "Selected - type a number",
                           x1, y2 + 18, COL_SELECTED, (0, 0, 0))

        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        assigned = len(student_map)
        total = len(persons)
        status = f"Assigned: {assigned}/{total} persons"
        cv2.putText(display, status, (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_NORMAL if assigned >= 2 else COL_UNASSIGNED, 2)

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
            if len(student_map) < 2:
                log_info("Assign at least 2 students for passing papers detection.")
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

def run_detection(cap, estimator, tracker, student_map, video_path, port,
                  roi_polygon=None):
    """Run detection loop, streaming annotated frames via Flask."""
    global _latest_frame

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print()
    print("=" * 70)
    print(f"  AISENTINEL - Passing Papers Detection (Pi + Hailo)")
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Students : {len(student_map)} assigned")
    print(f"  Wrist proximity     : {WRIST_PROXIMITY_PX}px (at ref scale)")
    print(f"  Min proximity dur   : {MIN_INTERACTION_SEC}s")
    print(f"  Row tolerance       : {ROW_TOLERANCE_PX}px (perspective-scaled)")
    print(f"  Consecutive nums only: Yes")
    print(f"  Reference bbox h    : {REFERENCE_BBOX_HEIGHT}px")
    print(f"  Cooldown            : {EVENT_COOLDOWN_SEC}s between repeated flags")
    print(f"  ROI                 : {'Yes (' + str(len(roi_polygon)) + ' vertices)' if roi_polygon is not None else 'No (full frame)'}")
    print(f"  Evidence dir        : {EVIDENCE_DIR}")
    local_ip = get_local_ip()
    print(f"  Web stream          : http://{local_ip}:{port}")
    print("=" * 70)
    print()

    # Build student states
    students: dict[int, StudentState] = {}
    for tid, num in student_map.items():
        students[tid] = StudentState(track_id=tid, student_num=num)

    assigned_tids = set(student_map.keys())

    # Per-pair interaction state, keyed by frozenset(tid_a, tid_b)
    pair_states: dict[frozenset, PairInteractionState] = {}

    def get_pair_state(tid_a, tid_b):
        key = frozenset((tid_a, tid_b))
        if key not in pair_states:
            pair_states[key] = PairInteractionState(tid_a=tid_a, tid_b=tid_b)
        return pair_states[key]

    # Per-student keypoint data for current frame
    frame_kp_data = {}  # tid -> (kp_xy, kp_conf)

    frame_idx = 1
    total_alerts = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_info("End of video reached.")
                break
            frame_idx += 1
            ts_sec = frame_idx / fps

            # ── Pose inference on Hailo ─────────────────────────
            t0 = time.perf_counter()
            detections = estimator.detect_pose(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # ── Filter by ROI ─────────────────────────────────────
            detections = filter_detections_by_roi(detections, roi_polygon)

            # ── Track only assigned students ──────────────────────
            track_ids = tracker.update(detections)

            annotated = frame.copy()

            # Draw ROI boundary on stream
            if roi_polygon is not None:
                cv2.polylines(annotated, [roi_polygon], isClosed=True,
                              color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            frame_events = []

            # Collect all tracked student bboxes for neighbor lookup
            all_student_bboxes = {}
            frame_kp_data.clear()
            per_student_labels = {}

            # ---- First pass: collect data + compute per-student signals ----
            for i, det in enumerate(detections):
                tid = track_ids[i]
                if tid == -1 or tid not in students:
                    # Draw unassigned persons faintly
                    bbox = det['bbox']
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_UNASSIGNED, 1)
                    draw_label(annotated, f"ID:{tid}", x1, y1, COL_UNASSIGNED)
                    continue

                bbox = tuple(det['bbox'])
                kp = det['keypoints']  # (17, 3)
                kp_xy = kp[:, :2]
                kp_conf = kp[:, 2]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                all_student_bboxes[tid] = bbox
                frame_kp_data[tid] = (kp_xy, kp_conf)
                state = students[tid]
                per_student_labels[tid] = [COL_NORMAL, []]

                # Draw skeleton and wrist keypoints
                draw_skeleton(annotated, kp_xy, kp_conf)
                for kp_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
                    if kp_idx < len(kp_conf) and kp_conf[kp_idx] > KP_CONF_THRESH:
                        wx, wy = int(kp_xy[kp_idx][0]), int(kp_xy[kp_idx][1])
                        cv2.circle(annotated, (wx, wy), 5, COL_WRIST, -1, cv2.LINE_AA)

            # ---- Second pass: evaluate pair interactions ----
            evaluated_pairs = set()

            for tid_a in list(frame_kp_data.keys()):
                neighbors = find_row_neighbors(tid_a, all_student_bboxes, students)

                for tid_b, direction in neighbors:
                    pair_key = frozenset((tid_a, tid_b))
                    if pair_key in evaluated_pairs:
                        continue
                    evaluated_pairs.add(pair_key)

                    if tid_b not in frame_kp_data:
                        continue

                    state_a = students[tid_a]
                    state_b = students[tid_b]

                    # Only consecutive student numbers can trigger passing
                    if abs(state_a.student_num - state_b.student_num) != 1:
                        continue

                    ps = get_pair_state(tid_a, tid_b)
                    kp_a_xy, kp_a_conf = frame_kp_data[tid_a]
                    kp_b_xy, kp_b_conf = frame_kp_data[tid_b]

                    bbox_b = all_student_bboxes[tid_b]
                    bbox_a = all_student_bboxes[tid_a]
                    center_b = ((bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2)
                    center_a = ((bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2)

                    # Perspective-adaptive threshold based on average bbox height
                    avg_pair_h = ((bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])) / 2.0
                    pscale = _perspective_scale(avg_pair_h)
                    scaled_prox_px = WRIST_PROXIMITY_PX * pscale

                    # Reset per-frame state
                    ps.frame_proximity = False
                    ps.frame_proximity_dist = 9999.0

                    # -- Wrist-to-wrist proximity --
                    prox_dist = compute_wrist_proximity(
                        kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf)
                    ps.frame_proximity_dist = prox_dist
                    ps.frame_proximity = prox_dist < scaled_prox_px

                    # Draw proximity line between closest wrists
                    if prox_dist < scaled_prox_px * 1.5:
                        best_pts = None
                        best_d = 9999.0
                        for (wa, wb) in [
                            (_kp_pos(kp_a_xy, kp_a_conf, KP_RIGHT_WRIST),
                             _kp_pos(kp_b_xy, kp_b_conf, KP_LEFT_WRIST)),
                            (_kp_pos(kp_a_xy, kp_a_conf, KP_LEFT_WRIST),
                             _kp_pos(kp_b_xy, kp_b_conf, KP_RIGHT_WRIST)),
                        ]:
                            if wa and wb:
                                d = _dist(wa, wb)
                                if d < best_d:
                                    best_d = d
                                    best_pts = (wa, wb)

                        if best_pts:
                            pt_a, pt_b = best_pts
                            line_col = COL_FLAGGED if ps.frame_proximity else COL_WARNING
                            cv2.line(annotated,
                                     (int(pt_a[0]), int(pt_a[1])),
                                     (int(pt_b[0]), int(pt_b[1])),
                                     line_col, 2, cv2.LINE_AA)
                            mid_x = int((pt_a[0] + pt_b[0]) / 2)
                            mid_y = int((pt_a[1] + pt_b[1]) / 2)
                            draw_label(annotated, f"{prox_dist:.0f}px",
                                       mid_x, mid_y, line_col)

                    # -- Temporal proximity tracking --
                    if ps.frame_proximity:
                        if ps.interaction_start < 0:
                            ps.interaction_start = ts_sec
                        ps.last_proximity_time = ts_sec
                        ps.peak_proximity_dist = min(
                            ps.peak_proximity_dist, prox_dist)

                        interaction_dur = ts_sec - ps.interaction_start

                        if center_b[0] > center_a[0]:
                            alert_dir = "RIGHT"
                        else:
                            alert_dir = "LEFT"

                        if tid_a in per_student_labels:
                            per_student_labels[tid_a][0] = COL_WARNING
                            per_student_labels[tid_a][1].append(
                                f"PROX S#{state_b.student_num} {prox_dist:.0f}px {interaction_dur:.1f}s")
                        if tid_b in per_student_labels:
                            per_student_labels[tid_b][0] = COL_WARNING
                            per_student_labels[tid_b][1].append(
                                f"PROX S#{state_a.student_num} {prox_dist:.0f}px {interaction_dur:.1f}s")

                        # Draw connection line between student centers
                        cv2.line(annotated,
                                 (int(center_a[0]), int(center_a[1])),
                                 (int(center_b[0]), int(center_b[1])),
                                 COL_NEIGHBOR_LINE, 2, cv2.LINE_AA)

                        # -- Fire alert when proximity sustained >= MIN_INTERACTION_SEC --
                        if (interaction_dur >= MIN_INTERACTION_SEC
                                and ps.can_flag(ts_sec)):

                            ps.last_flagged_at = ts_sec
                            state_a.total_alerts += 1
                            state_b.total_alerts += 1
                            total_alerts += 1

                            detail = (
                                f"{alert_dir}, "
                                f"dur={interaction_dur:.1f}s, "
                                f"closest={ps.peak_proximity_dist:.0f}px"
                            )
                            log_alert(
                                "PASSING PAPERS",
                                [state_a.student_num, state_b.student_num],
                                ts_sec, detail, TC.RED,
                            )
                            frame_events.append((
                                state_a.student_num,
                                state_b.student_num,
                                alert_dir,
                            ))

                            if tid_a in per_student_labels:
                                per_student_labels[tid_a][0] = COL_FLAGGED
                            if tid_b in per_student_labels:
                                per_student_labels[tid_b][0] = COL_FLAGGED

                            # Reset so the next proximity period starts fresh
                            ps.reset_interaction()

                    else:
                        # Proximity dropped — reset interaction timer
                        if ps.interaction_start > 0:
                            ps.reset_interaction()


            # ---- Third pass: draw person boxes + labels ----
            for i, det in enumerate(detections):
                tid = track_ids[i]
                if tid == -1 or tid not in students:
                    continue
                bbox = det['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                state = students[tid]

                box_color, behavior_labels = per_student_labels.get(
                    tid, (COL_NORMAL, []))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                draw_label(annotated, f"Student #{state.student_num}",
                           x1, y1 - 2, box_color)

                lbl_y = y1 + 18
                for bl in behavior_labels:
                    draw_label(annotated, bl, x1, lbl_y, box_color)
                    lbl_y += 18

            # -- HUD --------------------------------------------------
            n_tracked = sum(1 for t in track_ids if t in students)
            hud_lines = [
                f"Frame: {frame_idx}/{total_frames} | Time: {fmt_ts(ts_sec)}",
                f"Tracked: {n_tracked} | Assigned: {len(students)} | "
                f"Alerts: {total_alerts} | Inf: {inference_ms:.0f}ms",
            ]
            for i, line in enumerate(hud_lines):
                y_pos = 25 + i * 28
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COL_FLAGGED if total_alerts else COL_HUD,
                            2, cv2.LINE_AA)

            # -- Alert banner -----------------------------------------
            if frame_events:
                banner_y = h - 40
                for src_num, nbr_num, direction in frame_events:
                    txt = f"ALERT: S#{src_num} & S#{nbr_num} PASSING PAPERS ({direction})"
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_FLAGGED, 2, cv2.LINE_AA)
                    banner_y -= 35

            # -- Timestamp watermark ----------------------------------
            ts_text = fmt_ts(ts_sec)
            (tw, th_), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # -- Save evidence ----------------------------------------
            for src_num, nbr_num, direction in frame_events:
                save_evidence(annotated, [src_num, nbr_num], ts_sec)

            # -- Push to web stream -----------------------------------
            with _frame_lock:
                _latest_frame = annotated

            # -- Progress ---------------------------------------------
            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")

    except KeyboardInterrupt:
        log_info("Interrupted by user.")

    # -- Summary --------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Summary: {Path(video_path).name}")
    print("-" * 70)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Students tracked : {len(students)}")
    print(f"  Total alerts     : {total_alerts}")
    for tid, state in sorted(students.items(), key=lambda x: x[1].student_num):
        if state.total_alerts > 0:
            print(f"    Student #{state.student_num:2d} : {state.total_alerts} passing events")
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print(f"  No passing papers detected.")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Passing Papers Detection (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_passing_papers_pi.py
  python3 front_node_passing_papers_pi.py --model /path/to/yolov8s_pose.hef
  python3 front_node_passing_papers_pi.py --port 9090
        """,
    )
    parser.add_argument("--model", default=str(POSE_MODEL_PATH),
                        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080,
                        help="Flask web server port (default: 8080)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Person detection confidence (default: 0.5)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AISENTINEL - Passing Papers Detection (Pi + Hailo)")
    print("  Detects: Multi-signal hand interaction between neighbors")
    print(f"  Signals: arm extension + wrist velocity + wrist proximity")
    print("=" * 60)
    print()

    # ── Validate Hailo ──────────────────────────────────────
    if not HAILO_AVAILABLE:
        print(f"{TC.RED}[ERROR] hailo_platform is required.{TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"{TC.RED}[ERROR] HEF model not found: {model_path}{TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
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

    # ── Load Hailo pose estimator ───────────────────────────
    estimator = HailoPoseEstimator(
        str(model_path),
        conf_threshold=args.confidence,
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

    # ── ROI calibration ────────────────────────────────────
    log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = calibrate_roi(first_frame, disp_scale)
    if roi_result is not None and isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        log_info("Cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    log_info("Running pose detection on first frame for student assignment...")

    # ── Detect persons on first frame ───────────────────────
    first_detections = estimator.detect_pose(first_frame)
    first_detections = filter_detections_by_roi(first_detections, roi_polygon)

    # ── Create tracker and assign initial IDs ───────────────
    tracker = IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)

    log_info(f"Detected {len(first_detections)} persons (within ROI).")
    print()
    print(f"  {TC.BOLD}Instructions:{TC.RESET}")
    print(f"    1. Click on a person to select them (cyan highlight)")
    print(f"    2. Type the student number (digits)")
    print(f"    3. Press ENTER to assign")
    print(f"    4. Repeat for each student (need at least 2 for neighbor detection)")
    print(f"    5. Press S to start detection")
    print()

    # ── Assignment phase (local OpenCV window) ──────────────
    student_map = run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) < 2:
        cap.release()
        log_info("Need at least 2 students for passing papers detection. Exiting.")
        sys.exit(0)

    # ── Lock tracker to only assigned students ────────────────
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
    run_detection(cap, estimator, tracker, student_map, video_path, args.port,
                  roi_polygon=roi_polygon)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
