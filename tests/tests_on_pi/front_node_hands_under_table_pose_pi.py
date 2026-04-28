#!/usr/bin/env python3
"""
Hands Under Table Detection (Pose Estimation) - Raspberry Pi + Hailo AI HAT
============================================================================
Pose-based alternative to front_node_hands_under_table_pi.py.

Instead of a custom object detection model with 'student' and 'hand' classes,
this version uses a YOLO pose estimation model and monitors wrist keypoints
(KP_LEFT_WRIST=9, KP_RIGHT_WRIST=10) relative to desk polygon ROIs.

Advantages over the object-detection approach:
  - Wrist keypoints are inherently tied to the person — no hand-association
    heuristic needed (eliminates misassignment between nearby students)
  - Explicit left/right wrist tracking
  - Low keypoint confidence = occlusion signal (hands likely hidden)
  - Shares the same model used by head_behavior and passing_papers detectors,
    so one inference pass can power multiple behaviors

Algorithm:
  1. Run YOLO pose estimation on Hailo NPU
  2. IoU-track persons for persistent IDs across frames
  3. For each assigned student at their desk: check if wrist keypoints are
     confident AND located inside the desk polygon
  4. Majority-vote over a sliding window to smooth missed detections
  5. Flag only when wrists missing >= sustained threshold AND majority confirms

Workflow:
  1. File dialog opens to select a video
  2. ROI calibration: draw a polygon boundary on the first frame
     (only persons inside this region will be tracked)
  3. First frame shown with detected persons — click to assign student numbers
  4. Desk ROI calibration: draw polygon ROIs for each desk
  5. Web stream starts at http://<pi-ip>:8080 with live annotations
  6. Console alerts + evidence screenshots saved to ./evidence_hands/

ROI Drawing Controls (local OpenCV window):
    Left-click      — add vertex
    Right-click     — close polygon (>= 3 pts)
    BACKSPACE       — undo last vertex
    S               — skip (no ROI, track entire frame)
    ESC             — cancel

Student Assignment Controls:
    Left-click on person  -> select person (highlighted in cyan)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start (need >= 1 assignment)
    ESC                   -> quit

Desk ROI Drawing Controls:
    Left-click      — place polygon vertex
    Right-click     — close current polygon (finish desk)
    Z               — undo last vertex
    C               — clear all
    ENTER / SPACE   — confirm & start processing
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

from front_node_pi_model_paths import POSE_MODEL_PATH
import front_node_pi_interactive as pi_ui

EVIDENCE_DIR = SCRIPT_DIR / "evidence_hands"

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
HANDS_MISSING_SUSTAIN_SEC = 2.5     # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0           # cooldown between repeated flags
KP_CONF_THRESH = 0.8               # wrist keypoint confidence threshold
                                    # (higher than default 0.3 to reduce hallucinated positions)
MIN_WRISTS_PRESENT = 2             # how many wrists must be in desk ROI to count as "present"
                                    # set to 2 to require both hands visible on desk

# ── Tracking & Smoothing ────────────────────────────────────
SMOOTH_WINDOW_FRAMES = 12           # sliding window size for majority vote
SMOOTH_MISSING_RATIO = 0.6          # fraction of window that must be "missing" to confirm
STUDENT_ABSENT_RESET_SEC = 2.0      # reset desk if student undetected for this long

# ── Colors (BGR) ─────────────────────────────────────────────
COL_STUDENT = (0, 255, 0)           # green
COL_UNASSIGNED = (128, 128, 128)    # gray — unassigned person
COL_SELECTED = (255, 255, 0)        # cyan — selected person
COL_WRIST = (255, 0, 255)           # magenta — wrist keypoints
COL_WRIST_MISSING = (0, 0, 200)     # dark red — wrist below confidence
COL_DESK_ROI = (255, 0, 0)          # blue — desk polygon
COL_DESK_FILL = (255, 0, 0)         # blue — translucent fill
COL_ALERT = (0, 0, 255)             # red
COL_WARNING = (0, 165, 255)         # orange
COL_HUD = (0, 255, 0)               # green

# ── Skeleton for drawing ────────────────────────────────────
SKELETON = [
    (KP_NOSE, 1), (KP_NOSE, 2),
    (1, KP_LEFT_EAR), (2, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW), (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP), (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
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


def log_alert(student_id: int, desk_idx: int, ts_sec: float, detail: str = ""):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.RED}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{TC.RED}HANDS UNDER TABLE - Student #{student_id} at Desk #{desk_idx + 1}{TC.RESET}"
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


def save_evidence(annotated_frame, raw_frame, video_name, desk_idx, student_id, ts_sec):
    """Save both annotated and raw evidence frames."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")

    fname_ann = f"{video_name}_desk{desk_idx + 1}_sid{student_id}_{ts_str}_annotated.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname_ann), annotated_frame)

    fname_raw = f"{video_name}_desk{desk_idx + 1}_sid{student_id}_{ts_str}_raw.jpg"
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
#  HAILO POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════

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

        Args:
            detections: list of dicts with 'bbox' key ([x1,y1,x2,y2])

        Returns:
            list of track_ids aligned with detections (same length/order).
            Unmatched detections get track_id = -1 when locked.
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
#  DESK ROI CALIBRATION (Polygon)
# ══════════════════════════════════════════════════════════════

def calibrate_desk_rois(frame):
    """
    Interactive polygon ROI drawing on the first frame.
    User clicks vertices to define each desk polygon, right-clicks to
    close a polygon, and presses ENTER/SPACE to confirm all.

    Returns list of polygons, each polygon is a numpy array of shape (N, 2).
    Returns None if cancelled.
    """
    fh, fw = frame.shape[:2]
    scale = min(1.0, 1280 / fw)

    polygons = []
    current_points = []

    def on_mouse(event, mx, my, flags, param):
        ox = int(mx / scale)
        oy = int(my / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_points) >= 3:
                polygons.append(np.array(current_points, dtype=np.int32))
                current_points.clear()

    win = "AISENTINEL - Desk ROI Calibration (Pose)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    instructions = [
        "Left-click: place vertex | Right-click: close polygon",
        "Z: undo | C: clear | ENTER/SPACE: confirm | ESC: cancel",
    ]

    while True:
        display = frame.copy()

        overlay = display.copy()
        for i, poly in enumerate(polygons):
            cv2.fillPoly(overlay, [poly], COL_DESK_FILL)
            cv2.polylines(display, [poly], True, COL_DESK_ROI, 2, cv2.LINE_AA)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(display, f"Desk {i + 1}", (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_DESK_ROI, 2)
        cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        if len(current_points) > 0:
            for j in range(len(current_points) - 1):
                cv2.line(display, current_points[j], current_points[j + 1],
                         (0, 255, 0), 2, cv2.LINE_AA)
            for pt in current_points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
            last = current_points[-1]
            cv2.putText(display, f"{len(current_points)} pts (right-click to close)",
                        (last[0] + 10, last[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display, f"Desks defined: {len(polygons)}", (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if scale < 1.0:
            show = cv2.resize(display, (int(fw * scale), int(fh * scale)))
        else:
            show = display

        cv2.imshow(win, show)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            if len(current_points) >= 3:
                polygons.append(np.array(current_points, dtype=np.int32))
                current_points.clear()
            if len(polygons) == 0:
                log_info("No desk ROIs drawn. Draw at least one or press ESC to cancel.")
                continue
            break
        elif key == ord("z"):
            if current_points:
                current_points.pop()
            elif polygons:
                polygons.pop()
        elif key == ord("c"):
            polygons.clear()
            current_points.clear()
        elif key == 27:  # ESC
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    log_info(f"Calibration complete: {len(polygons)} desk ROIs defined.")
    return polygons


# ══════════════════════════════════════════════════════════════
#  ROI CALIBRATION (tracking area boundary)
# ══════════════════════════════════════════════════════════════

def calibrate_roi(frame, disp_scale):
    """Let the user draw a polygon ROI on the first frame.

    Controls:
        Left-click   -> add vertex
        Right-click  -> close polygon (need >= 3 vertices)
        BACKSPACE    -> undo last vertex
        S            -> skip (no ROI, track entire frame)
        ESC          -> cancel / quit

    Returns:
        np.array of shape (N, 2) with polygon vertices, or None to skip,
        or "CANCEL" string if ESC pressed.
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
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
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
    """Filter detections to only those whose bbox center is inside the ROI."""
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


# ══════════════════════════════════════════════════════════════
#  ASSIGNMENT PHASE (OpenCV window on local display)
# ══════════════════════════════════════════════════════════════

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
                color = COL_SELECTED
                thickness = 3
            elif tid in student_map:
                color = COL_STUDENT
                thickness = 2
            else:
                color = COL_UNASSIGNED
                thickness = 2

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            draw_skeleton(display, p["kp_xy"], p["kp_conf"],
                          color=COL_SELECTED if i == selected_idx else (255, 255, 0))

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


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def point_in_polygon(px, py, polygon):
    """Check if a point is inside the polygon."""
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


def bbox_polygon_intersection_area(bbox, polygon, img_shape):
    """Compute intersection area between a bounding box and a polygon."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_w = x2 - x1
    roi_h = y2 - y1

    shifted_poly = polygon.copy()
    shifted_poly[:, 0] -= x1
    shifted_poly[:, 1] -= y1

    poly_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [shifted_poly], 255)

    return float(np.count_nonzero(poly_mask))


def find_desk_for_student(student_bbox, desk_polygons, img_shape):
    """Assign a student to the desk with the largest overlap."""
    best_idx = None
    best_area = 0.0

    for i, poly in enumerate(desk_polygons):
        area = bbox_polygon_intersection_area(student_bbox, poly, img_shape)
        if area > best_area:
            best_area = area
            best_idx = i

    return best_idx, best_area


# ══════════════════════════════════════════════════════════════
#  PER-DESK TRACKING STATE (with temporal smoothing)
# ══════════════════════════════════════════════════════════════

class DeskState:
    """Tracks the hands-missing state for a single desk."""

    def __init__(self, desk_idx: int):
        self.desk_idx = desk_idx

        self.assigned_student_id = None
        self.history = deque(maxlen=SMOOTH_WINDOW_FRAMES)
        self.hands_missing_start = -1.0
        self.last_flagged_at = -999.0
        self.total_alerts = 0
        self.last_student_seen_at = -1.0

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

    def reset_assignment(self):
        self.assigned_student_id = None
        self.last_student_seen_at = -1.0
        self.reset()


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_desk_rois(img, desk_polygons, desk_states):
    """Draw all desk ROI polygons with status coloring."""
    overlay = img.copy()
    for i, poly in enumerate(desk_polygons):
        state = desk_states[i]
        if state.hands_missing_start > 0:
            color = COL_WARNING
        else:
            color = COL_DESK_ROI

        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)

        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        label = f"Desk {i + 1}"
        if state.assigned_student_id is not None:
            label += f" [S#{state.assigned_student_id}]"
        cv2.putText(img, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)


def draw_wrist_status(img, wx, wy, confident, in_polygon):
    """Draw a wrist keypoint with status indication."""
    if confident and in_polygon:
        color = COL_WRIST         # magenta — visible and in desk
    elif confident:
        color = COL_WARNING       # orange — visible but outside desk
    else:
        color = COL_WRIST_MISSING  # dark red — low confidence / occluded
    cv2.circle(img, (int(wx), int(wy)), 6, color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(wx), int(wy)), 6, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  FLASK WEB SERVER
# ══════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Hands Under Table Detection (Pose)</title>
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
    <h1>AISENTINEL - Hands Under Table (Pose)</h1>
    <p class="info">Raspberry Pi 5 + Hailo AI HAT | Wrist Keypoint Tracking</p>
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
#  WRIST PRESENCE CHECK
# ══════════════════════════════════════════════════════════════

def check_wrists_in_desk(keypoints, desk_polygon):
    """Check how many wrists are confident AND inside the desk polygon.

    Args:
        keypoints: np.array(17, 3) — [x, y, confidence] per keypoint
        desk_polygon: np.array(N, 2) — polygon vertices

    Returns:
        (wrists_present, wrist_details) where wrist_details is a list of
        dicts with keys: side, x, y, confident, in_polygon
    """
    wrist_details = []

    for side, kp_idx in [("L", KP_LEFT_WRIST), ("R", KP_RIGHT_WRIST)]:
        wx, wy, wconf = keypoints[kp_idx]
        confident = wconf >= KP_CONF_THRESH
        in_poly = False
        if confident:
            in_poly = point_in_polygon(wx, wy, desk_polygon)

        wrist_details.append({
            "side": side,
            "x": float(wx),
            "y": float(wy),
            "conf": float(wconf),
            "confident": confident,
            "in_polygon": in_poly,
        })

    wrists_present = sum(1 for w in wrist_details if w["confident"] and w["in_polygon"])
    return wrists_present, wrist_details


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════

def run_detection(cap, estimator, tracker, student_map, desk_polygons,
                  video_path, port, roi_polygon=None):
    """Run detection loop, streaming annotated frames via Flask.

    Only assigned students (from student_map) are monitored.
    """
    global _latest_frame

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)
    duration = total_frames / fps if fps > 0 else 0

    assigned_tids = set(student_map.keys())
    desk_states = [DeskState(i) for i in range(len(desk_polygons))]

    print()
    print("=" * 60)
    local_ip = get_local_ip()
    print(f"  AISENTINEL - Hands Under Table Detection (Pose)")
    print(f"  Video        : {Path(video_path).name}")
    print(f"  Resolution   : {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames : {total_frames}")
    print(f"  Students     : {len(student_map)} assigned")
    print(f"  Desk ROIs    : {len(desk_polygons)}")
    print(f"  ROI          : {'Yes (' + str(len(roi_polygon)) + ' vertices)' if roi_polygon is not None else 'No (full frame)'}")
    print(f"  Wrist conf   : >= {KP_CONF_THRESH}")
    print(f"  Min wrists   : {MIN_WRISTS_PRESENT} in desk ROI to count as present")
    print(f"  Threshold    : hands missing for {HANDS_MISSING_SUSTAIN_SEC}s")
    print(f"  Cooldown     : {EVENT_COOLDOWN_SEC}s")
    print(f"  Smoothing    : {SMOOTH_WINDOW_FRAMES} frames, "
          f">= {SMOOTH_MISSING_RATIO:.0%} missing to confirm")
    print(f"  Evidence     : {EVIDENCE_DIR}")
    print(f"  Web stream   : http://{local_ip}:{port}")
    print("=" * 60)
    print()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    total_alerts = 0

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
            #  1. RUN POSE ESTIMATION — Hailo NPU inference
            # ──────────────────────────────────────────────────────
            t0 = time.perf_counter()
            persons = estimator.detect_pose(frame)
            persons = filter_detections_by_roi(persons, roi_polygon)
            inference_ms = (time.perf_counter() - t0) * 1000

            annotated = frame.copy()

            # Draw ROI boundary if set
            if roi_polygon is not None:
                cv2.polylines(annotated, [roi_polygon], True,
                              (0, 255, 255), 1, cv2.LINE_AA)

            # ──────────────────────────────────────────────────────
            #  2. IoU TRACK PERSONS for persistent IDs
            # ──────────────────────────────────────────────────────
            track_ids = tracker.update(persons)

            # Build per-track data only for assigned students
            student_tracks = {}   # track_id -> bbox
            student_kps = {}      # track_id -> keypoints (17, 3)

            for i, person in enumerate(persons):
                tid = track_ids[i]
                if tid == -1 or tid not in assigned_tids:
                    # Draw unassigned persons in gray
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_UNASSIGNED, 1)
                    continue

                student_tracks[tid] = person['bbox']
                student_kps[tid] = person['keypoints']

                # Draw assigned person bbox and skeleton
                x1, y1, x2, y2 = person['bbox']
                snum = student_map[tid]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_STUDENT, 2)
                draw_label(annotated, f"#{snum} {person['confidence']:.0%}",
                           x1, y1 - 2, COL_STUDENT)

                kp_xy = person['keypoints'][:, :2]
                kp_conf = person['keypoints'][:, 2]
                draw_skeleton(annotated, kp_xy, kp_conf)

            # ──────────────────────────────────────────────────────
            #  3. MATCH ASSIGNED STUDENTS TO DESKS (intersection area)
            # ──────────────────────────────────────────────────────
            student_to_best_desk = {}
            for track_id, s_bbox in student_tracks.items():
                desk_idx, area = find_desk_for_student(s_bbox, desk_polygons, img_shape)
                if desk_idx is not None and area > 0:
                    student_to_best_desk[track_id] = (desk_idx, area)

            desk_to_candidates = defaultdict(list)
            for track_id, (desk_idx, area) in student_to_best_desk.items():
                desk_to_candidates[desk_idx].append((track_id, area))

            desk_best_student = {}
            for desk_idx, candidates in desk_to_candidates.items():
                candidates.sort(key=lambda x: x[1], reverse=True)
                desk_best_student[desk_idx] = candidates[0][0]

            for i, state in enumerate(desk_states):
                best_sid = desk_best_student.get(i)

                if best_sid is not None:
                    if state.assigned_student_id is None:
                        state.assigned_student_id = best_sid
                        state.last_student_seen_at = ts_sec
                    elif state.assigned_student_id == best_sid:
                        state.last_student_seen_at = ts_sec
                    else:
                        state.assigned_student_id = best_sid
                        state.last_student_seen_at = ts_sec
                        state.reset()
                else:
                    if state.assigned_student_id is not None:
                        if state.assigned_student_id in student_tracks:
                            state.reset_assignment()
                        else:
                            elapsed_absent = ts_sec - state.last_student_seen_at
                            if state.last_student_seen_at > 0 and elapsed_absent > STUDENT_ABSENT_RESET_SEC:
                                state.reset_assignment()

            # ──────────────────────────────────────────────────────
            #  4. WRIST PRESENCE CHECK PER DESK
            # ──────────────────────────────────────────────────────
            frame_events = []

            for i, state in enumerate(desk_states):
                sid = state.assigned_student_id
                if sid is None or sid not in student_tracks:
                    state.push_observation(True)
                    continue

                kps = student_kps[sid]
                poly = desk_polygons[i]

                wrists_present, wrist_details = check_wrists_in_desk(kps, poly)

                # Draw wrist status indicators
                for wd in wrist_details:
                    draw_wrist_status(annotated, wd["x"], wd["y"],
                                      wd["confident"], wd["in_polygon"])

                hands_present = wrists_present >= MIN_WRISTS_PRESENT

                # ──────────────────────────────────────────────────
                #  5. TEMPORAL SMOOTHING (majority vote)
                # ──────────────────────────────────────────────────
                state.push_observation(hands_present)
                smoothed_missing = state.majority_says_missing()

                # ──────────────────────────────────────────────────
                #  6. SUSTAINED DETECTION with smoothed signal
                # ──────────────────────────────────────────────────
                if smoothed_missing:
                    if state.hands_missing_start < 0:
                        state.hands_missing_start = ts_sec
                    elapsed = ts_sec - state.hands_missing_start

                    if elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.can_flag(ts_sec):
                        state.last_flagged_at = ts_sec
                        state.total_alerts += 1
                        total_alerts += 1

                        # Build detail string with per-wrist info
                        wrist_info = ", ".join(
                            f"{wd['side']}: conf={wd['conf']:.2f} "
                            f"{'IN' if wd['in_polygon'] else 'OUT'}"
                            for wd in wrist_details
                        )
                        snum = student_map.get(sid, sid)
                        log_alert(snum, i, ts_sec,
                                  f"sustained {elapsed:.1f}s, smoothed missing "
                                  f"({sum(1 for v in state.history if not v)}"
                                  f"/{len(state.history)} frames) | {wrist_info}")
                        frame_events.append(i)
                else:
                    state.hands_missing_start = -1.0

            # ── Draw desk ROIs ────────────────────────────────────
            draw_desk_rois(annotated, desk_polygons, desk_states)

            # Draw alert indicators on flagged desks
            for i, state in enumerate(desk_states):
                if state.hands_missing_start > 0 and state.assigned_student_id is not None:
                    elapsed = ts_sec - state.hands_missing_start
                    poly = desk_polygons[i]
                    cx = int(poly[:, 0].mean())
                    cy = int(poly[:, 1].mean()) + 20

                    if elapsed >= HANDS_MISSING_SUSTAIN_SEC:
                        txt = f"ALERT! ({elapsed:.1f}s)"
                        color = COL_ALERT
                    else:
                        txt = f"Watching ({elapsed:.1f}s)"
                        color = COL_WARNING

                    cv2.putText(annotated, txt, (cx - 50, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(annotated, txt, (cx - 50, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                                cv2.LINE_AA)

            # ── Save evidence (annotated + raw) ───────────────────
            for desk_idx in frame_events:
                tid = desk_states[desk_idx].assigned_student_id or 0
                snum = student_map.get(tid, tid)
                save_evidence(annotated, raw_frame, video_name, desk_idx, snum, ts_sec)

            # ── HUD ───────────────────────────────────────────────
            ts_text = fmt_ts(ts_sec)
            hud1 = f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}"
            tracked_count = len(student_tracks)
            hud2 = (f"Monitored: {tracked_count}/{len(student_map)} | "
                    f"Alerts: {total_alerts} | Inf: {inference_ms:.0f}ms")

            cv2.putText(annotated, hud1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_HUD, 2, cv2.LINE_AA)

            hud_color = COL_ALERT if total_alerts > 0 else COL_HUD
            cv2.putText(annotated, hud2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)

            # Alert banner
            if frame_events:
                banner_y = h - 30
                for desk_idx in frame_events:
                    tid = desk_states[desk_idx].assigned_student_id or 0
                    snum = student_map.get(tid, tid)
                    txt = f"ALERT: Student #{snum} wrists missing from Desk #{desk_idx + 1}"
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4,
                                cv2.LINE_AA)
                    cv2.putText(annotated, txt, (10, banner_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_ALERT, 2,
                                cv2.LINE_AA)
                    banner_y -= 35

            # Timestamp watermark bottom-right
            (tw, th_), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # ── Push to web stream ────────────────────────────────
            with _frame_lock:
                _latest_frame = annotated

            # ── Progress ──────────────────────────────────────────
            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")

    except KeyboardInterrupt:
        log_info("Interrupted by user.")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Summary: {Path(video_path).name}")
    print("-" * 60)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Desk ROIs        : {len(desk_polygons)}")
    print(f"  Total alerts     : {total_alerts}")
    for i, state in enumerate(desk_states):
        if state.total_alerts > 0:
            tid = state.assigned_student_id or 0
            snum = student_map.get(tid, "?")
            print(f"    Desk #{i + 1:2d} (Student #{snum})  : {state.total_alerts} alerts")
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No hands-under-table detected.")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global KP_CONF_THRESH, MIN_WRISTS_PRESENT
    import argparse

    parser = argparse.ArgumentParser(
        description="AISENTINEL - Hands Under Table Detection via Pose (Pi + Hailo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_hands_under_table_pose_pi.py
  python3 front_node_hands_under_table_pose_pi.py --model /path/to/pose.hef
  python3 front_node_hands_under_table_pose_pi.py --port 9090
  python3 front_node_hands_under_table_pose_pi.py --min-wrists 2
        """,
    )
    parser.add_argument("--video", default=None,
                        help="Optional path to a video file")
    parser.add_argument("--model", default=None,
                        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})")
    parser.add_argument("--port", type=int, default=8080,
                        help="Flask web server port (default: 8080)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Person detection confidence threshold (default: 0.5)")
    parser.add_argument("--wrist-conf", type=float, default=KP_CONF_THRESH,
                        help=f"Wrist keypoint confidence threshold (default: {KP_CONF_THRESH})")
    parser.add_argument("--min-wrists", type=int, default=MIN_WRISTS_PRESENT,
                        choices=[1, 2],
                        help=f"Min wrists in desk ROI to count as present (default: {MIN_WRISTS_PRESENT})")
    args = parser.parse_args()

    KP_CONF_THRESH = args.wrist_conf
    MIN_WRISTS_PRESENT = args.min_wrists

    print()
    print("=" * 60)
    print("  AISENTINEL - Hands Under Table Detection (Pose)")
    print("  Uses wrist keypoints instead of hand object detection")
    print(f"  Wrist confidence   : >= {KP_CONF_THRESH}")
    print(f"  Min wrists in desk : {MIN_WRISTS_PRESENT}")
    print(f"  Sustained threshold: {HANDS_MISSING_SUSTAIN_SEC}s")
    print(f"  Smoothing window   : {SMOOTH_WINDOW_FRAMES} frames "
          f"(missing ratio >= {SMOOTH_MISSING_RATIO:.0%})")
    print("=" * 60)
    print()

    video_path = pi_ui.select_video(args.video, select_video_dialog)
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)

    model_arg = pi_ui.select_pose_model(args.model)
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
        print(f"{TC.RED}[ERROR] Pose HEF model not found: {model_path}{TC.RESET}")
        print("Use a YOLO pose model compiled for Hailo (e.g. yolo_pose_model.hef).")
        sys.exit(1)

    # ── Select video via file dialog ────────────────────────
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Load Hailo pose estimator ───────────────────────────
    estimator = HailoPoseEstimator(
        str(model_path),
        conf_threshold=args.confidence,
        kpt_threshold=KP_CONF_THRESH,
    )

    # ── Open video & read first frame for calibration ────────
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
    first_detections = estimator.detect_pose(first_frame)
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

    # ── Desk ROI calibration (local OpenCV window) ───────────
    log_info("Now draw polygon ROIs for each desk on the first frame.")
    desk_polygons = calibrate_desk_rois(first_frame)
    if desk_polygons is None or len(desk_polygons) == 0:
        cap.release()
        log_info("No desk ROIs defined. Exiting.")
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
    run_detection(cap, estimator, tracker, student_map, desk_polygons,
                  video_path, args.port, roi_polygon=roi_polygon)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
