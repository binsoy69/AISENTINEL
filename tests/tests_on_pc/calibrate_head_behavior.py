#!/usr/bin/env python3
"""
Head Behavior Threshold Calibration Tool
========================================

Video-based calibration for head-behavior thresholds. The default flow opens a
file picker, runs pose detection on the first frame, lets you click one student
to monitor, then shows live values while you tune the threshold sliders.

Displays for the selected student:
  - head_tilt_angle_deg: ear-to-ear roll angle
  - head_turn_ratio: baseline-corrected nose offset / shoulder width
  - shoulder_turn_angle_deg: shoulder-line deviation for overhead camera

Controls:
    Trackbars  - adjust head behavior thresholds and keypoint confidence
    SPACE      - pause / resume video
    Q / ESC    - quit and print final config-ready values

Requirements:
    pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from front_node_pc_common import POSE_MODEL_CANDIDATES, first_existing


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
POSE_MODEL_PATH = first_existing(POSE_MODEL_CANDIDATES) or Path("yolo11n-pose.pt")

# COCO 17-keypoint indices
KP_NOSE = 0
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# Colors (BGR)
COL_NORMAL = (0, 255, 0)
COL_UNASSIGNED = (128, 128, 128)
COL_SELECTED = (255, 255, 0)
COL_HEAD_TILT = (0, 165, 255)
COL_SHOULDER_TURN = (255, 191, 0)
COL_TRIGGERED = (0, 0, 255)
COL_GUIDE = (255, 255, 0)
COL_DIM = (100, 100, 100)

SKELETON = [
    (KP_NOSE, 1),
    (KP_NOSE, 2),
    (1, KP_LEFT_EAR),
    (2, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, 7),
    (KP_RIGHT_SHOULDER, 8),
    (7, 9),
    (8, 10),
    (KP_LEFT_SHOULDER, 11),
    (KP_RIGHT_SHOULDER, 12),
    (11, 12),
]

# Slider ranges/defaults
TILT_SLIDER_MAX = 90
TURN_SLIDER_MAX = 100
SHOULDER_SLIDER_MAX = 90
CONF_SLIDER_MAX = 100

TILT_DEFAULT = 30
TURN_DEFAULT = 26
SHOULDER_DEFAULT = 20
CONF_DEFAULT = 30

YAW_MIN_SHOULDER_WIDTH_PX = 20.0
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_LOST = 90
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720


def parse_args():
    parser = argparse.ArgumentParser(
        description="AISENTINEL - Head Behavior Threshold Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/tests_on_pc/calibrate_head_behavior.py
  python tests/tests_on_pc/calibrate_head_behavior.py --video path/to/exam.mp4
        """,
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional video path. If omitted, a file picker opens.",
    )
    return parser.parse_args()


def select_video_dialog():
    """Open a tkinter file picker for video selection."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[WARN] File picker unavailable: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select calibration video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def select_video_path(video_arg):
    if video_arg:
        return Path(video_arg)

    print("[INFO] Opening video file picker...")
    selected = select_video_dialog()
    if not selected:
        print("[INFO] No video selected.")
        return None
    return Path(selected)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _clip_box(box, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return [x1, y1, x2, y2]


def result_to_pose_detections(result, frame_shape):
    boxes = getattr(result, "boxes", None)
    keypoints_data = getattr(result, "keypoints", None)
    if boxes is None or keypoints_data is None or len(boxes) == 0:
        return []

    h, w = frame_shape[:2]
    bboxes = _to_numpy(boxes.xyxy)
    box_confs = _to_numpy(getattr(boxes, "conf", None))
    if box_confs is None:
        box_confs = np.ones(len(bboxes), dtype=np.float32)

    kps_xy = _to_numpy(getattr(keypoints_data, "xy", None))
    if kps_xy is None:
        return []

    kps_conf = _to_numpy(getattr(keypoints_data, "conf", None))
    if kps_conf is None:
        kp_data = _to_numpy(getattr(keypoints_data, "data", None))
        if kp_data is not None and kp_data.ndim == 3 and kp_data.shape[-1] >= 3:
            kps_conf = kp_data[:, :, 2]
        else:
            kps_conf = np.ones(kps_xy.shape[:2], dtype=np.float32)

    detections = []
    count = min(len(bboxes), len(kps_xy), len(kps_conf))
    for i in range(count):
        bbox = _clip_box(bboxes[i], w, h)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "bbox": bbox,
                "confidence": float(box_confs[i]) if i < len(box_confs) else 1.0,
                "kp_xy": kps_xy[i],
                "kp_conf": kps_conf[i],
            }
        )
    return detections


class IoUTracker:
    """Small frame-to-frame IoU tracker for keeping one selected student stable."""

    def __init__(self, iou_threshold=0.3, max_lost=90):
        self._next_id = 1
        self._tracks = {}
        self._locked = False
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def keep_only(self, track_ids_to_keep):
        keep = set(track_ids_to_keep)
        for tid in list(self._tracks):
            if tid not in keep:
                del self._tracks[tid]
        self._locked = True

    def update(self, detections):
        if not detections:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]
            return []

        det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)

        if not self._tracks:
            if self._locked:
                return [-1] * len(detections)

            ids = []
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": det["bbox"], "lost": 0}
                ids.append(tid)
            return ids

        track_ids = list(self._tracks.keys())
        track_boxes = np.array(
            [self._tracks[tid]["bbox"] for tid in track_ids],
            dtype=np.float32,
        )
        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        pairs = []
        for ti in range(len(track_ids)):
            for di in range(len(detections)):
                if iou_matrix[ti, di] > self.iou_threshold:
                    pairs.append((iou_matrix[ti, di], ti, di))
        pairs.sort(reverse=True)

        assigned_tracks = set()
        assigned_dets = set()
        matches = {}
        for _, ti, di in pairs:
            if ti in assigned_tracks or di in assigned_dets:
                continue
            matches[di] = track_ids[ti]
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        result_ids = []
        for di, det in enumerate(detections):
            if di in matches:
                tid = matches[di]
                self._tracks[tid]["bbox"] = det["bbox"]
                self._tracks[tid]["lost"] = 0
                result_ids.append(tid)
            elif self._locked:
                result_ids.append(-1)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": det["bbox"], "lost": 0}
                result_ids.append(tid)

        for ti, tid in enumerate(track_ids):
            if ti not in assigned_tracks and tid in self._tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]

        return result_ids

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        iou = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        for i, a in enumerate(boxes_a):
            ax1, ay1, ax2, ay2 = a
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j, b in enumerate(boxes_b):
                bx1, by1, bx2, by2 = b
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0.0
        return iou


def display_scale_for(frame, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return 1.0
    return min(1.0, max_width / w, max_height / h)


def resize_for_display(frame, scale):
    if scale >= 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_skeleton(img, kp_xy, kp_conf, conf_thresh, color=COL_GUIDE):
    for i, j in SKELETON:
        if i < len(kp_conf) and j < len(kp_conf):
            if kp_conf[i] > conf_thresh and kp_conf[j] > conf_thresh:
                p1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                p2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    for k in range(min(len(kp_xy), 13)):
        if kp_conf[k] > conf_thresh:
            cv2.circle(
                img,
                (int(kp_xy[k][0]), int(kp_xy[k][1])),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )


def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(
        img,
        text,
        (x + 2, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        fg,
        1,
        cv2.LINE_AA,
    )


def draw_gauge(img, x, y, value, threshold, label, max_val, color_normal, color_triggered):
    bar_w = 350
    bar_h = 18
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)

    fill = min(value / max_val, 1.0) if max_val > 0 else 0.0
    triggered = value > threshold
    color = color_triggered if triggered else color_normal
    cv2.rectangle(img, (x, y), (x + int(bar_w * fill), y + bar_h), color, -1)

    threshold = max(0.0, min(float(threshold), max_val))
    thresh_x = x + int((threshold / max_val) * bar_w) if max_val > 0 else x
    cv2.line(img, (thresh_x, y - 2), (thresh_x, y + bar_h + 2), (255, 255, 255), 2)

    status = "TRIGGERED" if triggered else "ok"
    text = f"{label}: {value:.2f} (thresh: {threshold:.2f}) [{status}]"
    cv2.putText(
        img,
        text,
        (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_triggered if triggered else (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def compute_signed_yaw(kp_xy, kp_conf, conf_thresh):
    if (
        kp_conf[KP_NOSE] >= conf_thresh
        and kp_conf[KP_LEFT_SHOULDER] >= conf_thresh
        and kp_conf[KP_RIGHT_SHOULDER] >= conf_thresh
    ):
        nose_x = float(kp_xy[KP_NOSE][0])
        ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
        rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
        shoulder_width = abs(rs_x - ls_x)
        if shoulder_width >= YAW_MIN_SHOULDER_WIDTH_PX:
            shoulder_center_x = (ls_x + rs_x) / 2.0
            return True, (nose_x - shoulder_center_x) / shoulder_width
    return False, 0.0


def measure_head_tilt(kp_xy, kp_conf, conf_thresh, baseline_yaw=0.0):
    """Return (valid, roll_angle_deg, baseline_corrected_yaw_ratio)."""
    roll_angle = 0.0
    yaw_ratio = 0.0
    has_roll = False
    has_yaw = False

    if kp_conf[KP_LEFT_EAR] >= conf_thresh and kp_conf[KP_RIGHT_EAR] >= conf_thresh:
        le = kp_xy[KP_LEFT_EAR]
        re = kp_xy[KP_RIGHT_EAR]
        raw = abs(
            math.degrees(
                math.atan2(
                    float(re[1]) - float(le[1]),
                    float(re[0]) - float(le[0]),
                )
            )
        )
        roll_angle = raw if raw <= 90 else 180 - raw
        has_roll = True

    yaw_valid, signed_yaw = compute_signed_yaw(kp_xy, kp_conf, conf_thresh)
    if yaw_valid:
        yaw_ratio = abs(signed_yaw - baseline_yaw)
        has_yaw = True

    return (has_roll or has_yaw), roll_angle, yaw_ratio


def measure_shoulder_turn(kp_xy, kp_conf, conf_thresh):
    if (
        kp_conf[KP_LEFT_SHOULDER] < conf_thresh
        or kp_conf[KP_RIGHT_SHOULDER] < conf_thresh
    ):
        return False, 0.0, ""

    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    ls_y = float(kp_xy[KP_LEFT_SHOULDER][1])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
    rs_y = float(kp_xy[KP_RIGHT_SHOULDER][1])

    shoulder_dist = math.hypot(rs_x - ls_x, rs_y - ls_y)
    if shoulder_dist < 10:
        return False, 0.0, ""

    raw_angle = math.degrees(math.atan2(rs_y - ls_y, rs_x - ls_x))
    angle = abs(raw_angle) if abs(raw_angle) <= 90 else 180 - abs(raw_angle)
    direction = "RIGHT" if raw_angle > 0 else "LEFT"
    return True, angle, direction


def select_student(first_frame, detections, track_ids, conf_thresh):
    if not detections:
        print("[ERROR] No students detected in the first video frame.")
        print("[ERROR] Choose a video where the target student is visible at the start.")
        return None, 0.0

    persons = []
    for det, tid in zip(detections, track_ids):
        persons.append(
            {
                "track_id": tid,
                "bbox": det["bbox"],
                "kp_xy": det["kp_xy"],
                "kp_conf": det["kp_conf"],
            }
        )

    selected_idx = -1
    scale = display_scale_for(first_frame)
    win = "AISENTINEL - Select Student"

    def on_mouse(event, mx, my, _flags, _param):
        nonlocal selected_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox = int(mx / scale)
        oy = int(my / scale)
        selected_idx = -1
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                selected_idx = i
                break

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    print()
    print("[INFO] Select one student in the first frame.")
    print("[INFO] Click a student, then press ENTER or S to start. ESC cancels.")
    print()

    while True:
        display = first_frame.copy()
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
            color = COL_SELECTED if i == selected_idx else COL_UNASSIGNED
            thickness = 3 if i == selected_idx else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            draw_skeleton(display, person["kp_xy"], person["kp_conf"], conf_thresh, color=color)
            label = "Selected student" if i == selected_idx else f"Person {i} ID:{person['track_id']}"
            draw_label(display, label, x1, y1 - 2, color, (0, 0, 0) if i == selected_idx else (255, 255, 255))

        instructions = [
            "Click one student to monitor",
            "ENTER/S start | ESC cancel",
        ]
        for i, text in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        status = "Selected: none" if selected_idx < 0 else f"Selected: Person {selected_idx}"
        cv2.putText(
            display,
            status,
            (10, first_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COL_SELECTED if selected_idx >= 0 else COL_UNASSIGNED,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win, resize_for_display(display, scale))
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(win)
            return None, 0.0
        if key in (13, ord("s"), ord("S")):
            if selected_idx < 0:
                print("[INFO] Click a student before starting.")
                continue
            person = persons[selected_idx]
            yaw_valid, baseline_yaw = compute_signed_yaw(
                person["kp_xy"],
                person["kp_conf"],
                conf_thresh,
            )
            if not yaw_valid:
                baseline_yaw = 0.0
                print("[INFO] Baseline yaw unavailable; using 0.00.")
            else:
                print(f"[INFO] Baseline yaw for selected student: {baseline_yaw:+.3f}")
            cv2.destroyWindow(win)
            return person["track_id"], baseline_yaw


def _noop(_value):
    pass


def read_thresholds(win):
    tilt_thresh = max(cv2.getTrackbarPos("Head Tilt (deg)", win), 1)
    turn_thresh_pct = cv2.getTrackbarPos("Head Turn (%)", win)
    shoulder_thresh = max(cv2.getTrackbarPos("Shoulder Turn (deg)", win), 1)
    conf_thresh_pct = cv2.getTrackbarPos("KP Conf (%)", win)
    return {
        "tilt": float(tilt_thresh),
        "turn_pct": float(turn_thresh_pct),
        "turn": float(turn_thresh_pct) / 100.0,
        "shoulder": float(shoulder_thresh),
        "conf": float(conf_thresh_pct) / 100.0,
    }


def draw_selected_pose_details(annotated, kp_xy, kp_conf, measurement, thresholds):
    conf_thresh = thresholds["conf"]

    if kp_conf[KP_LEFT_EAR] >= conf_thresh and kp_conf[KP_RIGHT_EAR] >= conf_thresh:
        le = (int(kp_xy[KP_LEFT_EAR][0]), int(kp_xy[KP_LEFT_EAR][1]))
        re = (int(kp_xy[KP_RIGHT_EAR][0]), int(kp_xy[KP_RIGHT_EAR][1]))
        color = COL_TRIGGERED if measurement["roll_triggered"] else COL_HEAD_TILT
        cv2.line(annotated, le, re, color, 2, cv2.LINE_AA)
        mid_ear = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2 - 10)
        cv2.putText(
            annotated,
            f"{measurement['roll_angle']:.1f} deg",
            mid_ear,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    yaw_valid, _signed_yaw = compute_signed_yaw(kp_xy, kp_conf, conf_thresh)
    if yaw_valid:
        nose_pt = (int(kp_xy[KP_NOSE][0]), int(kp_xy[KP_NOSE][1]))
        sc_x = int((kp_xy[KP_LEFT_SHOULDER][0] + kp_xy[KP_RIGHT_SHOULDER][0]) / 2)
        sc_y = int((kp_xy[KP_LEFT_SHOULDER][1] + kp_xy[KP_RIGHT_SHOULDER][1]) / 2)
        color = COL_TRIGGERED if measurement["yaw_triggered"] else COL_HEAD_TILT
        cv2.line(annotated, nose_pt, (sc_x, sc_y), color, 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{measurement['yaw_ratio']:.2f}",
            (nose_pt[0] + 5, nose_pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if measurement["shoulder_valid"]:
        ls_pt = (int(kp_xy[KP_LEFT_SHOULDER][0]), int(kp_xy[KP_LEFT_SHOULDER][1]))
        rs_pt = (int(kp_xy[KP_RIGHT_SHOULDER][0]), int(kp_xy[KP_RIGHT_SHOULDER][1]))
        color = COL_TRIGGERED if measurement["shoulder_triggered"] else COL_SHOULDER_TURN
        cv2.line(annotated, ls_pt, rs_pt, color, 3, cv2.LINE_AA)
        mid_s = ((ls_pt[0] + rs_pt[0]) // 2, (ls_pt[1] + rs_pt[1]) // 2 - 12)
        cv2.putText(
            annotated,
            f"{measurement['shoulder_angle']:.1f} deg ({measurement['shoulder_dir']})",
            mid_s,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def build_measurement(det, thresholds, baseline_yaw):
    kp_xy = det["kp_xy"]
    kp_conf = det["kp_conf"]
    tilt_valid, roll_angle, yaw_ratio = measure_head_tilt(
        kp_xy,
        kp_conf,
        thresholds["conf"],
        baseline_yaw=baseline_yaw,
    )
    roll_triggered = roll_angle > thresholds["tilt"]
    yaw_triggered = yaw_ratio > thresholds["turn"]
    tilt_triggered = tilt_valid and (roll_triggered or yaw_triggered)

    shoulder_valid, shoulder_angle, shoulder_dir = measure_shoulder_turn(
        kp_xy,
        kp_conf,
        thresholds["conf"],
    )
    shoulder_triggered = shoulder_valid and shoulder_angle > thresholds["shoulder"]

    return {
        "tilt_valid": tilt_valid,
        "roll_angle": roll_angle,
        "roll_triggered": roll_triggered,
        "yaw_ratio": yaw_ratio,
        "yaw_triggered": yaw_triggered,
        "tilt_triggered": tilt_triggered,
        "shoulder_valid": shoulder_valid,
        "shoulder_angle": shoulder_angle,
        "shoulder_dir": shoulder_dir,
        "shoulder_triggered": shoulder_triggered,
    }


def draw_hud(annotated, selected_measurement, thresholds, selected_track_id, frame_idx, total_frames):
    h, w = annotated.shape[:2]
    panel_h = 230
    panel_w = 470
    panel_x = 10
    panel_y = max(10, h - panel_h - 10)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    frame_label = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"
    lines = [
        ("CURRENT THRESHOLDS:", (200, 200, 200)),
        (f"head_tilt_angle_deg      = {thresholds['tilt']:.1f}", COL_HEAD_TILT),
        (f"head_turn_ratio          = {thresholds['turn']:.2f}", COL_HEAD_TILT),
        (f"shoulder_turn_angle_deg  = {thresholds['shoulder']:.1f}", COL_SHOULDER_TURN),
        (f"keypoint_confidence      = {thresholds['conf']:.2f}", COL_GUIDE),
        (f"Selected track ID        = {selected_track_id}", COL_SELECTED),
        (frame_label, (220, 220, 220)),
    ]

    y = panel_y + 20
    for text, color in lines:
        cv2.putText(annotated, text, (panel_x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22

    if selected_measurement is None:
        cv2.putText(
            annotated,
            "Selected student: lost / not detected",
            (panel_x + 5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COL_DIM,
            1,
            cv2.LINE_AA,
        )
        return

    y += 4
    roll_color = COL_TRIGGERED if selected_measurement["roll_triggered"] else COL_NORMAL
    yaw_color = COL_TRIGGERED if selected_measurement["yaw_triggered"] else COL_NORMAL
    shoulder_color = COL_TRIGGERED if selected_measurement["shoulder_triggered"] else COL_NORMAL

    if selected_measurement["tilt_valid"]:
        live_lines = [
            (f"Current head_tilt_angle_deg: {selected_measurement['roll_angle']:.1f}", roll_color),
            (f"Current head_turn_ratio:     {selected_measurement['yaw_ratio']:.2f}", yaw_color),
        ]
    else:
        live_lines = [("Current head values: low confidence", COL_DIM)]

    if selected_measurement["shoulder_valid"]:
        live_lines.append(
            (
                f"Current shoulder_turn_angle_deg: {selected_measurement['shoulder_angle']:.1f} {selected_measurement['shoulder_dir']}",
                shoulder_color,
            )
        )
    else:
        live_lines.append(("Current shoulder_turn_angle_deg: low confidence", COL_DIM))

    for text, color in live_lines:
        cv2.putText(annotated, text, (panel_x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22


def draw_gauges(annotated, selected_measurement, thresholds):
    if selected_measurement is None:
        return

    _h, w = annotated.shape[:2]
    gauge_x = max(10, w - 380)
    if selected_measurement["tilt_valid"]:
        draw_gauge(
            annotated,
            gauge_x,
            30,
            selected_measurement["roll_angle"],
            thresholds["tilt"],
            "Head Tilt Deg",
            90.0,
            COL_HEAD_TILT,
            COL_TRIGGERED,
        )
        draw_gauge(
            annotated,
            gauge_x,
            75,
            selected_measurement["yaw_ratio"] * 100,
            thresholds["turn_pct"],
            "Head Turn %",
            100.0,
            COL_HEAD_TILT,
            COL_TRIGGERED,
        )
    if selected_measurement["shoulder_valid"]:
        draw_gauge(
            annotated,
            gauge_x,
            120,
            selected_measurement["shoulder_angle"],
            thresholds["shoulder"],
            "Shoulder Turn Deg",
            90.0,
            COL_SHOULDER_TURN,
            COL_TRIGGERED,
        )


def render_calibration_frame(
    frame,
    detections,
    track_ids,
    selected_track_id,
    baseline_yaw,
    thresholds,
    frame_idx,
    total_frames,
    paused,
):
    annotated = frame.copy()
    selected_measurement = None

    for index, (det, tid) in enumerate(zip(detections, track_ids)):
        kp_xy = det["kp_xy"]
        kp_conf = det["kp_conf"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        is_selected = tid == selected_track_id

        measurement = build_measurement(
            det,
            thresholds,
            baseline_yaw=baseline_yaw if is_selected else 0.0,
        )

        if is_selected:
            if measurement["tilt_triggered"] or measurement["shoulder_triggered"]:
                box_color = COL_TRIGGERED
            else:
                box_color = COL_SELECTED
            thickness = 3
            selected_measurement = measurement
        else:
            box_color = COL_UNASSIGNED
            thickness = 1

        draw_skeleton(annotated, kp_xy, kp_conf, thresholds["conf"], color=box_color)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        if is_selected:
            draw_selected_pose_details(annotated, kp_xy, kp_conf, measurement, thresholds)
            labels = ["Selected student"]
            if measurement["tilt_valid"]:
                labels.append(f"head_tilt: {measurement['roll_angle']:.1f}")
                labels.append(f"head_turn: {measurement['yaw_ratio']:.2f}")
            if measurement["shoulder_valid"]:
                labels.append(f"shoulder: {measurement['shoulder_angle']:.1f} {measurement['shoulder_dir']}")
        else:
            labels = [f"Person {index}" if tid < 0 else f"Person {index} ID:{tid}"]

        lbl_y = y1 - 5
        for label in labels:
            draw_label(
                annotated,
                label,
                x1,
                lbl_y,
                box_color,
                (0, 0, 0) if is_selected and box_color == COL_SELECTED else (255, 255, 255),
            )
            lbl_y -= 20

    draw_hud(annotated, selected_measurement, thresholds, selected_track_id, frame_idx, total_frames)
    draw_gauges(annotated, selected_measurement, thresholds)

    if paused:
        cv2.putText(
            annotated,
            "PAUSED (SPACE to resume)",
            (max(10, annotated.shape[1] // 2 - 150), 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COL_TRIGGERED,
            2,
            cv2.LINE_AA,
        )

    return annotated


def print_final_config(thresholds):
    print()
    print("=" * 60)
    print("  Final Calibrated Thresholds")
    print("=" * 60)
    print()
    print("  Copy this into config/front_node.ini or config/mid_node.ini:")
    print()
    print("[head_behavior]")
    print(f"head_tilt_angle_deg = {thresholds['tilt']:.1f}")
    print(f"head_turn_ratio = {thresholds['turn']:.2f}")
    print(f"shoulder_turn_angle_deg = {thresholds['shoulder']:.1f}")
    print(f"keypoint_confidence = {thresholds['conf']:.2f}")
    print()
    print("=" * 60)


def load_model():
    if not POSE_MODEL_PATH.exists():
        print(f"[INFO] Model not found at {POSE_MODEL_PATH}")
        print("[INFO] Ultralytics will auto-download on first use.")
    print(f"[INFO] Loading pose model: {POSE_MODEL_PATH}")
    model = YOLO(str(POSE_MODEL_PATH))
    print("[INFO] Model loaded.")
    return model


def main():
    args = parse_args()

    print()
    print("=" * 60)
    print("  AISENTINEL - Head Behavior Calibration")
    print("  Select a video, choose one student, tune thresholds")
    print("=" * 60)
    print()

    video_path = select_video_path(args.video)
    if video_path is None:
        return
    if not video_path.is_file():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    model = load_model()

    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print("[ERROR] Cannot read first frame from video.")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"[INFO] Video resolution: {frame_w}x{frame_h}")
    if total_frames > 0:
        print(f"[INFO] Video frames: {total_frames}")
    if fps > 0:
        print(f"[INFO] Video FPS: {fps:.2f}")

    print("[INFO] Running pose detection on first frame...")
    first_results = model(first_frame, verbose=False, imgsz=640)
    first_detections = result_to_pose_detections(first_results[0], first_frame.shape)

    tracker = IoUTracker(
        iou_threshold=TRACK_IOU_THRESHOLD,
        max_lost=TRACK_MAX_LOST,
    )
    first_track_ids = tracker.update(first_detections)
    print(f"[INFO] Detected {len(first_detections)} student candidate(s).")

    selected_track_id, baseline_yaw = select_student(
        first_frame,
        first_detections,
        first_track_ids,
        CONF_DEFAULT / 100.0,
    )
    if selected_track_id is None:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Student selection cancelled.")
        return

    tracker.keep_only({selected_track_id})
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    display_scale = display_scale_for(first_frame)
    display_w = int(frame_w * display_scale)
    display_h = int(frame_h * display_scale)
    if display_scale < 1.0:
        print(
            f"[INFO] Display preview scaled to {display_w}x{display_h} "
            f"from {frame_w}x{frame_h}."
        )

    win = "AISENTINEL - Head Behavior Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Head Tilt (deg)", win, TILT_DEFAULT, TILT_SLIDER_MAX, _noop)
    cv2.createTrackbar("Head Turn (%)", win, TURN_DEFAULT, TURN_SLIDER_MAX, _noop)
    cv2.createTrackbar("Shoulder Turn (deg)", win, SHOULDER_DEFAULT, SHOULDER_SLIDER_MAX, _noop)
    cv2.createTrackbar("KP Conf (%)", win, CONF_DEFAULT, CONF_SLIDER_MAX, _noop)

    print()
    print("[INFO] Calibration running.")
    print("[INFO] Adjust sliders in the OpenCV window.")
    print("[INFO] SPACE pauses/resumes. Q or ESC quits and prints config values.")
    print()

    paused = False
    last_frame = None
    last_detections = []
    last_track_ids = []
    frame_idx = 0
    thresholds = {
        "tilt": float(TILT_DEFAULT),
        "turn_pct": float(TURN_DEFAULT),
        "turn": float(TURN_DEFAULT) / 100.0,
        "shoulder": float(SHOULDER_DEFAULT),
        "conf": float(CONF_DEFAULT) / 100.0,
    }

    try:
        while True:
            thresholds = read_thresholds(win)

            if paused and last_frame is not None:
                frame = last_frame
                detections = last_detections
                track_ids = last_track_ids
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video reached.")
                    break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                results = model(frame, verbose=False, imgsz=640)
                detections = result_to_pose_detections(results[0], frame.shape)
                track_ids = tracker.update(detections)
                last_frame = frame.copy()
                last_detections = detections
                last_track_ids = track_ids

            annotated = render_calibration_frame(
                frame,
                detections,
                track_ids,
                selected_track_id,
                baseline_yaw,
                thresholds,
                frame_idx,
                total_frames,
                paused,
            )

            cv2.imshow(win, resize_for_display(annotated, display_scale))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print("[INFO] Paused." if paused else "[INFO] Resumed.")

            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print_final_config(thresholds)


if __name__ == "__main__":
    main()
