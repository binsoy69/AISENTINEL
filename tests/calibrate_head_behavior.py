#!/usr/bin/env python3
"""
Head Behavior Threshold Calibration Tool
=========================================
Live camera feed with real-time pose angle measurements and
OpenCV trackbar sliders for tuning detection thresholds.

Displays:
  - Head tilt angle (ear-to-ear)
  - Look-at-neighbor offset ratio (nose vs shoulder midpoint)
  - Visual indicators showing when thresholds are exceeded

Use the sliders to find the right threshold values, then copy
them into front_node_head_behavior_pc.py.

Controls:
    Trackbars  - adjust HEAD_TILT_ANGLE_DEG, LOOK_NEIGHBOR_RATIO, KP_CONF_THRESH
    Q / ESC    - quit and print final threshold values
    SPACE      - freeze / unfreeze frame for inspection

Requirements:
    pip install ultralytics opencv-python numpy
"""

import sys
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
POSE_MODEL_PATH = REPO_ROOT / "yolo26s-pose.pt"

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)
COL_HEAD_TILT = (0, 165, 255)       # orange
COL_LOOK_NEIGHBOR = (255, 0, 255)   # magenta
COL_TRIGGERED = (0, 0, 255)         # red
COL_GUIDE = (255, 255, 0)           # cyan
COL_DIM = (100, 100, 100)

# ── Skeleton (upper body) ───────────────────────────────────
SKELETON = [
    (KP_NOSE, 1), (KP_NOSE, 2),
    (1, KP_LEFT_EAR), (2, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, 7), (KP_RIGHT_SHOULDER, 8),
    (7, 9), (8, 10),
    (KP_LEFT_SHOULDER, 11), (KP_RIGHT_SHOULDER, 12),
    (11, 12),
]

# ── Default slider ranges ───────────────────────────────────
# Sliders use integers; we scale floats by a multiplier.
TILT_SLIDER_MAX = 90        # degrees
RATIO_SLIDER_MAX = 100      # percent (displayed as 0.00-1.00)
CONF_SLIDER_MAX = 100       # percent

TILT_DEFAULT = 30           # degrees
RATIO_DEFAULT = 35          # percent -> 0.35
CONF_DEFAULT = 30           # percent -> 0.30

CAMERA_INDEX = 0


# ── Drawing helpers ──────────────────────────────────────────
def draw_skeleton(img, kp_xy, kp_conf, conf_thresh, color=COL_GUIDE):
    for i, j in SKELETON:
        if i < len(kp_conf) and j < len(kp_conf):
            if kp_conf[i] > conf_thresh and kp_conf[j] > conf_thresh:
                p1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                p2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    for k in range(min(len(kp_xy), 13)):
        if kp_conf[k] > conf_thresh:
            cv2.circle(img, (int(kp_xy[k][0]), int(kp_xy[k][1])),
                       3, color, -1, cv2.LINE_AA)


def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def draw_gauge(img, x, y, value, threshold, label, max_val,
               color_normal, color_triggered):
    """Draw a horizontal bar gauge showing value vs threshold."""
    bar_w, bar_h = 200, 16
    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (80, 80, 80), 1)

    # Value fill
    fill = min(value / max_val, 1.0) if max_val > 0 else 0
    triggered = value > threshold
    color = color_triggered if triggered else color_normal
    cv2.rectangle(img, (x, y), (x + int(bar_w * fill), y + bar_h), color, -1)

    # Threshold marker
    thresh_x = x + int((threshold / max_val) * bar_w)
    cv2.line(img, (thresh_x, y - 2), (thresh_x, y + bar_h + 2),
             (255, 255, 255), 2)

    # Label
    status = "TRIGGERED" if triggered else "ok"
    text = f"{label}: {value:.1f} (thresh: {threshold:.1f}) [{status}]"
    cv2.putText(img, text, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color_triggered if triggered else (220, 220, 220),
                1, cv2.LINE_AA)


# ── Pose measurement functions ───────────────────────────────
def measure_head_tilt(kp_xy, kp_conf, conf_thresh):
    """Returns (valid, angle_degrees)."""
    if (kp_conf[KP_LEFT_EAR] < conf_thresh or
            kp_conf[KP_RIGHT_EAR] < conf_thresh):
        return False, 0.0
    le = kp_xy[KP_LEFT_EAR]
    re = kp_xy[KP_RIGHT_EAR]
    raw = abs(math.degrees(
        math.atan2(float(re[1]) - float(le[1]),
                   float(re[0]) - float(le[0]))
    ))
    # Normalize: 0° = level head, 90° = fully sideways.
    # atan2 gives ~0 or ~180 for level ears depending on ear ordering
    # in the image (mirrored), so map >90 back toward 0.
    angle = raw if raw <= 90 else 180 - raw
    return True, angle


def measure_look_neighbor(kp_xy, kp_conf, conf_thresh):
    """Returns (valid, offset_ratio, direction, nose_x, shoulder_center_x)."""
    if kp_conf[KP_NOSE] < conf_thresh:
        return False, 0.0, "", 0, 0
    if (kp_conf[KP_LEFT_SHOULDER] < conf_thresh or
            kp_conf[KP_RIGHT_SHOULDER] < conf_thresh):
        return False, 0.0, "", 0, 0

    nose_x = float(kp_xy[KP_NOSE][0])
    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])

    shoulder_center_x = (ls_x + rs_x) / 2.0
    shoulder_width = abs(rs_x - ls_x)

    if shoulder_width < 5:
        return False, 0.0, "", 0, 0

    offset = nose_x - shoulder_center_x
    offset_ratio = abs(offset) / shoulder_width
    direction = "RIGHT" if offset > 0 else "LEFT"
    return True, offset_ratio, direction, nose_x, shoulder_center_x


# ── Trackbar callback (no-op, we read values each frame) ────
def _noop(_):
    pass


# ── Main ─────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Threshold Calibration Tool")
    print("  Adjust sliders to find optimal thresholds")
    print("=" * 60)
    print()

    # Load model
    if not POSE_MODEL_PATH.exists():
        print(f"[INFO] Model not found at {POSE_MODEL_PATH}")
        print("[INFO] Ultralytics will auto-download on first use.")
    print(f"[INFO] Loading pose model: {POSE_MODEL_PATH.name}")
    model = YOLO(str(POSE_MODEL_PATH))
    print("[INFO] Model loaded.")

    # Open camera
    print(f"[INFO] Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {cam_w}x{cam_h}")

    # Create window and trackbars
    win = "AISENTINEL - Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("Head Tilt (deg)", win, TILT_DEFAULT, TILT_SLIDER_MAX, _noop)
    cv2.createTrackbar("Look Ratio (%)", win, RATIO_DEFAULT, RATIO_SLIDER_MAX, _noop)
    cv2.createTrackbar("KP Conf (%)", win, CONF_DEFAULT, CONF_SLIDER_MAX, _noop)

    print()
    print("[INFO] Calibration running. Adjust sliders in the window.")
    print("[INFO] Press Q or ESC to quit. SPACE to freeze/unfreeze.")
    print()

    frozen = False
    frozen_frame = None

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera.")
                break
        else:
            frame = frozen_frame.copy()

        # Read current slider values
        tilt_thresh = cv2.getTrackbarPos("Head Tilt (deg)", win)
        ratio_thresh_pct = cv2.getTrackbarPos("Look Ratio (%)", win)
        conf_thresh_pct = cv2.getTrackbarPos("KP Conf (%)", win)

        tilt_thresh = max(tilt_thresh, 1)
        ratio_thresh = ratio_thresh_pct / 100.0
        conf_thresh = conf_thresh_pct / 100.0

        # Run pose detection
        results = model(frame, verbose=False, imgsz=640)

        annotated = frame.copy()
        keypoints_data = results[0].keypoints
        boxes = results[0].boxes

        has_detections = (
            boxes is not None
            and keypoints_data is not None
            and len(boxes) > 0
        )

        person_measurements = []

        if has_detections:
            bboxes = boxes.xyxy.cpu().numpy()
            kps_xy = keypoints_data.xy.cpu().numpy()
            kps_conf = keypoints_data.conf.cpu().numpy()

            for i in range(len(bboxes)):
                kp_xy = kps_xy[i]
                kp_conf = kps_conf[i]
                x1, y1, x2, y2 = [int(v) for v in bboxes[i]]

                # Measure head tilt
                tilt_valid, tilt_angle = measure_head_tilt(
                    kp_xy, kp_conf, conf_thresh)
                tilt_triggered = tilt_valid and tilt_angle > tilt_thresh

                # Measure look-at-neighbor
                look_valid, look_ratio, look_dir, nose_x, sc_x = \
                    measure_look_neighbor(kp_xy, kp_conf, conf_thresh)
                look_triggered = look_valid and look_ratio > ratio_thresh

                # Determine box color
                if tilt_triggered and look_triggered:
                    box_color = COL_TRIGGERED
                elif tilt_triggered:
                    box_color = COL_HEAD_TILT
                elif look_triggered:
                    box_color = COL_LOOK_NEIGHBOR
                else:
                    box_color = COL_NORMAL

                # Draw skeleton
                draw_skeleton(annotated, kp_xy, kp_conf, conf_thresh,
                              color=box_color)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Draw ear-to-ear line if tilt is valid
                if tilt_valid:
                    le = (int(kp_xy[KP_LEFT_EAR][0]),
                          int(kp_xy[KP_LEFT_EAR][1]))
                    re = (int(kp_xy[KP_RIGHT_EAR][0]),
                          int(kp_xy[KP_RIGHT_EAR][1]))
                    line_col = COL_TRIGGERED if tilt_triggered else COL_HEAD_TILT
                    cv2.line(annotated, le, re, line_col, 2, cv2.LINE_AA)
                    # Angle text near ear
                    mid_ear = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2 - 10)
                    cv2.putText(annotated, f"{tilt_angle:.1f} deg", mid_ear,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_col,
                                1, cv2.LINE_AA)

                # Draw nose-to-shoulder-center line if look is valid
                if look_valid:
                    nose_pt = (int(kp_xy[KP_NOSE][0]),
                               int(kp_xy[KP_NOSE][1]))
                    sc_y = int((kp_xy[KP_LEFT_SHOULDER][1] +
                                kp_xy[KP_RIGHT_SHOULDER][1]) / 2)
                    sc_pt = (int(sc_x), sc_y)
                    line_col = (COL_TRIGGERED if look_triggered
                                else COL_LOOK_NEIGHBOR)
                    cv2.line(annotated, nose_pt, sc_pt, line_col, 2,
                             cv2.LINE_AA)
                    # Ratio text near nose
                    cv2.putText(
                        annotated,
                        f"{look_ratio:.2f} ({look_dir})",
                        (nose_pt[0] + 5, nose_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_col,
                        1, cv2.LINE_AA)

                # Per-person label
                labels = [f"Person {i}"]
                if tilt_valid:
                    labels.append(f"Tilt: {tilt_angle:.1f} deg")
                if look_valid:
                    labels.append(f"Look: {look_ratio:.2f} {look_dir}")

                lbl_y = y1 - 5
                for lbl in labels:
                    draw_label(annotated, lbl, x1, lbl_y, box_color)
                    lbl_y -= 20

                person_measurements.append({
                    "idx": i,
                    "tilt_valid": tilt_valid,
                    "tilt_angle": tilt_angle,
                    "tilt_triggered": tilt_triggered,
                    "look_valid": look_valid,
                    "look_ratio": look_ratio,
                    "look_dir": look_dir,
                    "look_triggered": look_triggered,
                })

        # ── HUD Panel (bottom-left) ─────────────────────────────
        panel_h = 160
        panel_w = 420
        panel_y = cam_h - panel_h - 10
        panel_x = 10

        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                       (panel_x + panel_w, panel_y + panel_h),
                       (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Current threshold values
        cv2.putText(annotated, "CURRENT THRESHOLDS:", (panel_x + 5, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated,
                    f"HEAD_TILT_ANGLE_DEG = {tilt_thresh:.1f}",
                    (panel_x + 5, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_HEAD_TILT, 1)
        cv2.putText(annotated,
                    f"LOOK_NEIGHBOR_RATIO = {ratio_thresh:.2f}",
                    (panel_x + 5, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LOOK_NEIGHBOR, 1)
        cv2.putText(annotated,
                    f"KP_CONF_THRESH      = {conf_thresh:.2f}",
                    (panel_x + 5, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GUIDE, 1)

        # Live readings for first person
        if person_measurements:
            pm = person_measurements[0]
            y_off = panel_y + 105
            if pm["tilt_valid"]:
                cv2.putText(annotated,
                            f"Live Tilt: {pm['tilt_angle']:.1f} deg",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_TRIGGERED if pm["tilt_triggered"]
                            else COL_NORMAL, 1)
            else:
                cv2.putText(annotated, "Live Tilt: -- (low conf)",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)
            y_off += 22
            if pm["look_valid"]:
                cv2.putText(annotated,
                            f"Live Look: {pm['look_ratio']:.2f} {pm['look_dir']}",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_TRIGGERED if pm["look_triggered"]
                            else COL_NORMAL, 1)
            else:
                cv2.putText(annotated, "Live Look: -- (low conf)",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)
        else:
            cv2.putText(annotated, "No person detected",
                        (panel_x + 5, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)

        # ── Gauges (top-right) ──────────────────────────────────
        if person_measurements:
            pm = person_measurements[0]
            gauge_x = cam_w - 380
            if pm["tilt_valid"]:
                draw_gauge(annotated, gauge_x, 30,
                           pm["tilt_angle"], float(tilt_thresh),
                           "Head Tilt", 90.0,
                           COL_HEAD_TILT, COL_TRIGGERED)
            if pm["look_valid"]:
                draw_gauge(annotated, gauge_x, 75,
                           pm["look_ratio"] * 100, ratio_thresh_pct,
                           "Look Ratio", 100.0,
                           COL_LOOK_NEIGHBOR, COL_TRIGGERED)

        # Frozen indicator
        if frozen:
            cv2.putText(annotated, "FROZEN (SPACE to unfreeze)",
                        (cam_w // 2 - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Show ────────────────────────────────────────────────
        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if not frozen:
                frozen = True
                frozen_frame = frame.copy()
                print("[INFO] Frame frozen. Press SPACE to unfreeze.")
            else:
                frozen = False
                print("[INFO] Unfrozen.")

    # ── Cleanup & print final values ────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # Read final slider positions
    final_tilt = tilt_thresh
    final_ratio = ratio_thresh
    final_conf = conf_thresh

    print()
    print("=" * 60)
    print("  Final Calibrated Thresholds")
    print("=" * 60)
    print()
    print("  Copy these into front_node_head_behavior_pc.py:")
    print()
    print(f"  HEAD_TILT_ANGLE_DEG = {final_tilt:.1f}")
    print(f"  LOOK_NEIGHBOR_RATIO = {final_ratio:.2f}")
    print(f"  KP_CONF_THRESH      = {final_conf:.2f}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
