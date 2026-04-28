#!/usr/bin/env python3
"""
Head Behavior Threshold Calibration Tool
=========================================
Live camera feed with real-time pose angle measurements and
OpenCV trackbar sliders for tuning detection thresholds.

Displays:
  - Head tilt angle (ear-to-ear roll)
  - Head turn ratio (nose offset / shoulder width - yaw)
  - Shoulder turn angle (shoulder-line deviation for overhead camera)
  - Visual indicators showing when thresholds are exceeded

Use the sliders to find the right threshold values, then copy
them into front_node_head_behavior_pc.py.

Controls:
    Trackbars  - adjust HEAD_TILT_ANGLE_DEG, HEAD_TURN_RATIO, SHOULDER_TURN_ANGLE_DEG, KP_CONF_THRESH
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

from front_node_pc_common import POSE_MODEL_CANDIDATES, first_existing

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
POSE_MODEL_PATH = first_existing(POSE_MODEL_CANDIDATES) or Path("yolo11n-pose.pt")

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)
COL_HEAD_TILT = (0, 165, 255)       # orange
COL_SHOULDER_TURN = (255, 191, 0)   # deep sky blue (BGR)
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
TILT_SLIDER_MAX = 90        # degrees (ear-to-ear roll)
TURN_SLIDER_MAX = 100       # percent (nose offset / shoulder width, displayed as 0.00-1.00)
SHOULDER_SLIDER_MAX = 90    # degrees
CONF_SLIDER_MAX = 100       # percent

TILT_DEFAULT = 30           # degrees
TURN_DEFAULT = 26           # percent -> 0.26
SHOULDER_DEFAULT = 20       # degrees (overhead camera)
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
    """Horizontal bar gauge with threshold marker."""
    bar_w = 350
    bar_h = 18

    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)

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
    """Returns (valid, roll_angle, yaw_ratio).
    roll_angle: ear-to-ear angle in degrees (0 = level).
    yaw_ratio: nose offset / shoulder width (0 = centered).
    Either or both may be valid; valid=True if at least one signal exists.
    """
    roll_angle = 0.0
    yaw_ratio = 0.0
    has_roll = False
    has_yaw = False

    # Roll: ear-to-ear angle
    if (kp_conf[KP_LEFT_EAR] >= conf_thresh and
            kp_conf[KP_RIGHT_EAR] >= conf_thresh):
        le = kp_xy[KP_LEFT_EAR]
        re = kp_xy[KP_RIGHT_EAR]
        raw = abs(math.degrees(
            math.atan2(float(re[1]) - float(le[1]),
                       float(re[0]) - float(le[0]))
        ))
        roll_angle = raw if raw <= 90 else 180 - raw
        has_roll = True

    # Yaw: nose offset from shoulder center
    if (kp_conf[KP_NOSE] >= conf_thresh and
            kp_conf[KP_LEFT_SHOULDER] >= conf_thresh and
            kp_conf[KP_RIGHT_SHOULDER] >= conf_thresh):
        nose_x = float(kp_xy[KP_NOSE][0])
        ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
        rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
        shoulder_width = abs(rs_x - ls_x)
        if shoulder_width >= 5:
            shoulder_center_x = (ls_x + rs_x) / 2.0
            yaw_ratio = abs(nose_x - shoulder_center_x) / shoulder_width
            has_yaw = True

    return (has_roll or has_yaw), roll_angle, yaw_ratio


def measure_shoulder_turn(kp_xy, kp_conf, conf_thresh):
    """
    Shoulder angle measurement for OVERHEAD camera calibration.
    Measures the angle of the shoulder line relative to horizontal.
    Returns (valid, angle_degrees, direction).
    """
    if (kp_conf[KP_LEFT_SHOULDER] < conf_thresh or
            kp_conf[KP_RIGHT_SHOULDER] < conf_thresh):
        return False, 0.0, ""

    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    ls_y = float(kp_xy[KP_LEFT_SHOULDER][1])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
    rs_y = float(kp_xy[KP_RIGHT_SHOULDER][1])

    shoulder_dist = math.hypot(rs_x - ls_x, rs_y - ls_y)
    if shoulder_dist < 10:
        return False, 0.0, ""

    raw_angle = math.degrees(
        math.atan2(rs_y - ls_y, rs_x - ls_x)
    )
    angle = abs(raw_angle) if abs(raw_angle) <= 90 else 180 - abs(raw_angle)
    direction = "RIGHT" if raw_angle > 0 else "LEFT"
    return True, angle, direction


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
    cv2.createTrackbar("Head Turn (%)", win, TURN_DEFAULT, TURN_SLIDER_MAX, _noop)
    cv2.createTrackbar("Shoulder Turn (deg)", win, SHOULDER_DEFAULT, SHOULDER_SLIDER_MAX, _noop)
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
        turn_thresh_pct = cv2.getTrackbarPos("Head Turn (%)", win)
        shoulder_thresh = cv2.getTrackbarPos("Shoulder Turn (deg)", win)
        conf_thresh_pct = cv2.getTrackbarPos("KP Conf (%)", win)

        tilt_thresh = max(tilt_thresh, 1)
        turn_thresh = turn_thresh_pct / 100.0
        shoulder_thresh = max(shoulder_thresh, 1)
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

                # Measure head tilt (roll + yaw)
                tilt_valid, roll_angle, yaw_ratio = measure_head_tilt(
                    kp_xy, kp_conf, conf_thresh)
                roll_triggered = roll_angle > tilt_thresh
                yaw_triggered = yaw_ratio > turn_thresh
                tilt_triggered = tilt_valid and (roll_triggered or yaw_triggered)

                # Measure shoulder turn (overhead camera)
                shoulder_valid, shoulder_angle, shoulder_dir = \
                    measure_shoulder_turn(kp_xy, kp_conf, conf_thresh)
                shoulder_triggered = (shoulder_valid
                                      and shoulder_angle > shoulder_thresh)

                # Determine box color (priority: triggered > individual)
                any_triggered = (tilt_triggered or shoulder_triggered)
                if any_triggered:
                    box_color = COL_TRIGGERED
                elif shoulder_triggered:
                    box_color = COL_SHOULDER_TURN
                else:
                    box_color = COL_NORMAL

                # Draw skeleton
                draw_skeleton(annotated, kp_xy, kp_conf, conf_thresh,
                              color=box_color)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Draw ear-to-ear line (roll)
                if (kp_conf[KP_LEFT_EAR] >= conf_thresh and
                        kp_conf[KP_RIGHT_EAR] >= conf_thresh):
                    le = (int(kp_xy[KP_LEFT_EAR][0]),
                          int(kp_xy[KP_LEFT_EAR][1]))
                    re = (int(kp_xy[KP_RIGHT_EAR][0]),
                          int(kp_xy[KP_RIGHT_EAR][1]))
                    line_col = COL_TRIGGERED if roll_triggered else COL_HEAD_TILT
                    cv2.line(annotated, le, re, line_col, 2, cv2.LINE_AA)
                    mid_ear = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2 - 10)
                    cv2.putText(annotated, f"{roll_angle:.1f} deg", mid_ear,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_col,
                                1, cv2.LINE_AA)

                # Draw nose-to-shoulder-center line (yaw)
                if (kp_conf[KP_NOSE] >= conf_thresh and
                        kp_conf[KP_LEFT_SHOULDER] >= conf_thresh and
                        kp_conf[KP_RIGHT_SHOULDER] >= conf_thresh):
                    nose_pt = (int(kp_xy[KP_NOSE][0]),
                               int(kp_xy[KP_NOSE][1]))
                    sc_x = int((kp_xy[KP_LEFT_SHOULDER][0] +
                                kp_xy[KP_RIGHT_SHOULDER][0]) / 2)
                    sc_y = int((kp_xy[KP_LEFT_SHOULDER][1] +
                                kp_xy[KP_RIGHT_SHOULDER][1]) / 2)
                    yaw_col = COL_TRIGGERED if yaw_triggered else COL_HEAD_TILT
                    cv2.line(annotated, nose_pt, (sc_x, sc_y), yaw_col, 2,
                             cv2.LINE_AA)
                    cv2.putText(annotated, f"{yaw_ratio:.2f}",
                                (nose_pt[0] + 5, nose_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, yaw_col,
                                1, cv2.LINE_AA)

                # Draw shoulder line if shoulder measurement is valid
                if shoulder_valid:
                    ls_pt = (int(kp_xy[KP_LEFT_SHOULDER][0]),
                             int(kp_xy[KP_LEFT_SHOULDER][1]))
                    rs_pt = (int(kp_xy[KP_RIGHT_SHOULDER][0]),
                             int(kp_xy[KP_RIGHT_SHOULDER][1]))
                    s_col = (COL_TRIGGERED if shoulder_triggered
                             else COL_SHOULDER_TURN)
                    cv2.line(annotated, ls_pt, rs_pt, s_col, 3, cv2.LINE_AA)
                    mid_s = ((ls_pt[0] + rs_pt[0]) // 2,
                             (ls_pt[1] + rs_pt[1]) // 2 - 12)
                    cv2.putText(annotated,
                                f"{shoulder_angle:.1f} deg ({shoulder_dir})",
                                mid_s, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                s_col, 1, cv2.LINE_AA)

                # Per-person label
                labels = [f"Person {i}"]
                if tilt_valid:
                    labels.append(f"Roll: {roll_angle:.1f} deg")
                    labels.append(f"Yaw: {yaw_ratio:.2f}")
                if shoulder_valid:
                    labels.append(
                        f"Shoulder: {shoulder_angle:.1f} deg {shoulder_dir}")

                lbl_y = y1 - 5
                for lbl in labels:
                    draw_label(annotated, lbl, x1, lbl_y, box_color)
                    lbl_y -= 20

                person_measurements.append({
                    "idx": i,
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
                })

        # ── HUD Panel (bottom-left) ─────────────────────────────
        panel_h = 200
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
                    f"HEAD_TURN_RATIO     = {turn_thresh:.2f}",
                    (panel_x + 5, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_HEAD_TILT, 1)
        cv2.putText(annotated,
                    f"SHOULDER_TURN_DEG   = {shoulder_thresh:.1f}",
                    (panel_x + 5, panel_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_SHOULDER_TURN, 1)
        cv2.putText(annotated,
                    f"KP_CONF_THRESH      = {conf_thresh:.2f}",
                    (panel_x + 5, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GUIDE, 1)

        # Live readings for first person
        if person_measurements:
            pm = person_measurements[0]
            y_off = panel_y + 125
            if pm["tilt_valid"]:
                cv2.putText(annotated,
                            f"Live Roll: {pm['roll_angle']:.1f} deg",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_TRIGGERED if pm["roll_triggered"]
                            else COL_NORMAL, 1)
                y_off += 22
                cv2.putText(annotated,
                            f"Live Yaw: {pm['yaw_ratio']:.2f}",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_TRIGGERED if pm["yaw_triggered"]
                            else COL_NORMAL, 1)
            else:
                cv2.putText(annotated, "Live Tilt: -- (low conf)",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)
            y_off += 22
            if pm["shoulder_valid"]:
                cv2.putText(annotated,
                            f"Live Shoulder: {pm['shoulder_angle']:.1f} deg "
                            f"{pm['shoulder_dir']}",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_TRIGGERED if pm["shoulder_triggered"]
                            else COL_NORMAL, 1)
            else:
                cv2.putText(annotated, "Live Shoulder: -- (low conf)",
                            (panel_x + 5, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)
        else:
            cv2.putText(annotated, "No person detected",
                        (panel_x + 5, panel_y + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DIM, 1)

        # ── Gauges (top-right) ──────────────────────────────────
        if person_measurements:
            pm = person_measurements[0]
            gauge_x = cam_w - 380
            if pm["tilt_valid"]:
                draw_gauge(annotated, gauge_x, 30,
                           pm["roll_angle"], float(tilt_thresh),
                           "Head Roll", 90.0,
                           COL_HEAD_TILT, COL_TRIGGERED)
                draw_gauge(annotated, gauge_x, 75,
                           pm["yaw_ratio"] * 100, turn_thresh_pct,
                           "Head Yaw", 100.0,
                           COL_HEAD_TILT, COL_TRIGGERED)
            if pm["shoulder_valid"]:
                draw_gauge(annotated, gauge_x, 120,
                           pm["shoulder_angle"], float(shoulder_thresh),
                           "Shoulder Turn", 90.0,
                           COL_SHOULDER_TURN, COL_TRIGGERED)

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
    final_turn = turn_thresh
    final_shoulder = shoulder_thresh
    final_conf = conf_thresh

    print()
    print("=" * 60)
    print("  Final Calibrated Thresholds")
    print("=" * 60)
    print()
    print("  Copy these into front_node_head_behavior_pc.py:")
    print()
    print(f"  HEAD_TILT_ANGLE_DEG      = {final_tilt:.1f}")
    print(f"  HEAD_TURN_RATIO          = {final_turn:.2f}")
    print(f"  SHOULDER_TURN_ANGLE_DEG  = {final_shoulder:.1f}")
    print(f"  KP_CONF_THRESH           = {final_conf:.2f}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
