#!/usr/bin/env python3
"""
Head Behavior Threshold Calibration Tool
=========================================
Live camera feed with real-time pose angle measurements and
OpenCV trackbar sliders for tuning detection thresholds.

Displays:
  - Look-at-neighbor offset ratio (nose vs shoulder midpoint)
  - Minimal visual indicators (kept small to avoid covering camera view)

Use the sliders to find the right threshold values, then copy
them into front_node_head_behavior_pc.py.

Controls:
    Trackbars  - adjust LOOK_NEIGHBOR_RATIO, KP_CONF_THRESH
    Q / ESC    - quit and print final threshold values
    SPACE      - freeze / unfreeze frame for inspection

Requirements:
    pip install ultralytics opencv-python numpy
"""

import sys
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
RATIO_SLIDER_MAX = 100      # percent (displayed as 0.00-1.00)
CONF_SLIDER_MAX = 100       # percent
EAR_SYM_SLIDER_MAX = 100    # percent (displayed as 0.00-1.00)

RATIO_DEFAULT = 35          # percent -> 0.35
CONF_DEFAULT = 30           # percent -> 0.30
EAR_SYM_DEFAULT = 60        # percent -> 0.60

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


# ── Pose measurement functions ───────────────────────────────
def measure_look_neighbor(kp_xy, kp_conf, conf_thresh, ear_sym_thresh):
    """Returns (valid, offset_ratio, direction, nose_x, shoulder_center_x,
               ear_symmetry, body_suppressed)."""
    if kp_conf[KP_NOSE] < conf_thresh:
        return False, 0.0, "", 0, 0, 0.0, False
    if (kp_conf[KP_LEFT_SHOULDER] < conf_thresh or
            kp_conf[KP_RIGHT_SHOULDER] < conf_thresh):
        return False, 0.0, "", 0, 0, 0.0, False

    nose_x = float(kp_xy[KP_NOSE][0])
    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])

    shoulder_center_x = (ls_x + rs_x) / 2.0
    shoulder_width = abs(rs_x - ls_x)

    if shoulder_width < 5:
        return False, 0.0, "", 0, 0, 0.0, False

    offset = nose_x - shoulder_center_x
    offset_ratio = abs(offset) / shoulder_width
    direction = "RIGHT" if offset > 0 else "LEFT"

    # Body-orientation compensation via ear symmetry
    left_ear_conf = float(kp_conf[KP_LEFT_EAR])
    right_ear_conf = float(kp_conf[KP_RIGHT_EAR])
    max_ear = max(left_ear_conf, right_ear_conf)
    min_ear = min(left_ear_conf, right_ear_conf)

    ear_symmetry = (min_ear / max_ear) if max_ear > conf_thresh else 1.0
    body_suppressed = False

    if max_ear > conf_thresh and ear_symmetry < ear_sym_thresh:
        body_turn_dir = "RIGHT" if left_ear_conf < right_ear_conf else "LEFT"
        if direction == body_turn_dir:
            body_suppressed = True

    return True, offset_ratio, direction, nose_x, shoulder_center_x, ear_symmetry, body_suppressed


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

    cv2.createTrackbar("Look Ratio (%)", win, RATIO_DEFAULT, RATIO_SLIDER_MAX, _noop)
    cv2.createTrackbar("Ear Sym (%)", win, EAR_SYM_DEFAULT, EAR_SYM_SLIDER_MAX, _noop)
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
        ratio_thresh_pct = cv2.getTrackbarPos("Look Ratio (%)", win)
        ear_sym_thresh_pct = cv2.getTrackbarPos("Ear Sym (%)", win)
        conf_thresh_pct = cv2.getTrackbarPos("KP Conf (%)", win)

        ratio_thresh = ratio_thresh_pct / 100.0
        ear_sym_thresh = ear_sym_thresh_pct / 100.0
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

                # Measure look-at-neighbor
                look_valid, look_ratio, look_dir, nose_x, sc_x, \
                    ear_sym, body_suppressed = \
                    measure_look_neighbor(kp_xy, kp_conf, conf_thresh,
                                         ear_sym_thresh)
                look_triggered = (look_valid and look_ratio > ratio_thresh
                                  and not body_suppressed)

                # Determine box color
                box_color = (COL_TRIGGERED if look_triggered
                             else COL_LOOK_NEIGHBOR if look_valid
                             else COL_NORMAL)

                # Draw skeleton (thin lines to keep view clear)
                draw_skeleton(annotated, kp_xy, kp_conf, conf_thresh,
                              color=box_color)

                # Draw bounding box (thin)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)

                # Draw nose-to-shoulder-center line if look is valid
                if look_valid:
                    nose_pt = (int(kp_xy[KP_NOSE][0]),
                               int(kp_xy[KP_NOSE][1]))
                    sc_y = int((kp_xy[KP_LEFT_SHOULDER][1] +
                                kp_xy[KP_RIGHT_SHOULDER][1]) / 2)
                    sc_pt = (int(sc_x), sc_y)
                    line_col = (COL_TRIGGERED if look_triggered
                                else COL_LOOK_NEIGHBOR)
                    cv2.line(annotated, nose_pt, sc_pt, line_col, 1,
                             cv2.LINE_AA)

                # Compact per-person label (single line, top of bbox)
                if look_valid:
                    sup_tag = " SUP" if body_suppressed else ""
                    lbl = f"L:{look_ratio:.2f} {look_dir}{sup_tag} E:{ear_sym:.2f}"
                    draw_label(annotated, lbl, x1, y1 - 5, box_color)

                person_measurements.append({
                    "idx": i,
                    "look_valid": look_valid,
                    "look_ratio": look_ratio,
                    "look_dir": look_dir,
                    "look_triggered": look_triggered,
                    "ear_sym": ear_sym,
                    "body_suppressed": body_suppressed,
                })

        # ── Compact status bar (bottom-left, single strip) ────────
        bar_h = 20
        bar_y = cam_h - bar_h - 4
        bar_x = 4

        # Build status text
        status_parts = [
            f"Ratio={ratio_thresh:.2f}",
            f"EarSym={ear_sym_thresh:.2f}",
            f"Conf={conf_thresh:.2f}",
        ]
        if person_measurements:
            pm = person_measurements[0]
            if pm["look_valid"]:
                sup_tag = " SUP" if pm["body_suppressed"] else ""
                status_parts.append(
                    f"| Look:{pm['look_ratio']:.2f} "
                    f"{pm['look_dir']}{sup_tag}")
                status_parts.append(f"Ear:{pm['ear_sym']:.2f}")
            else:
                status_parts.append("| Look:--")
        else:
            status_parts.append("| No person")

        status_text = "  ".join(status_parts)
        (tw, th), _ = cv2.getTextSize(status_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Semi-transparent background bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (bar_x, bar_y - 2),
                      (bar_x + tw + 8, bar_y + th + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        cv2.putText(annotated, status_text, (bar_x + 4, bar_y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220),
                    1, cv2.LINE_AA)

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

    print()
    print("=" * 60)
    print("  Final Calibrated Thresholds")
    print("=" * 60)
    print()
    print("  Copy these into front_node_head_behavior_pc.py:")
    print()
    print(f"  LOOK_NEIGHBOR_RATIO  = {ratio_thresh:.2f}")
    print(f"  EAR_SYMMETRY_THRESH  = {ear_sym_thresh:.2f}")
    print(f"  KP_CONF_THRESH       = {conf_thresh:.2f}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
