#!/usr/bin/env python3
"""
Front Node Pose-Based Behavior Detection - PC Test Program
============================================================
Runs YOLOv11n-pose with ByteTrack on video files to detect three
pose-based cheating behaviors from the Front Node perspective:

  1. Head tilting      — ear-to-ear angle > 25° sustained for 3s
  2. Hands under the table — wrists below manually-calibrated desk edge lines,
                         sustained 5s
  3. Passing papers    — wrist extends far beyond shoulder width laterally
                         toward a same-row neighbor (immediate flag)

Desk Zone Calibration:
  Before processing, the first frame is shown and you draw desk edge lines
  by clicking LEFT then RIGHT endpoint for each desk.  Zones are saved to
  a JSON file so you only calibrate once per video/camera angle.

Usage:
    python front_node_pose_test_pc.py --video path/to/exam.mp4
    python front_node_pose_test_pc.py --video exam.mp4 --zones saved.json
    python front_node_pose_test_pc.py --video exam.mp4 --desk-ratio 0.70
    python front_node_pose_test_pc.py                          # Interactive

Controls (display window):
    q / ESC  - Quit
    SPACE    - Pause / Resume

Requirements:
    pip install ultralytics opencv-python numpy lap
"""

import argparse
import sys
import os
import math
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from front_node_pc_common import POSE_MODEL_CANDIDATES, first_existing

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

POSE_MODEL_PATH = first_existing(POSE_MODEL_CANDIDATES) or Path("yolo11n-pose.pt")
DETECTION_OUTPUT_DIR = SCRIPT_DIR / "pose_detection_output"
BYTETRACK_CONFIG = SCRIPT_DIR / "bytetrack_front.yaml"
if not BYTETRACK_CONFIG.exists():
    BYTETRACK_CONFIG = SCRIPT_DIR.parent / "tests_on_pi" / "bytetrack_front.yaml"

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
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

# ── Behavior Thresholds (from PROJECT.md) ────────────────────
HEAD_TILT_ANGLE_DEG = 25.0
HEAD_TILT_SUSTAIN_SEC = 3.0

HANDS_UNDER_DESK_RATIO = 0.70    # fallback if no desk lines calibrated
HANDS_UNDER_WRIST_CONF = 0.3
HANDS_UNDER_SUSTAIN_SEC = 5.0

PASSING_REACH_MULTIPLIER = 1.2   # wrist must reach 1.2x shoulder-width from center
PASSING_ROW_TOLERANCE_PX = 50

EVENT_COOLDOWN_SEC = 10.0

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)
COL_HEAD_TILT = (0, 165, 255)     # orange
COL_HANDS_UNDER = (0, 0, 255)    # red
COL_PASSING = (255, 0, 255)      # magenta
COL_SKELETON = (255, 255, 0)     # cyan
COL_DESK_LINE = (0, 0, 255)      # red — desk edge

# Skeleton connections
SKELETON = [
    (KP_NOSE, KP_LEFT_EYE), (KP_NOSE, KP_RIGHT_EYE),
    (KP_LEFT_EYE, KP_LEFT_EAR), (KP_RIGHT_EYE, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW), (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP), (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
]


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


def log_alert(behavior: str, student_id: int, ts_sec: float,
              detail: str = "", color: str = TC.RED):
    ts = fmt_ts(ts_sec)
    print(
        f"{color}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{color}{behavior} - Student #{student_id}{TC.RESET}"
        + (f" | {detail}" if detail else "")
    )


def log_info(msg: str):
    print(f"{TC.CYAN}[INFO]{TC.RESET} {msg}")


# ══════════════════════════════════════════════════════════════
#  DESK ZONE CALIBRATION
# ══════════════════════════════════════════════════════════════

def save_desk_zones(lines, path):
    """Save desk lines to JSON."""
    data = [{"p1": list(p1), "p2": list(p2)} for p1, p2 in lines]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log_info(f"Desk zones saved: {path}")


def load_desk_zones(path):
    """Load desk lines from JSON."""
    with open(path) as f:
        data = json.load(f)
    lines = [(tuple(d["p1"]), tuple(d["p2"])) for d in data]
    log_info(f"Desk zones loaded: {path} ({len(lines)} lines)")
    return lines


def calibrate_desk_zones(video_path):
    """
    Interactive calibration UI.
    Shows the first frame; user clicks LEFT then RIGHT endpoint
    for each desk front edge.  Returns list of ((x1,y1),(x2,y2)).

    Controls:
        Left-click  — place point (first = left end, second = right end)
        Z           — undo last point / line
        C           — clear all
        ENTER/SPACE — confirm & start processing
        ESC         — cancel (returns None)
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"{TC.RED}[ERROR] Cannot read first frame from {video_path}{TC.RESET}")
        return None

    fh, fw = frame.shape[:2]
    scale = min(1.0, 1280 / fw)

    lines = []           # completed desk lines (original-res coords)
    pending_pt = None    # first click waiting for second

    def on_mouse(event, mx, my, flags, param):
        nonlocal pending_pt
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Convert display coords → original frame coords
        ox = int(mx / scale)
        oy = int(my / scale)
        if pending_pt is None:
            pending_pt = (ox, oy)
        else:
            lines.append((pending_pt, (ox, oy)))
            pending_pt = None

    win = "AISENTINEL - Desk Zone Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    instructions = [
        "Click LEFT then RIGHT endpoint for each desk edge",
        "Z: undo | C: clear | ENTER/SPACE: confirm | ESC: cancel",
    ]

    while True:
        display = frame.copy()

        # Draw completed lines
        for i, (p1, p2) in enumerate(lines):
            cv2.line(display, p1, p2, COL_DESK_LINE, 2, cv2.LINE_AA)
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2 - 12
            cv2.putText(display, f"Desk {i + 1}", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_DESK_LINE, 2)

        # Draw pending first point
        if pending_pt is not None:
            cv2.circle(display, pending_pt, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(display, "Click right endpoint...",
                        (pending_pt[0] + 10, pending_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Instructions overlay
        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Line count
        cv2.putText(display, f"Desk lines: {len(lines)}", (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Scale for display
        if scale < 1.0:
            show = cv2.resize(display, (int(fw * scale), int(fh * scale)))
        else:
            show = display

        cv2.imshow(win, show)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            if len(lines) == 0:
                log_info("No desk lines drawn.  Draw at least one or press ESC to skip.")
                continue
            break
        elif key == ord("z"):
            if pending_pt is not None:
                pending_pt = None
            elif lines:
                lines.pop()
        elif key == ord("c"):
            lines.clear()
            pending_pt = None
        elif key == 27:  # ESC
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    log_info(f"Calibration complete: {len(lines)} desk lines defined.")
    return lines


# ══════════════════════════════════════════════════════════════
#  DESK-LINE GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def desk_y_at_x(line, x):
    """Interpolate the desk line's y-coordinate at a given x."""
    (lx, ly), (rx, ry) = line
    if lx > rx:
        lx, ly, rx, ry = rx, ry, lx, ly
    if rx == lx:
        return (ly + ry) / 2.0
    t = max(0.0, min(1.0, (x - lx) / (rx - lx)))
    return ly + t * (ry - ly)


def find_desk_for_student(bbox, desk_lines):
    """
    Find the desk line most likely belonging to this student.
    Picks the line whose x-range overlaps the student's bbox and whose
    y is closest to the student's vertical center.
    Returns (line_index, line) or (None, None).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    best_idx = None
    best_dist = float("inf")

    for i, ((lx, ly), (rx, ry)) in enumerate(desk_lines):
        min_x = min(lx, rx) - 30
        max_x = max(lx, rx) + 30
        if cx < min_x or cx > max_x:
            continue  # no horizontal overlap
        line_y = desk_y_at_x(desk_lines[i], cx)
        dist = abs(line_y - cy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is not None:
        return best_idx, desk_lines[best_idx]
    return None, None


# ══════════════════════════════════════════════════════════════
#  PER-STUDENT STATE
# ══════════════════════════════════════════════════════════════

@dataclass
class StudentState:
    track_id: int

    head_tilt_start: float = -1.0
    head_tilt_flagged_at: float = -999.0

    hands_under_start: float = -1.0
    hands_under_flagged_at: float = -999.0

    passing_flagged_at: float = -999.0

    last_seen_frame: int = 0

    def can_flag(self, behavior: str, now: float) -> bool:
        last = {
            "head_tilt": self.head_tilt_flagged_at,
            "hands_under_table": self.hands_under_flagged_at,
            "passing_papers": self.passing_flagged_at,
        }.get(behavior, -999.0)
        return (now - last) > EVENT_COOLDOWN_SEC


# ══════════════════════════════════════════════════════════════
#  BEHAVIOR DETECTION
# ══════════════════════════════════════════════════════════════

def detect_head_tilt(kp_xy, kp_conf, angle_thresh):
    """
    Ear-to-ear angle (PROJECT.md formula).
    Returns (is_tilted, angle_degrees).
    """
    if kp_conf[KP_LEFT_EAR] < 0.3 or kp_conf[KP_RIGHT_EAR] < 0.3:
        return False, 0.0
    le = kp_xy[KP_LEFT_EAR]
    re = kp_xy[KP_RIGHT_EAR]
    angle = abs(math.degrees(
        math.atan2(float(re[1]) - float(le[1]),
                   float(re[0]) - float(le[0]))
    ))
    return angle > angle_thresh, angle


def detect_hands_under_lines(kp_xy, kp_conf, bbox, desk_lines):
    """
    Both wrists below their desk edge line, using calibrated desk lines.
    Only flags when a visible wrist is positionally below the line.
    Low-confidence (invisible) wrists alone do NOT trigger — from the front
    camera, wrist keypoints often drop confidence while hands are still on
    the desk due to partial occlusion.  Both-invisible only triggers if
    BOTH wrists disappear simultaneously.
    Returns (is_under, detail_str, desk_y_at_center).
    """
    idx, line = find_desk_for_student(bbox, desk_lines)
    if line is None:
        return False, "", -1

    cx = (bbox[0] + bbox[2]) / 2.0
    dy = desk_y_at_x(line, cx)

    lw_conf = float(kp_conf[KP_LEFT_WRIST])
    rw_conf = float(kp_conf[KP_RIGHT_WRIST])
    lw_y = float(kp_xy[KP_LEFT_WRIST][1])
    rw_y = float(kp_xy[KP_RIGHT_WRIST][1])

    lw_visible = lw_conf >= HANDS_UNDER_WRIST_CONF
    rw_visible = rw_conf >= HANDS_UNDER_WRIST_CONF

    # Visible wrist is below the desk line (positional check)
    left_under = lw_visible and (lw_y > dy)
    right_under = rw_visible and (rw_y > dy)

    # Both wrists invisible = likely truly hidden (under desk or out of view)
    both_invisible = (not lw_visible) and (not rw_visible)

    if (left_under and right_under) or both_invisible:
        return (True,
                f"desk={idx + 1} Lconf={lw_conf:.2f} Rconf={rw_conf:.2f} desk_y={dy:.0f}",
                dy)
    return False, "", dy


def detect_hands_under_ratio(kp_xy, kp_conf, bbox, desk_ratio):
    """Visibility-only: flag when left, right, or both wrists are invisible."""
    lw_conf = float(kp_conf[KP_LEFT_WRIST])
    rw_conf = float(kp_conf[KP_RIGHT_WRIST])

    lw_visible = lw_conf >= HANDS_UNDER_WRIST_CONF
    rw_visible = rw_conf >= HANDS_UNDER_WRIST_CONF

    if not lw_visible or not rw_visible:
        parts = []
        if not lw_visible:
            parts.append("L-invisible")
        if not rw_visible:
            parts.append("R-invisible")
        detail = f"{' & '.join(parts)} Lconf={lw_conf:.2f} Rconf={rw_conf:.2f}"
        return True, detail, -1
    return False, "", -1


def detect_passing_papers(kp_xy, kp_conf, reach_mult):
    """
    Shoulder-relative lateral wrist reach.
    If a wrist extends beyond reach_mult * shoulder_width from the
    shoulder midpoint, the student is reaching toward a neighbor.
    Returns (is_reaching, direction, wrist_x, wrist_y).
    """
    ls_conf = float(kp_conf[KP_LEFT_SHOULDER])
    rs_conf = float(kp_conf[KP_RIGHT_SHOULDER])
    if ls_conf < 0.3 or rs_conf < 0.3:
        return False, None, 0.0, 0.0

    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
    shoulder_cx = (ls_x + rs_x) / 2.0
    shoulder_w = abs(rs_x - ls_x)

    if shoulder_w < 5:  # too small / unreliable
        return False, None, 0.0, 0.0

    threshold = shoulder_w * reach_mult

    for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
        if kp_conf[idx] < 0.3:
            continue
        wx = float(kp_xy[idx][0])
        wy = float(kp_xy[idx][1])
        lateral = wx - shoulder_cx
        if abs(lateral) > threshold:
            direction = "right" if lateral > 0 else "left"
            return True, direction, wx, wy

    return False, None, 0.0, 0.0


def find_neighbor(src_id, direction, frame_data, row_tol):
    """Find nearest tracked student in `direction` with similar y-center."""
    src = frame_data.get(src_id)
    if src is None:
        return None

    sx1, sy1, sx2, sy2 = src["bbox"]
    s_cy = (sy1 + sy2) / 2
    s_cx = (sx1 + sx2) / 2

    best_id, best_dist = None, float("inf")
    for tid, td in frame_data.items():
        if tid == src_id:
            continue
        nx1, ny1, nx2, ny2 = td["bbox"]
        n_cx = (nx1 + nx2) / 2
        n_cy = (ny1 + ny2) / 2

        if abs(n_cy - s_cy) > row_tol:
            continue

        if direction == "right" and n_cx > s_cx:
            d = n_cx - s_cx
        elif direction == "left" and n_cx < s_cx:
            d = s_cx - n_cx
        else:
            continue

        if d < best_dist:
            best_dist = d
            best_id = tid

    return best_id


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_skeleton(img, kp_xy, kp_conf, color=COL_SKELETON, kp_thresh=0.3):
    for i, j in SKELETON:
        if kp_conf[i] > kp_thresh and kp_conf[j] > kp_thresh:
            p1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
            p2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    for k in range(len(kp_xy)):
        if kp_conf[k] > kp_thresh:
            cv2.circle(img, (int(kp_xy[k][0]), int(kp_xy[k][1])),
                       3, color, -1, cv2.LINE_AA)


def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def draw_desk_lines(img, desk_lines):
    """Draw all calibrated desk edge lines on the frame."""
    for i, (p1, p2) in enumerate(desk_lines):
        cv2.line(img, p1, p2, COL_DESK_LINE, 2, cv2.LINE_AA)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2 - 8
        cv2.putText(img, f"D{i + 1}", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DESK_LINE, 1)


def save_screenshot(frame, video_name, ts_sec, behavior, student_id):
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"{video_name}_{ts_str}_{behavior}_student{student_id}.jpg"
    path = DETECTION_OUTPUT_DIR / fname
    cv2.imwrite(str(path), frame)
    log_info(f"Screenshot saved: {fname}")


# ══════════════════════════════════════════════════════════════
#  MAIN PROCESSING
# ══════════════════════════════════════════════════════════════

def process_video(video_path, model, tracker_cfg, *,
                  show_display=True,
                  desk_lines=None, desk_ratio=HANDS_UNDER_DESK_RATIO,
                  head_angle=HEAD_TILT_ANGLE_DEG,
                  head_sustain=HEAD_TILT_SUSTAIN_SEC,
                  hands_sustain=HANDS_UNDER_SUSTAIN_SEC,
                  reach_mult=PASSING_REACH_MULTIPLIER):

    use_desk_lines = desk_lines is not None and len(desk_lines) > 0
    desk_mode = f"{len(desk_lines)} calibrated lines" if use_desk_lines else f"ratio {desk_ratio:.0%} (fallback)"

    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{TC.RED}[ERROR] Cannot open video: {video_path}{TC.RESET}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print()
    print("=" * 70)
    print(f"  Processing: {Path(video_path).name}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Head tilt      : >{head_angle}deg sustained {head_sustain}s")
    print(f"  Hands under    : {desk_mode}, sustained {hands_sustain}s")
    print(f"  Passing papers : reach >{reach_mult}x shoulder width (immediate)")
    print("=" * 70)

    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    students: dict[int, StudentState] = {}
    frame_idx = 0
    paused = False
    stats = defaultdict(int)
    total_alerts = 0

    while True:
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(" "):
                paused = False
            elif key in (ord("q"), 27):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            log_info("End of video reached.")
            break
        frame_idx += 1
        ts_sec = frame_idx / fps

        # ── Pose + tracking ─────────────────────────────────
        results = model.track(
            frame, persist=True, tracker=tracker_cfg,
            verbose=False, imgsz=640,
        )

        annotated = frame.copy()
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        frame_events = []

        has_tracks = (
            boxes is not None
            and boxes.id is not None
            and keypoints is not None
            and len(boxes) > 0
        )

        # Draw desk lines on every frame
        if use_desk_lines:
            draw_desk_lines(annotated, desk_lines)

        if has_tracks:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.cpu().numpy()
            kps_xy = keypoints.xy.cpu().numpy()
            kps_conf = keypoints.conf.cpu().numpy()

            frame_data = {}
            for i, tid in enumerate(track_ids):
                bbox = tuple(bboxes[i].tolist())
                frame_data[tid] = {
                    "bbox": bbox,
                    "kp_xy": kps_xy[i],
                    "kp_conf": kps_conf[i],
                }
                if tid not in students:
                    students[tid] = StudentState(track_id=tid)
                students[tid].last_seen_frame = frame_idx

            # ── Detect & annotate ────────────────────────────
            for tid, sd in frame_data.items():
                state = students[tid]
                bbox = sd["bbox"]
                kp_xy = sd["kp_xy"]
                kp_conf = sd["kp_conf"]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                box_color = COL_NORMAL
                behavior_labels = []

                draw_skeleton(annotated, kp_xy, kp_conf)

                # ── 1. Head Tilt ────────────────────────────
                is_tilted, angle = detect_head_tilt(kp_xy, kp_conf, head_angle)

                if is_tilted:
                    if state.head_tilt_start < 0:
                        state.head_tilt_start = ts_sec
                    elapsed = ts_sec - state.head_tilt_start

                    if elapsed >= head_sustain and state.can_flag("head_tilt", ts_sec):
                        state.head_tilt_flagged_at = ts_sec
                        stats["head_tilt"] += 1
                        total_alerts += 1
                        log_alert("HEAD TILT", tid, ts_sec,
                                  f"angle={angle:.1f}deg sustained {elapsed:.1f}s",
                                  TC.YELLOW)
                        frame_events.append(("head_tilt", tid))

                    if elapsed >= head_sustain * 0.5:
                        behavior_labels.append(f"TILT {angle:.0f}deg ({elapsed:.1f}s)")
                        box_color = COL_HEAD_TILT
                else:
                    state.head_tilt_start = -1.0

                # ── 2. Hands Under the Table ────────────────
                if use_desk_lines:
                    is_under, detail, dy = detect_hands_under_lines(
                        kp_xy, kp_conf, bbox, desk_lines)
                else:
                    is_under, detail, dy = detect_hands_under_ratio(
                        kp_xy, kp_conf, bbox, desk_ratio)

                if is_under:
                    if state.hands_under_start < 0:
                        state.hands_under_start = ts_sec
                    elapsed = ts_sec - state.hands_under_start

                    if elapsed >= hands_sustain and state.can_flag("hands_under_table", ts_sec):
                        state.hands_under_flagged_at = ts_sec
                        stats["hands_under_table"] += 1
                        total_alerts += 1
                        log_alert("HANDS UNDER TABLE", tid, ts_sec,
                                  f"sustained {elapsed:.1f}s | {detail}",
                                  TC.RED)
                        frame_events.append(("hands_under_table", tid))

                    if elapsed >= hands_sustain * 0.5:
                        behavior_labels.append(f"HANDS UNDER ({elapsed:.1f}s)")
                        box_color = COL_HANDS_UNDER
                else:
                    state.hands_under_start = -1.0

                # ── 3. Passing Papers ───────────────────────
                is_reach, direction, wx, wy = detect_passing_papers(
                    kp_xy, kp_conf, reach_mult)

                if is_reach and state.can_flag("passing_papers", ts_sec):
                    neighbor = find_neighbor(
                        tid, direction, frame_data, PASSING_ROW_TOLERANCE_PX)

                    if neighbor is not None:
                        state.passing_flagged_at = ts_sec
                        if neighbor in students:
                            students[neighbor].passing_flagged_at = ts_sec
                        stats["passing_papers"] += 1
                        total_alerts += 1
                        log_alert(
                            "PASSING PAPERS", tid, ts_sec,
                            f"direction={direction}, neighbor=Student #{neighbor}",
                            TC.MAGENTA,
                        )
                        frame_events.append(("passing_papers", tid))

                        nd = frame_data.get(neighbor)
                        if nd:
                            nx1, ny1, nx2, ny2 = [int(v) for v in nd["bbox"]]
                            cv2.line(annotated,
                                     ((x1 + x2) // 2, (y1 + y2) // 2),
                                     ((nx1 + nx2) // 2, (ny1 + ny2) // 2),
                                     COL_PASSING, 2, cv2.LINE_AA)
                            draw_label(annotated, f"#{neighbor} PASSING",
                                       nx1, ny1, COL_PASSING)

                        # Draw wrist reach indicator
                        cv2.circle(annotated, (int(wx), int(wy)),
                                   8, COL_PASSING, 2, cv2.LINE_AA)

                        behavior_labels.append(f"PASSING {direction.upper()}")
                        box_color = COL_PASSING

                # ── Draw person box + student ID ────────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                draw_label(annotated, f"Student #{tid}", x1, y1, box_color)

                lbl_y = y1 + 18
                for bl in behavior_labels:
                    draw_label(annotated, bl, x1, lbl_y, box_color)
                    lbl_y += 18

        # ── HUD ─────────────────────────────────────────────
        n_tracked = len(frame_data) if has_tracks else 0
        hud = [
            f"Frame: {frame_idx}/{total_frames} | Time: {fmt_ts(ts_sec)}",
            f"Tracked: {n_tracked} | Alerts: {total_alerts}"
            + (f" | Desks: {len(desk_lines)}" if use_desk_lines else " | Desks: ratio"),
        ]
        for i, line in enumerate(hud):
            y_pos = 25 + i * 28
            cv2.putText(annotated, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if total_alerts else (0, 255, 0),
                        2, cv2.LINE_AA)

        # ── Save screenshots ────────────────────────────────
        for behavior, tid in frame_events:
            save_screenshot(annotated, video_name, ts_sec, behavior, tid)

        # ── Display ─────────────────────────────────────────
        if show_display:
            disp = annotated
            if disp_scale < 1.0:
                disp = cv2.resize(annotated,
                                  (int(w * disp_scale), int(h * disp_scale)))
            cv2.imshow("AISENTINEL - Pose Behavior Detection", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                log_info("Quit requested.")
                break
            elif key == ord(" "):
                paused = True
                log_info("Paused. Press SPACE to resume.")

        if frame_idx % 500 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")

    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    # ── Summary ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  Summary for: {Path(video_path).name}")
    print("-" * 70)
    print(f"  Frames processed       : {frame_idx}")
    print(f"  Unique students tracked : {len(students)}")
    print(f"  Total alerts            : {total_alerts}")
    for beh, count in sorted(stats.items()):
        print(f"    {beh:25s}: {count}")
    print(f"  Screenshots saved to   : {DETECTION_OUTPUT_DIR}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def select_video_interactive() -> list[str]:
    print()
    print("=" * 60)
    print("  AISENTINEL - Pose Behavior Detection Test (PC)")
    print("=" * 60)
    print()
    paths = []
    while True:
        p = input("  Enter video file path (or 'done'): ").strip().strip('"').strip("'")
        if p.lower() in ("done", "d", ""):
            if paths:
                break
            print("  Please provide at least one video file.")
            continue
        if not os.path.isfile(p):
            print(f"  [ERROR] File not found: {p}")
            continue
        paths.append(p)
        print(f"  Added: {Path(p).name}")
    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="AISENTINEL Front Node - Pose Behavior Detection (PC Test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detected Behaviors:
  1. Head tilting       - ear-to-ear angle exceeds threshold, sustained
  2. Hands under the table - wrists below desk edge (calibrated or ratio), sustained
  3. Passing papers     - wrist reach beyond shoulder width toward neighbor

Desk Zone Modes:
  Default: interactive calibration on first frame (saved to JSON)
  --zones FILE         : load previously saved desk zones
  --desk-ratio FLOAT   : skip calibration, use fixed bbox ratio (fallback)

Examples:
  python front_node_pose_test_pc.py --video exam.mp4
  python front_node_pose_test_pc.py --video exam.mp4 --zones exam_desks.json
  python front_node_pose_test_pc.py --video exam.mp4 --desk-ratio 0.70
  python front_node_pose_test_pc.py  # interactive mode
        """,
    )
    p.add_argument("--video", "-v", nargs="+",
                   help="Path(s) to video file(s)")
    p.add_argument("--pose-model", type=str, default=str(POSE_MODEL_PATH),
                   help=f"Path to pose model (default: {POSE_MODEL_PATH.name})")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no display window)")

    # Desk zone options
    desk_grp = p.add_mutually_exclusive_group()
    desk_grp.add_argument("--zones", type=str, default=None,
                          help="Load desk zones from JSON file (skip calibration)")
    desk_grp.add_argument("--desk-ratio", type=float, default=None,
                          help="Use fixed bbox ratio instead of calibration (e.g. 0.70)")

    # Behavior thresholds
    p.add_argument("--head-angle", type=float, default=HEAD_TILT_ANGLE_DEG,
                   help=f"Head tilt angle threshold degrees (default: {HEAD_TILT_ANGLE_DEG})")
    p.add_argument("--head-sustain", type=float, default=HEAD_TILT_SUSTAIN_SEC,
                   help=f"Head tilt sustain seconds (default: {HEAD_TILT_SUSTAIN_SEC})")
    p.add_argument("--hands-sustain", type=float, default=HANDS_UNDER_SUSTAIN_SEC,
                   help=f"Hands under the table sustain seconds (default: {HANDS_UNDER_SUSTAIN_SEC})")
    p.add_argument("--reach-multiplier", type=float, default=PASSING_REACH_MULTIPLIER,
                   help=f"Passing papers: wrist reach as multiple of shoulder width (default: {PASSING_REACH_MULTIPLIER})")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve video paths
    video_paths = args.video if args.video else select_video_interactive()

    for vp in video_paths:
        if not os.path.isfile(vp):
            print(f"{TC.RED}[ERROR] Video not found: {vp}{TC.RESET}")
            sys.exit(1)

    # Validate model
    if not os.path.isfile(args.pose_model):
        model_name = Path(args.pose_model).name
        if model_name == args.pose_model and model_name.startswith("yolo") and model_name.endswith(".pt"):
            print(f"{TC.YELLOW}[INFO] Pose model will be resolved by Ultralytics: {args.pose_model}{TC.RESET}")
        else:
            print(f"{TC.RED}[ERROR] Pose model not found: {args.pose_model}{TC.RESET}")
            print(f"  Expected at: {POSE_MODEL_PATH}")
            sys.exit(1)

    # Validate tracker config
    if not BYTETRACK_CONFIG.exists():
        print(f"{TC.RED}[ERROR] ByteTrack config not found: {BYTETRACK_CONFIG}{TC.RESET}")
        print("  Expected at: tests/bytetrack_front.yaml")
        sys.exit(1)
    tracker_cfg = str(BYTETRACK_CONFIG)

    # ── Resolve desk zones ──────────────────────────────────
    desk_lines = None
    desk_ratio = HANDS_UNDER_DESK_RATIO

    if args.zones:
        # Load from file
        if not os.path.isfile(args.zones):
            print(f"{TC.RED}[ERROR] Zones file not found: {args.zones}{TC.RESET}")
            sys.exit(1)
        desk_lines = load_desk_zones(args.zones)

    elif args.desk_ratio is not None:
        # Use fixed ratio (no calibration)
        desk_ratio = args.desk_ratio
        log_info(f"Using fixed desk ratio: {desk_ratio:.0%} (no calibration)")

    else:
        # Interactive calibration for EACH video
        # (calibrate on first video, offer to reuse for subsequent)
        pass  # handled per-video below

    # Load model
    log_info(f"Loading pose model: {args.pose_model}")
    model = YOLO(args.pose_model)
    log_info("Pose model loaded.")

    # Print config
    print()
    print(f"{TC.BOLD}Configuration:{TC.RESET}")
    print(f"  Pose model       : {args.pose_model}")
    print(f"  Tracker           : ByteTrack ({BYTETRACK_CONFIG.name})")
    print(f"  Head tilt         : >{args.head_angle}deg for {args.head_sustain}s")
    if desk_lines:
        print(f"  Hands under the table: {len(desk_lines)} calibrated desk lines, sustain {args.hands_sustain}s")
    elif args.desk_ratio is not None:
        print(f"  Hands under the table: ratio {desk_ratio:.0%} (fallback), sustain {args.hands_sustain}s")
    else:
        print(f"  Hands under the table: calibration on first frame, sustain {args.hands_sustain}s")
    print(f"  Passing papers    : reach >{args.reach_multiplier}x shoulder width (immediate)")
    print(f"  Event cooldown    : {EVENT_COOLDOWN_SEC}s")
    print()

    for vp in video_paths:
        # Per-video calibration if no --zones or --desk-ratio
        vid_desk_lines = desk_lines
        if vid_desk_lines is None and args.desk_ratio is None:
            log_info(f"Starting desk zone calibration for: {Path(vp).name}")
            vid_desk_lines = calibrate_desk_zones(vp)

            if vid_desk_lines is None:
                log_info("Calibration skipped. Using ratio fallback.")
            elif len(vid_desk_lines) > 0:
                # Auto-save zones
                zones_path = SCRIPT_DIR / f"{Path(vp).stem}_desk_zones.json"
                save_desk_zones(vid_desk_lines, str(zones_path))

        process_video(
            vp, model, tracker_cfg,
            show_display=not args.no_display,
            desk_lines=vid_desk_lines,
            desk_ratio=desk_ratio,
            head_angle=args.head_angle,
            head_sustain=args.head_sustain,
            hands_sustain=args.hands_sustain,
            reach_mult=args.reach_multiplier,
        )

    log_info("All videos processed. Done!")


if __name__ == "__main__":
    main()
