#!/usr/bin/env python3
"""
Head Behavior Detection Test - PC Test Program
================================================
Detects two pose-based cheating behaviors:

  1. Head Tilting            - ear-to-ear angle > threshold, sustained
  2. Shoulder Turn (OVERHEAD)- shoulder-line angle deviation, sustained

Both behaviors must be sustained for 4 seconds (configurable) before
triggering an alert and saving an evidence screenshot.

Workflow:
  1. File dialog opens to select a video file
  2. First frame shown with detected persons - click to assign student numbers
  3. Live detection window with annotations and console alerts
  4. Evidence screenshots saved to ./evidence/ on sustained detection

Controls:
  Assignment phase:
    Left-click on person  -> select person (highlighted in cyan)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start detection (need >= 1 assignment)
    ESC                   -> quit

  Detection phase:
    q / ESC  - quit
    SPACE    - pause / resume

Requirements:
    pip install ultralytics opencv-python numpy lap
"""

import sys
import os
import math
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

POSE_MODEL_PATH = REPO_ROOT / "yolo26s-pose.pt"
BYTETRACK_CONFIG = SCRIPT_DIR / "bytetrack_front.yaml"
EVIDENCE_DIR = SCRIPT_DIR / "evidence"

# ── COCO 17-Keypoint Indices ────────────────────────────────
KP_NOSE = 0
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# ── Behavior Thresholds ─────────────────────────────────────
HEAD_TILT_ANGLE_DEG = 30.0      # ear-to-ear roll angle threshold
HEAD_TURN_RATIO = 0.26          # nose offset / shoulder width threshold for yaw detection
SHOULDER_TURN_ANGLE_DEG = 20.0  # shoulder-line deviation from horizontal (overhead cam)
SUSTAINED_SEC = 3.0             # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0       # cooldown between repeated flags
KP_CONF_THRESH = 0.3            # minimum keypoint confidence

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
    for k in range(min(len(kp_xy), 13)):  # draw upper body keypoints
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


# ── Per-Student State ────────────────────────────────────────
@dataclass
class StudentState:
    track_id: int
    student_num: int

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


# ── Behavior Detection ───────────────────────────────────────
def detect_head_tilt(kp_xy, kp_conf):
    """
    Detects head tilting via two complementary signals:

    1. Roll (sideways lean): ear-to-ear angle vs horizontal.
       Triggers when angle > HEAD_TILT_ANGLE_DEG.

    2. Yaw (turning left/right): nose offset from shoulder midpoint,
       normalized by shoulder width.  Triggers when ratio > HEAD_TURN_RATIO.
       When the head turns, one ear gets occluded by the head so the
       ear-to-ear angle alone misses this motion.

    Returns (is_tilted, score) where score is the higher of the two
    normalized signals (0.0 = neutral, 1.0 = at threshold, >1.0 = exceeded).
    """
    roll_score = 0.0
    yaw_score = 0.0

    # ── Roll detection (ear-to-ear angle) ──────────────────────
    if (kp_conf[KP_LEFT_EAR] >= KP_CONF_THRESH and
            kp_conf[KP_RIGHT_EAR] >= KP_CONF_THRESH):
        le = kp_xy[KP_LEFT_EAR]
        re = kp_xy[KP_RIGHT_EAR]
        raw = abs(math.degrees(
            math.atan2(float(re[1]) - float(le[1]),
                       float(re[0]) - float(le[0]))
        ))
        angle = raw if raw <= 90 else 180 - raw
        roll_score = angle / HEAD_TILT_ANGLE_DEG if HEAD_TILT_ANGLE_DEG > 0 else 0.0

    # ── Yaw detection (nose offset from shoulder center) ───────
    if (kp_conf[KP_NOSE] >= KP_CONF_THRESH and
            kp_conf[KP_LEFT_SHOULDER] >= KP_CONF_THRESH and
            kp_conf[KP_RIGHT_SHOULDER] >= KP_CONF_THRESH):
        nose_x = float(kp_xy[KP_NOSE][0])
        ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
        rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])
        shoulder_width = abs(rs_x - ls_x)
        if shoulder_width >= 5:
            shoulder_center_x = (ls_x + rs_x) / 2.0
            offset_ratio = abs(nose_x - shoulder_center_x) / shoulder_width
            yaw_score = offset_ratio / HEAD_TURN_RATIO if HEAD_TURN_RATIO > 0 else 0.0

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


# ── File Dialog ──────────────────────────────────────────────
def select_video_dialog():
    """Open a file dialog to select a video file."""
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


# ── Assignment Phase ─────────────────────────────────────────
def run_assignment_phase(first_frame, initial_results, disp_scale):
    """
    Interactive student number assignment on the first frame.
    Returns student_map: {track_id: student_number} or None if cancelled.
    """
    boxes = initial_results[0].boxes
    keypoints = initial_results[0].keypoints

    has_tracks = (
        boxes is not None
        and boxes.id is not None
        and keypoints is not None
        and len(boxes) > 0
    )

    if not has_tracks:
        log_info("No persons detected in the first frame.")
        log_info("Press any key to proceed without assignments (or ESC to quit).")
        cv2.imshow("AISENTINEL - Assign Students", first_frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("AISENTINEL - Assign Students")
        if key == 27:
            return None
        return {}

    track_ids = boxes.id.int().cpu().tolist()
    bboxes = boxes.xyxy.cpu().numpy()
    kps_xy = keypoints.xy.cpu().numpy()
    kps_conf = keypoints.conf.cpu().numpy()

    # Build person data
    persons = []
    for i, tid in enumerate(track_ids):
        persons.append({
            "track_id": tid,
            "bbox": tuple(bboxes[i].tolist()),
            "kp_xy": kps_xy[i],
            "kp_conf": kps_conf[i],
        })

    # State
    student_map = {}         # track_id -> student_number
    selected_idx = -1        # index in persons list (-1 = none)
    input_buffer = ""        # digits being typed

    fh, fw = first_frame.shape[:2]
    win_name = "AISENTINEL - Assign Students"

    def on_mouse(event, mx, my, flags, param):
        nonlocal selected_idx, input_buffer
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Convert display coords to original frame coords
        ox = int(mx / disp_scale)
        oy = int(my / disp_scale)
        # Find which person was clicked
        for i, p in enumerate(persons):
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                selected_idx = i
                input_buffer = ""
                return
        # Clicked outside any person — deselect
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

        # Draw all persons
        for i, p in enumerate(persons):
            tid = p["track_id"]
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]

            # Determine color and label
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

            # Draw skeleton
            draw_skeleton(display, p["kp_xy"], p["kp_conf"],
                          color=COL_SELECTED if i == selected_idx else (255, 255, 0))

            # Label
            if tid in student_map:
                label = f"Student #{student_map[tid]}"
                draw_label(display, label, x1, y1 - 2, COL_NORMAL)
            else:
                label = f"[unassigned] (ID:{tid})"
                draw_label(display, label, x1, y1 - 2, COL_UNASSIGNED)

            # Show input buffer on selected person
            if i == selected_idx and input_buffer:
                buf_label = f"Typing: {input_buffer}_"
                draw_label(display, buf_label, x1, y2 + 18, COL_SELECTED, (0, 0, 0))
            elif i == selected_idx:
                draw_label(display, "Selected - type a number", x1, y2 + 18,
                           COL_SELECTED, (0, 0, 0))

        # Instructions overlay
        for i, txt in enumerate(instructions):
            y = 30 + i * 28
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Assignment count
        assigned = len(student_map)
        total = len(persons)
        status = f"Assigned: {assigned}/{total} persons"
        cv2.putText(display, status, (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_NORMAL if assigned > 0 else COL_UNASSIGNED, 2)

        # Scale for display
        if disp_scale < 1.0:
            show = cv2.resize(display, (int(fw * disp_scale), int(fh * disp_scale)))
        else:
            show = display

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win_name)
            return None

        elif key in (ord("s"), ord("S")):
            if len(student_map) == 0:
                log_info("Assign at least one student before starting.")
                continue
            break

        elif key == 13:  # ENTER — assign
            if selected_idx >= 0 and input_buffer:
                num = int(input_buffer)
                tid = persons[selected_idx]["track_id"]
                # Check for duplicate student numbers
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


# ── Main Detection Loop ──────────────────────────────────────
def run_detection(cap, model, tracker_cfg, student_map, video_path):
    """
    Run the live detection window with behavior analysis.
    Expects cap already open (positioned at frame 2+) and model.track()
    state preserved from the assignment phase so track IDs stay consistent.
    """
    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    print()
    print("=" * 70)
    print(f"  AISENTINEL - Head Behavior Detection")
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Students : {len(student_map)} assigned")
    print(f"  Head tilt roll : >{HEAD_TILT_ANGLE_DEG:.0f} deg (ear-to-ear), sustained {SUSTAINED_SEC}s")
    print(f"  Head tilt yaw  : >{HEAD_TURN_RATIO:.0%} offset ratio (nose/shoulder), sustained {SUSTAINED_SEC}s")
    print(f"  Shoulder turn  : >{SHOULDER_TURN_ANGLE_DEG:.0f} deg (overhead), sustained {SUSTAINED_SEC}s")
    print(f"  Cooldown       : {EVENT_COOLDOWN_SEC}s between repeated flags")
    print(f"  Evidence dir   : {EVIDENCE_DIR}")
    print("=" * 70)
    print()

    # Build student states from the assignment map
    students: dict[int, StudentState] = {}
    for tid, num in student_map.items():
        students[tid] = StudentState(track_id=tid, student_num=num)

    frame_idx = 1  # frame 1 was already processed during assignment
    paused = False
    stats = defaultdict(int)
    total_alerts = 0
    win_name = "AISENTINEL - Head Behavior Detection"

    while True:
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(" "):
                paused = False
                log_info("Resumed.")
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
        keypoints_data = results[0].keypoints
        frame_events = []  # (behavior, student_num) for screenshots

        has_tracks = (
            boxes is not None
            and boxes.id is not None
            and keypoints_data is not None
            and len(boxes) > 0
        )

        if has_tracks:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.cpu().numpy()
            kps_xy = keypoints_data.xy.cpu().numpy()
            kps_conf = keypoints_data.conf.cpu().numpy()

            for i, tid in enumerate(track_ids):
                bbox = tuple(bboxes[i].tolist())
                kp_xy = kps_xy[i]
                kp_conf = kps_conf[i]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Check if this track is an assigned student
                is_assigned = tid in students
                if not is_assigned:
                    # Draw unassigned person dimly
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_UNASSIGNED, 1)
                    draw_label(annotated, f"ID:{tid}", x1, y1, COL_UNASSIGNED)
                    continue

                state = students[tid]
                box_color = COL_NORMAL
                behavior_labels = []

                # Draw skeleton
                draw_skeleton(annotated, kp_xy, kp_conf)

                # ── 1. Head Tilt (roll + yaw) ────────────────
                is_tilted, tilt_score = detect_head_tilt(kp_xy, kp_conf)

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

                # ── Draw person box + student label ──────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                draw_label(annotated, f"Student #{state.student_num}",
                           x1, y1 - 2, box_color)

                # Behavior labels below the box
                lbl_y = y1 + 18
                for bl in behavior_labels:
                    draw_label(annotated, bl, x1, lbl_y, box_color)
                    lbl_y += 18

        # ── HUD ──────────────────────────────────────────────
        n_tracked = len(track_ids) if has_tracks else 0
        hud_lines = [
            f"Frame: {frame_idx}/{total_frames} | Time: {fmt_ts(ts_sec)}",
            f"Tracked: {n_tracked} | Assigned: {len(students)} | Alerts: {total_alerts}",
        ]
        for i, line in enumerate(hud_lines):
            y_pos = 25 + i * 28
            cv2.putText(annotated, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COL_FLAGGED if total_alerts else COL_NORMAL,
                        2, cv2.LINE_AA)

        # ── Alert banner on frame when events fire ───────────
        if frame_events:
            banner_y = h - 40
            for behavior, snum in frame_events:
                txt = f"ALERT: Student #{snum} - {behavior.replace('_', ' ').upper()}"
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_FLAGGED, 2, cv2.LINE_AA)
                banner_y -= 35

        # ── Save evidence screenshots ────────────────────────
        for behavior, snum in frame_events:
            save_evidence(annotated, snum, behavior, ts_sec)

        # ── Display ──────────────────────────────────────────
        disp = annotated
        if disp_scale < 1.0:
            disp = cv2.resize(annotated,
                              (int(w * disp_scale), int(h * disp_scale)))
        cv2.imshow(win_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            log_info("Quit requested.")
            break
        elif key == ord(" "):
            paused = True
            log_info("Paused. Press SPACE to resume.")

        # Progress
        if frame_idx % 500 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            log_info(f"Progress: {pct:.1f}% ({frame_idx}/{total_frames})")

    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  Summary: {Path(video_path).name}")
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


# ── Main ─────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Head Behavior Detection Test (PC)")
    print("  Detects: Head Tilting | Shoulder Turn")
    print("=" * 60)
    print()

    # ── Select video via file dialog ─────────────────────────
    log_info("Opening file dialog...")
    video_path = select_video_dialog()
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Validate tracker config ──────────────────────────────
    if BYTETRACK_CONFIG.exists():
        tracker_cfg = str(BYTETRACK_CONFIG)
        log_info(f"Using ByteTrack config: {BYTETRACK_CONFIG.name}")
    else:
        tracker_cfg = "bytetrack.yaml"
        log_info("Custom ByteTrack config not found, using ultralytics default.")

    # ── Load pose model ──────────────────────────────────────
    if not POSE_MODEL_PATH.exists():
        log_info(f"Pose model not found at: {POSE_MODEL_PATH}")
        log_info("Ultralytics will auto-download yolo11n-pose.pt on first use.")
    log_info(f"Loading pose model: {POSE_MODEL_PATH.name}")
    model = YOLO(str(POSE_MODEL_PATH))
    log_info("Pose model loaded.")

    # ── Open video & read first frame ──────────────────────────
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

    # Run tracking on first frame to get initial track IDs.
    # persist=True keeps tracker state alive for the detection phase.
    initial_results = model.track(
        first_frame, persist=True, tracker=tracker_cfg,
        verbose=False, imgsz=640,
    )

    n_detected = 0
    if (initial_results[0].boxes is not None and
            initial_results[0].boxes.id is not None):
        n_detected = len(initial_results[0].boxes)
    log_info(f"Detected {n_detected} persons. Opening assignment window...")
    print()
    print(f"  {TC.BOLD}Instructions:{TC.RESET}")
    print(f"    1. Click on a person to select them (cyan highlight)")
    print(f"    2. Type the student number (digits)")
    print(f"    3. Press ENTER to assign")
    print(f"    4. Repeat for each student you want to monitor")
    print(f"    5. Press S to start detection")
    print()

    # ── Assignment phase ─────────────────────────────────────
    student_map = run_assignment_phase(first_frame, initial_results, disp_scale)
    if student_map is None:
        cap.release()
        log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        log_info("No students assigned. Exiting.")
        sys.exit(0)

    # ── Run detection ────────────────────────────────────────
    # Cap is still open at frame 2. Tracker state from model.track()
    # persists so track IDs assigned during assignment stay consistent.
    log_info("Starting detection...")
    run_detection(cap, model, tracker_cfg, student_map, video_path)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
