#!/usr/bin/env python3
"""
Head Behavior Detection Test - PC Test Program
================================================
Detects two pose-based cheating behaviors from PROJECT.md:

  1. Head Tilting        - ear-to-ear angle > threshold, sustained
  2. Looking at Neighbor - nose offset from shoulder midpoint, sustained

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
HEAD_TILT_ANGLE_DEG = 30.0      # middle of 25-30 range from PROJECT.md
LOOK_NEIGHBOR_RATIO = 0.26      # nose offset / shoulder width
SUSTAINED_SEC = 3.0             # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0       # cooldown between repeated flags
KP_CONF_THRESH = 0.3            # minimum keypoint confidence

# ── Colors (BGR) ─────────────────────────────────────────────
COL_NORMAL = (0, 255, 0)
COL_UNASSIGNED = (128, 128, 128)
COL_SELECTED = (255, 255, 0)     # cyan
COL_HEAD_TILT = (0, 165, 255)    # orange
COL_LOOK_NEIGHBOR = (255, 0, 255)  # magenta
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

    # Looking at neighbor tracking
    look_neighbor_start: float = -1.0
    look_neighbor_flagged_at: float = -999.0

    def can_flag(self, behavior: str, now: float) -> bool:
        last = {
            "head_tilt": self.head_tilt_flagged_at,
            "looking_at_neighbor": self.look_neighbor_flagged_at,
        }.get(behavior, -999.0)
        return (now - last) > EVENT_COOLDOWN_SEC


# ── Behavior Detection ───────────────────────────────────────
def detect_head_tilt(kp_xy, kp_conf):
    """
    Ear-to-ear angle (PROJECT.md formula):
        angle = atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)
    Returns (is_tilted, angle_degrees).
    """
    if kp_conf[KP_LEFT_EAR] < KP_CONF_THRESH or kp_conf[KP_RIGHT_EAR] < KP_CONF_THRESH:
        return False, 0.0
    le = kp_xy[KP_LEFT_EAR]
    re = kp_xy[KP_RIGHT_EAR]
    raw = abs(math.degrees(
        math.atan2(float(re[1]) - float(le[1]),
                   float(re[0]) - float(le[0]))
    ))
    # Normalize: 0° = level head, 90° = fully sideways.
    # atan2 gives ~0 or ~180 for level ears depending on mirrored
    # ear ordering in the image, so map >90 back toward 0.
    angle = raw if raw <= 90 else 180 - raw
    return angle > HEAD_TILT_ANGLE_DEG, angle


def detect_looking_at_neighbor(kp_xy, kp_conf):
    """
    Nose offset from shoulder midpoint (PROJECT.md formula):
        offset = abs(nose.x - shoulder_center.x)
    Normalized by shoulder width for scale invariance.

    Compensates for body orientation: when a person's body is turned
    (one ear occluded), nose offset in the turn direction is natural
    posture, not looking at a neighbor.

    Returns (is_looking, offset_ratio, direction).
    """
    if kp_conf[KP_NOSE] < KP_CONF_THRESH:
        return False, 0.0, ""
    if (kp_conf[KP_LEFT_SHOULDER] < KP_CONF_THRESH or
            kp_conf[KP_RIGHT_SHOULDER] < KP_CONF_THRESH):
        return False, 0.0, ""

    nose_x = float(kp_xy[KP_NOSE][0])
    ls_x = float(kp_xy[KP_LEFT_SHOULDER][0])
    rs_x = float(kp_xy[KP_RIGHT_SHOULDER][0])

    shoulder_center_x = (ls_x + rs_x) / 2.0
    shoulder_width = abs(rs_x - ls_x)

    if shoulder_width < 5:  # too small / unreliable
        return False, 0.0, ""

    offset = nose_x - shoulder_center_x
    offset_ratio = abs(offset) / shoulder_width

    direction = "RIGHT" if offset > 0 else "LEFT"

    # ── Body-orientation compensation ──────────────────────────
    # When a person's body is angled to the camera, one ear becomes
    # occluded while the other stays visible.  The nose naturally
    # shifts toward the visible-ear side.  If the nose offset aligns
    # with this body turn, it's posture — not looking at a neighbor.
    left_ear_conf = float(kp_conf[KP_LEFT_EAR])
    right_ear_conf = float(kp_conf[KP_RIGHT_EAR])
    max_ear = max(left_ear_conf, right_ear_conf)
    min_ear = min(left_ear_conf, right_ear_conf)

    if max_ear > KP_CONF_THRESH:
        ear_symmetry = min_ear / max_ear  # 1.0 = both ears equally visible

        # Determine which way the body is turned based on ear visibility.
        # Low left-ear conf → body turned RIGHT (left ear occluded),
        # low right-ear conf → body turned LEFT  (right ear occluded).
        if ear_symmetry < 0.6:
            # Body is significantly turned to one side
            body_turn_dir = "RIGHT" if left_ear_conf < right_ear_conf else "LEFT"

            if direction == body_turn_dir:
                # Nose offset matches body orientation — this is natural
                # posture, not head-turning.  Suppress the detection.
                return False, offset_ratio, direction

    return offset_ratio > LOOK_NEIGHBOR_RATIO, offset_ratio, direction


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
    print(f"  Head tilt      : >{HEAD_TILT_ANGLE_DEG:.0f} deg, sustained {SUSTAINED_SEC}s")
    print(f"  Look neighbor  : >{LOOK_NEIGHBOR_RATIO:.0%} offset ratio, sustained {SUSTAINED_SEC}s")
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

                # ── 1. Looking at Neighbor ───────────────────
                is_looking, offset_ratio, direction = detect_looking_at_neighbor(
                    kp_xy, kp_conf)

                if is_looking:
                    if state.look_neighbor_start < 0:
                        state.look_neighbor_start = ts_sec
                    elapsed = ts_sec - state.look_neighbor_start

                    if elapsed >= SUSTAINED_SEC and state.can_flag("looking_at_neighbor", ts_sec):
                        state.look_neighbor_flagged_at = ts_sec
                        stats["looking_at_neighbor"] += 1
                        total_alerts += 1
                        log_alert("LOOKING AT NEIGHBOR", state.student_num, ts_sec,
                                  f"direction={direction}, offset={offset_ratio:.0%}, "
                                  f"sustained {elapsed:.1f}s",
                                  TC.MAGENTA)
                        frame_events.append(("looking_at_neighbor", state.student_num))

                    if elapsed >= 1.0:
                        behavior_labels.append(
                            f"LOOKING {direction} ({elapsed:.1f}s)")
                        if box_color == COL_NORMAL:
                            box_color = COL_LOOK_NEIGHBOR
                        if elapsed >= SUSTAINED_SEC:
                            box_color = COL_FLAGGED

                    # Draw nose-to-shoulder-center line
                    if (kp_conf[KP_NOSE] > KP_CONF_THRESH and
                            kp_conf[KP_LEFT_SHOULDER] > KP_CONF_THRESH and
                            kp_conf[KP_RIGHT_SHOULDER] > KP_CONF_THRESH):
                        nose = (int(kp_xy[KP_NOSE][0]), int(kp_xy[KP_NOSE][1]))
                        sc_x = int((kp_xy[KP_LEFT_SHOULDER][0] + kp_xy[KP_RIGHT_SHOULDER][0]) / 2)
                        sc_y = int((kp_xy[KP_LEFT_SHOULDER][1] + kp_xy[KP_RIGHT_SHOULDER][1]) / 2)
                        cv2.line(annotated, nose, (sc_x, sc_y), COL_LOOK_NEIGHBOR, 2, cv2.LINE_AA)
                else:
                    state.look_neighbor_start = -1.0

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
    print("  Detects: Head Tilting | Looking at Neighbor")
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
