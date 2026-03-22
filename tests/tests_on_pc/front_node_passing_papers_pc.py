#!/usr/bin/env python3
"""
Passing Papers Detection Test - PC Test Program
=================================================
Detects when a student passes a paper/note to a side-by-side neighbor
by monitoring wrist keypoint lateral exit from the student's person
bounding box.

Algorithm (multi-signal interaction detection):
  1. Track students with ByteTrack (person bboxes + persistent IDs)
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
  2. First frame shown with detected persons - click to assign student numbers
  3. Live detection window with annotations and console alerts
  4. Evidence screenshots saved to ./evidence_passing/ on sustained detection

Controls:
  Assignment phase:
    Left-click on person  -> select person (highlighted in cyan)
    0-9 keys              -> type student number
    ENTER                 -> assign number to selected person
    BACKSPACE             -> delete last digit
    S                     -> start detection (need >= 2 assignments)
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
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

# -- Paths --------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

POSE_MODEL_PATH = REPO_ROOT / "yolo26s-pose.pt"
BYTETRACK_CONFIG = SCRIPT_DIR / "bytetrack_front.yaml"
EVIDENCE_DIR = SCRIPT_DIR / "evidence_passing"

# -- COCO 17-Keypoint Indices ------------------------------------
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

# -- Behavior Thresholds -----------------------------------------
EVENT_COOLDOWN_SEC = 10.0       # cooldown between repeated flags for same student pair
KP_CONF_THRESH = 0.3            # minimum keypoint confidence
ROW_TOLERANCE_PX = 80           # max vertical (y-center) difference to be in same "row" (at reference scale)
REFERENCE_BBOX_HEIGHT = 300.0   # approx bbox height (px) for front-row student; thresholds scale relative to this

# -- Multi-Signal Interaction Thresholds --------------------------
ARM_EXTENSION_RATIO = 1.2       # shoulder-wrist / shoulder-hip ratio to count as "extended"
WRIST_PROXIMITY_PX = 120        # max distance between wrists of two students for proximity
WRIST_VELOCITY_TOWARD_THRESH = 3.0  # min px/frame wrist must move toward neighbor
MIN_INTERACTION_SEC = 0.4       # minimum proximity duration to trigger alert
MAX_INTERACTION_SEC = 4.0       # max duration — interactions longer than this are not passing
INTERACTION_SIGNAL_THRESH = 2   # how many signals (of 3) must be active to start tracking
PROXIMITY_HISTORY_SEC = 1.0     # time window to track proximity for temporal pattern

# -- Colors (BGR) -------------------------------------------------
COL_NORMAL = (0, 255, 0)        # green
COL_UNASSIGNED = (128, 128, 128)
COL_SELECTED = (255, 255, 0)    # cyan
COL_WARNING = (0, 165, 255)     # orange - wrist exiting
COL_FLAGGED = (0, 0, 255)       # red - confirmed passing
COL_WRIST = (255, 0, 255)       # magenta - wrist keypoints
COL_EXIT_LINE = (0, 0, 255)     # red - exit direction line
COL_NEIGHBOR_LINE = (255, 100, 0)  # blue - line to neighbor
COL_HUD = (0, 255, 0)

# -- Skeleton for drawing ----------------------------------------
SKELETON = [
    (KP_NOSE, 1), (KP_NOSE, 2),
    (1, KP_LEFT_EAR), (2, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, 7), (KP_RIGHT_SHOULDER, 8),
    (7, KP_LEFT_WRIST), (8, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, 11), (KP_RIGHT_SHOULDER, 12),
    (11, 12),
]


# -- Terminal helpers ---------------------------------------------
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


# -- Drawing helpers ----------------------------------------------
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


# -- File Dialog --------------------------------------------------
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


# -- Per-Student State --------------------------------------------
@dataclass
class StudentState:
    track_id: int
    student_num: int

    # Previous frame wrist positions for velocity computation
    prev_left_wrist: tuple = None   # (x, y) or None
    prev_right_wrist: tuple = None  # (x, y) or None

    # Current frame computed signals (reset each frame)
    left_arm_extended: bool = False
    right_arm_extended: bool = False
    left_arm_ratio: float = 0.0
    right_arm_ratio: float = 0.0
    left_wrist_velocity: tuple = (0.0, 0.0)  # (vx, vy)
    right_wrist_velocity: tuple = (0.0, 0.0)

    total_alerts: int = 0


@dataclass
class PairInteractionState:
    """Tracks the interaction state between a pair of students."""
    tid_a: int
    tid_b: int

    # Temporal tracking
    interaction_start: float = -1.0     # when signals first met threshold
    last_proximity_time: float = -1.0   # last frame with wrist proximity
    last_signal_time: float = -1.0      # last frame with >= INTERACTION_SIGNAL_THRESH signals

    # Signal history for temporal pattern
    had_arm_extension: bool = False      # arm extension seen during this interaction
    had_approach: bool = False           # wrist moving toward neighbor seen
    had_proximity: bool = False          # wrist-to-wrist proximity seen
    peak_proximity_dist: float = 9999.0  # closest wrist distance during interaction

    # Cooldown
    last_flagged_at: float = -999.0

    # Active signals this frame (reset each frame)
    frame_arm_ext: bool = False
    frame_approach: bool = False
    frame_proximity: bool = False
    frame_proximity_dist: float = 9999.0

    def can_flag(self, now: float) -> bool:
        return (now - self.last_flagged_at) > EVENT_COOLDOWN_SEC

    def reset_interaction(self):
        self.interaction_start = -1.0
        self.last_proximity_time = -1.0
        self.last_signal_time = -1.0
        self.had_arm_extension = False
        self.had_approach = False
        self.had_proximity = False
        self.peak_proximity_dist = 9999.0

    def active_signal_count(self) -> int:
        return sum([self.frame_arm_ext, self.frame_approach, self.frame_proximity])


# -- Geometry helpers ---------------------------------------------
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


# -- Signal 1: Arm Extension Detection ----------------------------
def compute_arm_extension(kp_xy, kp_conf):
    """
    Compute arm extension ratio for both arms.
    ratio = dist(shoulder, wrist) / dist(shoulder, hip)
    Returns (left_ratio, right_ratio). 0.0 if keypoints unavailable.
    """
    left_ratio = 0.0
    right_ratio = 0.0

    # Left arm: shoulder(5) -> wrist(9), shoulder(5) -> hip(11)
    l_sh = _kp_pos(kp_xy, kp_conf, KP_LEFT_SHOULDER)
    l_wr = _kp_pos(kp_xy, kp_conf, KP_LEFT_WRIST)
    l_hp = _kp_pos(kp_xy, kp_conf, KP_LEFT_HIP)
    if l_sh and l_wr and l_hp:
        sh_hp = _dist(l_sh, l_hp)
        if sh_hp > 1.0:  # avoid division by zero
            left_ratio = _dist(l_sh, l_wr) / sh_hp

    # Right arm: shoulder(6) -> wrist(10), shoulder(6) -> hip(12)
    r_sh = _kp_pos(kp_xy, kp_conf, KP_RIGHT_SHOULDER)
    r_wr = _kp_pos(kp_xy, kp_conf, KP_RIGHT_WRIST)
    r_hp = _kp_pos(kp_xy, kp_conf, KP_RIGHT_HIP)
    if r_sh and r_wr and r_hp:
        sh_hp = _dist(r_sh, r_hp)
        if sh_hp > 1.0:
            right_ratio = _dist(r_sh, r_wr) / sh_hp

    return left_ratio, right_ratio


# -- Signal 2: Wrist Velocity Toward Neighbor ---------------------
def wrist_moves_toward(wrist_pos, wrist_vel, neighbor_center,
                       speed_thresh=WRIST_VELOCITY_TOWARD_THRESH):
    """
    Check if the wrist velocity vector points toward the neighbor.
    Returns True if the wrist is moving toward neighbor_center.
    speed_thresh can be scaled for perspective-adaptive detection.
    """
    if wrist_pos is None or wrist_vel is None:
        return False
    vx, vy = wrist_vel
    speed = math.sqrt(vx * vx + vy * vy)
    if speed < speed_thresh:
        return False
    # Direction from wrist to neighbor
    dx = neighbor_center[0] - wrist_pos[0]
    dy = neighbor_center[1] - wrist_pos[1]
    d = math.sqrt(dx * dx + dy * dy)
    if d < 1.0:
        return False
    # Dot product of velocity and direction-to-neighbor (normalized)
    dot = (vx * dx + vy * dy) / (speed * d)
    return dot > 0.3  # cos(~72°) — reasonably toward


# -- Signal 3: Wrist-to-Wrist Proximity --------------------------
def compute_wrist_proximity(kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf):
    """
    Compute the minimum distance between wrist keypoints of two students.
    Checks: A.right_wrist <-> B.left_wrist  and  A.left_wrist <-> B.right_wrist
    Returns the minimum distance (float), or 9999.0 if keypoints unavailable.
    """
    min_dist = 9999.0

    # A.right_wrist <-> B.left_wrist (passing to the right)
    a_rw = _kp_pos(kp_a_xy, kp_a_conf, KP_RIGHT_WRIST)
    b_lw = _kp_pos(kp_b_xy, kp_b_conf, KP_LEFT_WRIST)
    if a_rw and b_lw:
        min_dist = min(min_dist, _dist(a_rw, b_lw))

    # A.left_wrist <-> B.right_wrist (passing to the left)
    a_lw = _kp_pos(kp_a_xy, kp_a_conf, KP_LEFT_WRIST)
    b_rw = _kp_pos(kp_b_xy, kp_b_conf, KP_RIGHT_WRIST)
    if a_lw and b_rw:
        min_dist = min(min_dist, _dist(a_lw, b_rw))

    return min_dist


# -- Neighbor finding (row-filtered, both directions) -------------
def find_row_neighbors(source_tid, all_student_bboxes, students):
    """
    Find all assigned students in the same row as source_tid.
    Returns list of (neighbor_tid, direction) where direction is "LEFT" or "RIGHT".
    """
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

    # Sort by horizontal distance (closest first)
    neighbors.sort(key=lambda x: abs(
        (all_student_bboxes[x[0]][0] + all_student_bboxes[x[0]][2]) / 2 - source_cx))
    return neighbors


# -- Assignment Phase ---------------------------------------------
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

    persons = []
    for i, tid in enumerate(track_ids):
        persons.append({
            "track_id": tid,
            "bbox": tuple(bboxes[i].tolist()),
            "kp_xy": kps_xy[i],
            "kp_conf": kps_conf[i],
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


# -- Main Detection Loop ------------------------------------------
def run_detection(cap, model, tracker_cfg, student_map, video_path):
    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    print()
    print("=" * 70)
    print(f"  AISENTINEL - Passing Papers Detection (Multi-Signal)")
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Students : {len(student_map)} assigned")
    print(f"  Arm extension ratio : {ARM_EXTENSION_RATIO}")
    print(f"  Wrist proximity     : {WRIST_PROXIMITY_PX}px (at ref scale)")
    print(f"  Interaction duration: {MIN_INTERACTION_SEC}-{MAX_INTERACTION_SEC}s")
    print(f"  Signal threshold    : {INTERACTION_SIGNAL_THRESH} of 3 signals")
    print(f"  Row tolerance       : {ROW_TOLERANCE_PX}px (perspective-scaled)")
    print(f"  Reference bbox h    : {REFERENCE_BBOX_HEIGHT}px")
    print(f"  Cooldown            : {EVENT_COOLDOWN_SEC}s between repeated flags")
    print(f"  Evidence dir        : {EVIDENCE_DIR}")
    print("=" * 70)
    print()

    # Build student states
    students: dict[int, StudentState] = {}
    for tid, num in student_map.items():
        students[tid] = StudentState(track_id=tid, student_num=num)

    # Per-pair interaction state, keyed by frozenset(tid_a, tid_b)
    pair_states: dict[frozenset, PairInteractionState] = {}

    def get_pair_state(tid_a, tid_b):
        key = frozenset((tid_a, tid_b))
        if key not in pair_states:
            pair_states[key] = PairInteractionState(tid_a=tid_a, tid_b=tid_b)
        return pair_states[key]

    # Per-student keypoint data for current frame (populated during first pass)
    frame_kp_data = {}  # tid -> (kp_xy, kp_conf)

    frame_idx = 1
    paused = False
    total_alerts = 0
    win_name = "AISENTINEL - Passing Papers Detection"

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

        # -- Pose + tracking --------------------------------------
        results = model.track(
            frame, persist=True, tracker=tracker_cfg,
            verbose=False, imgsz=640,
        )

        annotated = frame.copy()
        boxes = results[0].boxes
        keypoints_data = results[0].keypoints
        frame_events = []  # list of (student_num_source, student_num_neighbor, direction)

        has_tracks = (
            boxes is not None
            and boxes.id is not None
            and keypoints_data is not None
            and len(boxes) > 0
        )

        # Collect all tracked student bboxes for neighbor lookup
        all_student_bboxes = {}  # track_id -> (x1, y1, x2, y2)
        frame_kp_data.clear()
        per_student_labels = {}   # tid -> (box_color, [labels])

        if has_tracks:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.cpu().numpy()
            kps_xy = keypoints_data.xy.cpu().numpy()
            kps_conf = keypoints_data.conf.cpu().numpy()

            # ---- First pass: collect data + compute per-student signals ----
            for i, tid in enumerate(track_ids):
                bbox = tuple(bboxes[i].tolist())
                kp_xy = kps_xy[i]
                kp_conf = kps_conf[i]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if tid not in students:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_UNASSIGNED, 1)
                    draw_label(annotated, f"ID:{tid}", x1, y1, COL_UNASSIGNED)
                    continue

                all_student_bboxes[tid] = bbox
                frame_kp_data[tid] = (kp_xy, kp_conf)
                state = students[tid]
                per_student_labels[tid] = [COL_NORMAL, []]  # [box_color, labels]

                # Draw skeleton and wrist keypoints
                draw_skeleton(annotated, kp_xy, kp_conf)
                for kp_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
                    if kp_idx < len(kp_conf) and kp_conf[kp_idx] > KP_CONF_THRESH:
                        wx, wy = int(kp_xy[kp_idx][0]), int(kp_xy[kp_idx][1])
                        cv2.circle(annotated, (wx, wy), 5, COL_WRIST, -1, cv2.LINE_AA)

                # Signal 1: Arm extension
                left_ratio, right_ratio = compute_arm_extension(kp_xy, kp_conf)
                state.left_arm_extended = left_ratio >= ARM_EXTENSION_RATIO
                state.right_arm_extended = right_ratio >= ARM_EXTENSION_RATIO
                state.left_arm_ratio = left_ratio
                state.right_arm_ratio = right_ratio

                # Draw arm extension indicator
                if state.left_arm_extended:
                    l_sh = _kp_pos(kp_xy, kp_conf, KP_LEFT_SHOULDER)
                    l_wr = _kp_pos(kp_xy, kp_conf, KP_LEFT_WRIST)
                    if l_sh and l_wr:
                        cv2.arrowedLine(annotated,
                                        (int(l_sh[0]), int(l_sh[1])),
                                        (int(l_wr[0]), int(l_wr[1])),
                                        COL_WARNING, 2, tipLength=0.2)
                    per_student_labels[tid][1].append(
                        f"L-ARM EXT {left_ratio:.2f}")
                if state.right_arm_extended:
                    r_sh = _kp_pos(kp_xy, kp_conf, KP_RIGHT_SHOULDER)
                    r_wr = _kp_pos(kp_xy, kp_conf, KP_RIGHT_WRIST)
                    if r_sh and r_wr:
                        cv2.arrowedLine(annotated,
                                        (int(r_sh[0]), int(r_sh[1])),
                                        (int(r_wr[0]), int(r_wr[1])),
                                        COL_WARNING, 2, tipLength=0.2)
                    per_student_labels[tid][1].append(
                        f"R-ARM EXT {right_ratio:.2f}")

                # Signal 2: Wrist velocity (compare to previous frame)
                cur_lw = _kp_pos(kp_xy, kp_conf, KP_LEFT_WRIST)
                cur_rw = _kp_pos(kp_xy, kp_conf, KP_RIGHT_WRIST)

                if cur_lw and state.prev_left_wrist:
                    state.left_wrist_velocity = (
                        cur_lw[0] - state.prev_left_wrist[0],
                        cur_lw[1] - state.prev_left_wrist[1],
                    )
                else:
                    state.left_wrist_velocity = (0.0, 0.0)

                if cur_rw and state.prev_right_wrist:
                    state.right_wrist_velocity = (
                        cur_rw[0] - state.prev_right_wrist[0],
                        cur_rw[1] - state.prev_right_wrist[1],
                    )
                else:
                    state.right_wrist_velocity = (0.0, 0.0)

                # Draw velocity arrows on wrists
                for wpos, wvel in [(cur_lw, state.left_wrist_velocity),
                                   (cur_rw, state.right_wrist_velocity)]:
                    if wpos is not None:
                        speed = math.sqrt(wvel[0]**2 + wvel[1]**2)
                        if speed > WRIST_VELOCITY_TOWARD_THRESH:
                            scale = min(speed, 30.0)  # cap arrow length
                            ex = int(wpos[0] + wvel[0] / speed * scale * 2)
                            ey = int(wpos[1] + wvel[1] / speed * scale * 2)
                            cv2.arrowedLine(annotated,
                                            (int(wpos[0]), int(wpos[1])),
                                            (ex, ey),
                                            (0, 200, 255), 2, tipLength=0.4)

                # Update previous positions for next frame
                state.prev_left_wrist = cur_lw
                state.prev_right_wrist = cur_rw

            # ---- Second pass: evaluate pair interactions ----
            evaluated_pairs = set()  # avoid double-evaluating (A,B) and (B,A)

            for tid_a in list(frame_kp_data.keys()):
                neighbors = find_row_neighbors(tid_a, all_student_bboxes, students)

                for tid_b, direction in neighbors:
                    pair_key = frozenset((tid_a, tid_b))
                    if pair_key in evaluated_pairs:
                        continue
                    evaluated_pairs.add(pair_key)

                    if tid_b not in frame_kp_data:
                        continue

                    ps = get_pair_state(tid_a, tid_b)
                    state_a = students[tid_a]
                    state_b = students[tid_b]
                    kp_a_xy, kp_a_conf = frame_kp_data[tid_a]
                    kp_b_xy, kp_b_conf = frame_kp_data[tid_b]

                    bbox_b = all_student_bboxes[tid_b]
                    bbox_a = all_student_bboxes[tid_a]
                    center_b = ((bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2)
                    center_a = ((bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2)

                    # Perspective-adaptive thresholds based on average bbox height
                    avg_pair_h = ((bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])) / 2.0
                    pscale = _perspective_scale(avg_pair_h)
                    scaled_prox_px = WRIST_PROXIMITY_PX * pscale
                    scaled_vel_thresh = WRIST_VELOCITY_TOWARD_THRESH * pscale

                    # Reset per-frame signals
                    ps.frame_arm_ext = False
                    ps.frame_approach = False
                    ps.frame_proximity = False
                    ps.frame_proximity_dist = 9999.0

                    # -- Signal 1: Arm extension toward the other student --
                    # A extends toward B (B is to A's right -> A's right arm, etc.)
                    a_extends_toward_b = False
                    b_extends_toward_a = False
                    if center_b[0] > center_a[0]:
                        # B is to A's right
                        a_extends_toward_b = state_a.right_arm_extended
                        b_extends_toward_a = state_b.left_arm_extended
                    else:
                        a_extends_toward_b = state_a.left_arm_extended
                        b_extends_toward_a = state_b.right_arm_extended

                    ps.frame_arm_ext = a_extends_toward_b or b_extends_toward_a

                    # -- Signal 2: Wrist velocity toward the other student --
                    a_moves_toward_b = False
                    b_moves_toward_a = False

                    # Check A's wrists moving toward B
                    lw_a = _kp_pos(kp_a_xy, kp_a_conf, KP_LEFT_WRIST)
                    rw_a = _kp_pos(kp_a_xy, kp_a_conf, KP_RIGHT_WRIST)
                    if wrist_moves_toward(rw_a, state_a.right_wrist_velocity, center_b, scaled_vel_thresh):
                        a_moves_toward_b = True
                    if wrist_moves_toward(lw_a, state_a.left_wrist_velocity, center_b, scaled_vel_thresh):
                        a_moves_toward_b = True

                    # Check B's wrists moving toward A
                    lw_b = _kp_pos(kp_b_xy, kp_b_conf, KP_LEFT_WRIST)
                    rw_b = _kp_pos(kp_b_xy, kp_b_conf, KP_RIGHT_WRIST)
                    if wrist_moves_toward(rw_b, state_b.right_wrist_velocity, center_a, scaled_vel_thresh):
                        b_moves_toward_a = True
                    if wrist_moves_toward(lw_b, state_b.left_wrist_velocity, center_a, scaled_vel_thresh):
                        b_moves_toward_a = True

                    ps.frame_approach = a_moves_toward_b or b_moves_toward_a

                    # -- Signal 3: Wrist-to-wrist proximity --
                    prox_dist = compute_wrist_proximity(
                        kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf)
                    ps.frame_proximity_dist = prox_dist
                    ps.frame_proximity = prox_dist < scaled_prox_px

                    # Draw proximity line between closest wrists
                    if prox_dist < scaled_prox_px * 1.5:
                        # Find the closest wrist pair for drawing
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

                    # -- Temporal interaction tracking --
                    signal_count = ps.active_signal_count()

                    if signal_count >= INTERACTION_SIGNAL_THRESH:
                        ps.last_signal_time = ts_sec
                        if ps.interaction_start < 0:
                            ps.interaction_start = ts_sec

                        # Accumulate which signals have been seen
                        if ps.frame_arm_ext:
                            ps.had_arm_extension = True
                        if ps.frame_approach:
                            ps.had_approach = True
                        if ps.frame_proximity:
                            ps.had_proximity = True
                            ps.last_proximity_time = ts_sec
                            ps.peak_proximity_dist = min(
                                ps.peak_proximity_dist, prox_dist)

                        interaction_dur = ts_sec - ps.interaction_start

                        # Determine direction for the alert
                        if center_b[0] > center_a[0]:
                            alert_dir = "RIGHT"
                        else:
                            alert_dir = "LEFT"

                        # Add visual labels to both students
                        sig_str = (
                            f"{'E' if ps.frame_arm_ext else '-'}"
                            f"{'V' if ps.frame_approach else '-'}"
                            f"{'P' if ps.frame_proximity else '-'}"
                            f" {interaction_dur:.1f}s"
                        )
                        if tid_a in per_student_labels:
                            per_student_labels[tid_a][0] = COL_WARNING
                            per_student_labels[tid_a][1].append(
                                f"INTERACT S#{state_b.student_num} [{sig_str}]")
                        if tid_b in per_student_labels:
                            per_student_labels[tid_b][0] = COL_WARNING
                            per_student_labels[tid_b][1].append(
                                f"INTERACT S#{state_a.student_num} [{sig_str}]")

                        # Draw connection line between student centers
                        cv2.line(annotated,
                                 (int(center_a[0]), int(center_a[1])),
                                 (int(center_b[0]), int(center_b[1])),
                                 COL_NEIGHBOR_LINE, 2, cv2.LINE_AA)

                        # -- Check if interaction qualifies as PASSING --
                        if (interaction_dur >= MIN_INTERACTION_SEC
                                and ps.had_proximity
                                and ps.can_flag(ts_sec)):

                            # Flag both students
                            ps.last_flagged_at = ts_sec
                            state_a.total_alerts += 1
                            state_b.total_alerts += 1
                            total_alerts += 1

                            detail = (
                                f"{alert_dir}, "
                                f"signals: ext={'Y' if ps.had_arm_extension else 'N'} "
                                f"vel={'Y' if ps.had_approach else 'N'} "
                                f"prox={'Y' if ps.had_proximity else 'N'}, "
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

                            # Color both students red
                            if tid_a in per_student_labels:
                                per_student_labels[tid_a][0] = COL_FLAGGED
                            if tid_b in per_student_labels:
                                per_student_labels[tid_b][0] = COL_FLAGGED

                            # Reset interaction so we don't re-flag every frame
                            ps.reset_interaction()

                    else:
                        # Signals dropped below threshold — if no recent activity, reset
                        if (ps.interaction_start > 0
                                and ts_sec - ps.last_signal_time > PROXIMITY_HISTORY_SEC):
                            ps.reset_interaction()

                    # If proximity alone is active (even below signal threshold), warn
                    if ps.frame_proximity and signal_count < INTERACTION_SIGNAL_THRESH:
                        if tid_a in per_student_labels:
                            per_student_labels[tid_a][1].append(
                                f"PROX S#{state_b.student_num} {prox_dist:.0f}px")
                        if tid_b in per_student_labels:
                            per_student_labels[tid_b][1].append(
                                f"PROX S#{state_a.student_num} {prox_dist:.0f}px")

            # ---- Third pass: draw person boxes + labels ----
            for i, tid in enumerate(track_ids):
                if tid not in students:
                    continue
                bbox = tuple(bboxes[i].tolist())
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

        # -- Display ----------------------------------------------
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


# -- Main ---------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Passing Papers Detection Test (PC)")
    print("  Detects: Multi-signal hand interaction between neighbors")
    print(f"  Signals: arm extension + wrist velocity + wrist proximity")
    print("=" * 60)
    print()

    # -- Select video ---------------------------------------------
    log_info("Opening file dialog...")
    video_path = select_video_dialog()
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # -- Validate tracker config ----------------------------------
    if BYTETRACK_CONFIG.exists():
        tracker_cfg = str(BYTETRACK_CONFIG)
        log_info(f"Using ByteTrack config: {BYTETRACK_CONFIG.name}")
    else:
        tracker_cfg = "bytetrack.yaml"
        log_info("Custom ByteTrack config not found, using ultralytics default.")

    # -- Load pose model ------------------------------------------
    if not POSE_MODEL_PATH.exists():
        log_info(f"Pose model not found at: {POSE_MODEL_PATH}")
        log_info("Ultralytics will auto-download on first use.")
    log_info(f"Loading pose model: {POSE_MODEL_PATH.name}")
    model = YOLO(str(POSE_MODEL_PATH))
    log_info("Pose model loaded.")

    # -- Open video -----------------------------------------------
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
    print(f"    4. Repeat for each student (need at least 2 for neighbor detection)")
    print(f"    5. Press S to start detection")
    print()

    # -- Assignment phase -----------------------------------------
    student_map = run_assignment_phase(first_frame, initial_results, disp_scale)
    if student_map is None:
        cap.release()
        log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) < 2:
        cap.release()
        log_info("Need at least 2 students for passing papers detection. Exiting.")
        sys.exit(0)

    # -- Run detection --------------------------------------------
    log_info("Starting detection...")
    run_detection(cap, model, tracker_cfg, student_map, video_path)
    cap.release()
    log_info("Done!")


if __name__ == "__main__":
    main()
