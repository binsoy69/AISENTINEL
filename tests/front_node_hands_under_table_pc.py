#!/usr/bin/env python3
"""
Hands Under Table Detection Test - PC (v2 — Improved)
======================================================
Detects when a student's hands go missing from their desk ROI for a
sustained period, suggesting hands are hidden under the table.

Algorithm Improvements over v1:
  - ByteTrack student tracking for persistent IDs across frames
  - Student-to-desk assignment via bounding-box / polygon intersection area
  - Hands associated to the nearest student (not just any hand in ROI)
  - Temporal smoothing: majority-vote buffer over N frames per desk
  - Student presence validation: reset desk if student disappears
  - Improved evidence capture: raw + annotated frames, richer filenames

Detection Logic:
  1. Track students with model.track(persist=True) for stable IDs
  2. Assign each tracked student to a desk via max overlap area
  3. Associate each detected hand to the nearest student
  4. For each desk: check if the assigned student's hands are in the ROI
  5. Majority-vote over a sliding window to smooth missed detections
  6. Flag only when hands missing >= threshold AND majority confirms

Workflow:
  1. File picker dialog opens to select a video
  2. First frame shown — user draws polygon ROIs for each desk
  3. Detection runs with tracking, per-student hand association, smoothing
  4. On sustained detection: saves evidence to ./evidence_hands/

Desk ROI Drawing Controls:
    Left-click      — place polygon vertex
    Right-click     — close current polygon (finish desk)
    Z               — undo last vertex
    C               — clear all
    ENTER / SPACE   — confirm & start processing
    ESC             — cancel

Detection Controls:
    q / ESC  — Quit
    SPACE    — Pause / Resume

Requirements:
    pip install ultralytics opencv-python numpy lap
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

OBJ_MODEL_PATH = REPO_ROOT / "models" / "front_node" / "my_model.pt"
BYTETRACK_CONFIG = SCRIPT_DIR / "bytetrack_front.yaml"
EVIDENCE_DIR = SCRIPT_DIR / "evidence_hands"

# ── Detection classes ────────────────────────────────────────
CLASS_STUDENT = "student"
CLASS_HAND = "hand"

CONFIDENCE_THRESHOLDS = {
    "student": 0.5,
    "hand": 0.5,
}

# ── Behavior Thresholds ─────────────────────────────────────
HANDS_MISSING_SUSTAIN_SEC = 3.0     # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0           # cooldown between repeated flags

# ── Tracking & Smoothing ────────────────────────────────────
HAND_ASSOC_MARGIN_PX = 60           # max pixel distance from student bbox to claim a hand
SMOOTH_WINDOW_FRAMES = 12           # sliding window size for majority vote
SMOOTH_MISSING_RATIO = 0.6          # fraction of window that must be "missing" to confirm
STUDENT_ABSENT_RESET_SEC = 2.0      # reset desk if student undetected for this long

# ── Colors (BGR) ─────────────────────────────────────────────
COL_STUDENT = (0, 255, 0)           # green
COL_HAND = (255, 200, 0)            # cyan-ish
COL_DESK_ROI = (255, 0, 0)          # blue — desk polygon
COL_DESK_FILL = (255, 0, 0)         # blue — translucent fill
COL_ALERT = (0, 0, 255)             # red
COL_WARNING = (0, 165, 255)         # orange
COL_HUD = (0, 255, 0)               # green
COL_ASSOC_LINE = (200, 200, 0)      # teal — hand-to-student association line


# ── Terminal helpers ─────────────────────────────────────────
class TC:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
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


def save_evidence(annotated_frame, raw_frame, video_name, desk_idx, student_id, ts_sec):
    """Save both annotated and raw evidence frames."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")

    # Annotated frame
    fname_ann = f"{video_name}_desk{desk_idx + 1}_sid{student_id}_{ts_str}_annotated.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname_ann), annotated_frame)

    # Raw frame
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

    win = "AISENTINEL - Desk ROI Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    instructions = [
        "Left-click: place vertex | Right-click: close polygon",
        "Z: undo | C: clear | ENTER/SPACE: confirm | ESC: cancel",
    ]

    while True:
        display = frame.copy()

        # Draw completed polygons with translucent fill
        overlay = display.copy()
        for i, poly in enumerate(polygons):
            cv2.fillPoly(overlay, [poly], COL_DESK_FILL)
            cv2.polylines(display, [poly], True, COL_DESK_ROI, 2, cv2.LINE_AA)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(display, f"Desk {i + 1}", (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_DESK_ROI, 2)
        cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        # Draw current polygon being drawn
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

        # Instructions overlay
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
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def bbox_center(bbox):
    """Return (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(px, py, polygon):
    """Check if a point is inside the polygon."""
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


def bbox_polygon_intersection_area(bbox, polygon, img_shape):
    """
    Compute the intersection area between an axis-aligned bounding box
    and a polygon using binary mask rasterization.

    This is the key improvement for student-to-desk assignment:
    instead of a simple heuristic, we measure actual pixel overlap.
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Create bbox mask (only in the bounding region for speed)
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Shift polygon into ROI-local coordinates
    shifted_poly = polygon.copy()
    shifted_poly[:, 0] -= x1
    shifted_poly[:, 1] -= y1

    # Rasterize the polygon in the ROI region
    poly_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [shifted_poly], 255)

    # The bbox mask is all-ones in this ROI, so intersection = polygon mask sum
    return float(np.count_nonzero(poly_mask))


def find_desk_for_student(student_bbox, desk_polygons, img_shape):
    """
    Assign a student to the desk with the largest bounding-box / polygon
    intersection area. Returns (desk_index, area) or (None, 0).
    """
    best_idx = None
    best_area = 0.0

    for i, poly in enumerate(desk_polygons):
        area = bbox_polygon_intersection_area(student_bbox, poly, img_shape)
        if area > best_area:
            best_area = area
            best_idx = i

    return best_idx, best_area


def hand_distance_to_bbox(hand_center, student_bbox):
    """
    Compute the signed distance from a hand center to the nearest edge
    of the student bounding box. Returns 0 if inside, positive if outside.
    """
    hx, hy = hand_center
    sx1, sy1, sx2, sy2 = student_bbox

    # Clamp to nearest point on bbox
    cx = max(sx1, min(hx, sx2))
    cy = max(sy1, min(hy, sy2))

    dx = hx - cx
    dy = hy - cy
    return (dx * dx + dy * dy) ** 0.5


# ══════════════════════════════════════════════════════════════
#  PER-DESK TRACKING STATE (with temporal smoothing)
# ══════════════════════════════════════════════════════════════

class DeskState:
    """
    Tracks the hands-missing state for a single desk, including:
    - Assigned student track ID (persistent via ByteTrack)
    - Sliding-window majority vote for temporal smoothing
    - Student presence timer for absence-based reset
    """

    def __init__(self, desk_idx: int):
        self.desk_idx = desk_idx

        # Student assignment (persistent across frames via tracking)
        self.assigned_student_id = None

        # Temporal smoothing buffer: True = hands present, False = hands missing
        self.history = deque(maxlen=SMOOTH_WINDOW_FRAMES)

        # Sustained detection timing
        self.hands_missing_start = -1.0
        self.last_flagged_at = -999.0
        self.total_alerts = 0

        # Student presence validation
        self.last_student_seen_at = -1.0

    def can_flag(self, now: float) -> bool:
        return (now - self.last_flagged_at) > EVENT_COOLDOWN_SEC

    def push_observation(self, hands_present: bool):
        """Add a frame observation to the sliding window."""
        self.history.append(hands_present)

    def majority_says_missing(self) -> bool:
        """
        Returns True if the majority of the sliding window says hands are missing.
        Requires at least half the window to be filled before deciding.
        """
        if len(self.history) < SMOOTH_WINDOW_FRAMES // 2:
            return False
        missing_count = sum(1 for v in self.history if not v)
        return missing_count / len(self.history) >= SMOOTH_MISSING_RATIO

    def reset(self):
        """Full reset when student changes or disappears."""
        self.history.clear()
        self.hands_missing_start = -1.0

    def reset_assignment(self):
        """Clear the student assignment and all state."""
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


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Hands Under Table Detection Test (PC) v2")
    print("  Detects: hands missing from desk ROI (tracked + smoothed)")
    print(f"  Sustained threshold: {HANDS_MISSING_SUSTAIN_SEC}s")
    print(f"  Smoothing window: {SMOOTH_WINDOW_FRAMES} frames "
          f"(missing ratio >= {SMOOTH_MISSING_RATIO:.0%})")
    print("=" * 60)
    print()

    # ── Select video ──────────────────────────────────────────
    log_info("Opening file dialog...")
    video_path = select_video_dialog()
    if not video_path:
        log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{TC.RED}[ERROR] File not found: {video_path}{TC.RESET}")
        sys.exit(1)
    log_info(f"Selected: {video_path}")

    # ── Load model ────────────────────────────────────────────
    if not OBJ_MODEL_PATH.exists():
        print(f"{TC.RED}[ERROR] Model not found: {OBJ_MODEL_PATH}{TC.RESET}")
        sys.exit(1)

    log_info(f"Loading model: {OBJ_MODEL_PATH.name}")
    model = YOLO(str(OBJ_MODEL_PATH))
    log_info("Model loaded.")

    # Resolve class IDs for student and hand
    student_cls_id = None
    hand_cls_id = None
    print(f"\n{TC.BOLD}Model classes:{TC.RESET}")
    for idx, name in model.names.items():
        marker = ""
        if name == CLASS_STUDENT:
            student_cls_id = idx
            marker = "  << STUDENT"
        elif name == CLASS_HAND:
            hand_cls_id = idx
            marker = "  << HAND"
        thresh = CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){marker}")
    print()

    if student_cls_id is None or hand_cls_id is None:
        print(f"{TC.RED}[ERROR] Model must have '{CLASS_STUDENT}' and '{CLASS_HAND}' classes.{TC.RESET}")
        sys.exit(1)

    # Build the list of class IDs to track (student + hand)
    track_classes = [student_cls_id, hand_cls_id]

    # ── Tracker config ────────────────────────────────────────
    tracker_cfg = str(BYTETRACK_CONFIG) if BYTETRACK_CONFIG.exists() else "bytetrack.yaml"
    log_info(f"Tracker config: {tracker_cfg}")

    # ── Open video & read first frame for calibration ─────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{TC.RED}[ERROR] Cannot open video: {video_path}{TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{TC.RED}[ERROR] Cannot read first frame.{TC.RESET}")
        sys.exit(1)

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)
    duration = total_frames / fps if fps > 0 else 0
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    # ── Desk ROI calibration ──────────────────────────────────
    log_info("Draw polygon ROIs for each desk on the first frame.")
    desk_polygons = calibrate_desk_rois(first_frame)
    if desk_polygons is None or len(desk_polygons) == 0:
        cap.release()
        log_info("No desk ROIs defined. Exiting.")
        sys.exit(0)

    # Initialize per-desk state
    desk_states = [DeskState(i) for i in range(len(desk_polygons))]

    print()
    print("=" * 60)
    print(f"  Video        : {Path(video_path).name}")
    print(f"  Resolution   : {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames : {total_frames}")
    print(f"  Desk ROIs    : {len(desk_polygons)}")
    print(f"  Threshold    : hands missing for {HANDS_MISSING_SUSTAIN_SEC}s")
    print(f"  Cooldown     : {EVENT_COOLDOWN_SEC}s")
    print(f"  Smoothing    : {SMOOTH_WINDOW_FRAMES} frames, "
          f">= {SMOOTH_MISSING_RATIO:.0%} missing to confirm")
    print(f"  Hand margin  : {HAND_ASSOC_MARGIN_PX}px from student bbox")
    print(f"  Student reset: {STUDENT_ABSENT_RESET_SEC}s absent")
    print(f"  Evidence     : {EVIDENCE_DIR}")
    print("=" * 60)
    print()

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Detection loop ────────────────────────────────────────
    frame_idx = 0
    paused = False
    total_alerts = 0
    win_name = "AISENTINEL - Hands Under Table Detection v2"

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
        raw_frame = frame.copy()  # keep raw for evidence

        # ──────────────────────────────────────────────────────
        #  1. RUN TRACKING — persistent student IDs via ByteTrack
        # ──────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            verbose=False,
            imgsz=640,
            classes=track_classes,
        )
        boxes = results[0].boxes
        annotated = frame.copy()

        # Parse detections into students (with track IDs) and hands
        # student_tracks: {track_id: [x1, y1, x2, y2]}
        # hand_detections: list of [x1, y1, x2, y2]
        student_tracks = {}
        hand_detections = []

        if boxes is not None and len(boxes) > 0:
            has_ids = boxes.id is not None

            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j])
                conf = float(boxes.conf[j])
                label = model.names.get(cls_id, f"class_{cls_id}")
                coords = list(map(int, boxes.xyxy[j].tolist()))

                if label == CLASS_STUDENT:
                    if conf < CONFIDENCE_THRESHOLDS.get(CLASS_STUDENT, 0.5):
                        continue
                    # Get track ID (ByteTrack assigns these)
                    track_id = int(boxes.id[j]) if has_ids else j
                    student_tracks[track_id] = coords

                    # Draw student box with track ID
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_STUDENT, 2)
                    draw_label(annotated, f"S#{track_id} {conf:.0%}",
                               x1, y1 - 2, COL_STUDENT)

                elif label == CLASS_HAND:
                    if conf < CONFIDENCE_THRESHOLDS.get(CLASS_HAND, 0.5):
                        continue
                    hand_detections.append(coords)

                    # Draw hand box
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_HAND, 2)
                    draw_label(annotated, f"hand {conf:.0%}",
                               x1, y1 - 2, COL_HAND)

        # ──────────────────────────────────────────────────────
        #  2. ASSIGN STUDENTS TO DESKS (intersection area)
        # ──────────────────────────────────────────────────────
        # Build current frame's best student-to-desk mapping
        # If a desk already has an assigned student ID (from tracking),
        # keep it as long as that student is still detected.

        # First: find the best desk for each currently-detected student
        student_to_best_desk = {}  # track_id -> (desk_idx, area)
        for track_id, s_bbox in student_tracks.items():
            desk_idx, area = find_desk_for_student(s_bbox, desk_polygons, img_shape)
            if desk_idx is not None and area > 0:
                student_to_best_desk[track_id] = (desk_idx, area)

        # Resolve conflicts: if two students want the same desk, largest area wins
        desk_to_candidates = defaultdict(list)
        for track_id, (desk_idx, area) in student_to_best_desk.items():
            desk_to_candidates[desk_idx].append((track_id, area))

        # For each desk, pick the student with the largest overlap
        desk_best_student = {}  # desk_idx -> track_id
        for desk_idx, candidates in desk_to_candidates.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            desk_best_student[desk_idx] = candidates[0][0]

        # Update desk state assignments
        for i, state in enumerate(desk_states):
            best_sid = desk_best_student.get(i)

            if best_sid is not None:
                # A student is detected at this desk
                if state.assigned_student_id is None:
                    # First assignment
                    state.assigned_student_id = best_sid
                    state.last_student_seen_at = ts_sec
                elif state.assigned_student_id == best_sid:
                    # Same student, refresh presence timer
                    state.last_student_seen_at = ts_sec
                else:
                    # Different student claimed this desk — re-assign
                    # (happens if students swap seats or IDs shift)
                    state.assigned_student_id = best_sid
                    state.last_student_seen_at = ts_sec
                    state.reset()
            else:
                # No student detected at this desk this frame
                if state.assigned_student_id is not None:
                    # Check if the assigned student is still detected anywhere
                    if state.assigned_student_id in student_tracks:
                        # Student exists but moved away from this desk
                        state.reset_assignment()
                    else:
                        # Student not detected at all — allow brief absence
                        elapsed_absent = ts_sec - state.last_student_seen_at
                        if state.last_student_seen_at > 0 and elapsed_absent > STUDENT_ABSENT_RESET_SEC:
                            state.reset_assignment()
                        # Else: keep assignment, student is temporarily undetected

        # ──────────────────────────────────────────────────────
        #  3. ASSOCIATE HANDS WITH STUDENTS
        # ──────────────────────────────────────────────────────
        # Each hand is claimed by the nearest student within margin.
        # student_hands: {track_id: [hand_bbox, ...]}
        student_hands = defaultdict(list)

        for h_bbox in hand_detections:
            hx, hy = bbox_center(h_bbox)
            best_sid = None
            best_dist = float("inf")

            for track_id, s_bbox in student_tracks.items():
                dist = hand_distance_to_bbox((hx, hy), s_bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_sid = track_id

            # Assign hand if inside student bbox (dist=0) or within margin
            if best_sid is not None and best_dist <= HAND_ASSOC_MARGIN_PX:
                student_hands[best_sid].append(h_bbox)

                # Draw association line from hand center to student bbox center
                sx1, sy1, sx2, sy2 = student_tracks[best_sid]
                s_cx, s_cy = int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)
                cv2.line(annotated, (int(hx), int(hy)), (s_cx, s_cy),
                         COL_ASSOC_LINE, 1, cv2.LINE_AA)

        # ──────────────────────────────────────────────────────
        #  4. DESK-LEVEL HAND PRESENCE CHECK
        # ──────────────────────────────────────────────────────
        # For each desk: does the assigned student have at least one
        # hand whose center is inside the desk ROI?
        frame_events = []

        for i, state in enumerate(desk_states):
            sid = state.assigned_student_id
            if sid is None or sid not in student_tracks:
                # No active student at this desk — push "present" to
                # avoid accumulating false missing votes
                state.push_observation(True)
                continue

            # Get this student's associated hands
            hands = student_hands.get(sid, [])

            # Check if any of this student's hands are inside the desk ROI
            hands_in_desk = 0
            poly = desk_polygons[i]
            for h_bbox in hands:
                hx, hy = bbox_center(h_bbox)
                if point_in_polygon(hx, hy, poly):
                    hands_in_desk += 1

            hands_present = hands_in_desk > 0

            # ──────────────────────────────────────────────────
            #  5. TEMPORAL SMOOTHING (majority vote)
            # ──────────────────────────────────────────────────
            state.push_observation(hands_present)
            smoothed_missing = state.majority_says_missing()

            # ──────────────────────────────────────────────────
            #  6. SUSTAINED DETECTION with smoothed signal
            # ──────────────────────────────────────────────────
            if smoothed_missing:
                # Majority of recent frames say hands are missing
                if state.hands_missing_start < 0:
                    state.hands_missing_start = ts_sec
                elapsed = ts_sec - state.hands_missing_start

                if elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.can_flag(ts_sec):
                    state.last_flagged_at = ts_sec
                    state.total_alerts += 1
                    total_alerts += 1
                    log_alert(sid, i, ts_sec,
                              f"sustained {elapsed:.1f}s, smoothed missing "
                              f"({sum(1 for v in state.history if not v)}"
                              f"/{len(state.history)} frames)")
                    frame_events.append(i)
            else:
                # Smoothed signal says hands are present — reset
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
            sid = desk_states[desk_idx].assigned_student_id or 0
            save_evidence(annotated, raw_frame, video_name, desk_idx, sid, ts_sec)

        # ── HUD ───────────────────────────────────────────────
        ts_text = fmt_ts(ts_sec)
        hud1 = f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}"
        tracked_count = len(student_tracks)
        hud2 = (f"Tracked: {tracked_count} | Hands: {len(hand_detections)} "
                f"| Alerts: {total_alerts}")

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
                sid = desk_states[desk_idx].assigned_student_id or 0
                txt = f"ALERT: S#{sid} hands missing from Desk #{desk_idx + 1}"
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

        # ── Display ───────────────────────────────────────────
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

    cap.release()
    cv2.destroyAllWindows()

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
            sid = state.assigned_student_id or "?"
            print(f"    Desk #{i + 1:2d} (S#{sid})  : {state.total_alerts} alerts")
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No hands-under-table detected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
