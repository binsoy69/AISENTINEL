#!/usr/bin/env python3
"""
Hands Under Table Detection Test - PC
=======================================
Detects when a student's hands go missing from their desk ROI for a
sustained period, suggesting hands are hidden under the table.

Detection Logic:
  1. Detect person ("student") and "hand" bounding boxes using the
     front node object detection model (my_model.pt)
  2. Associate each student to a desk ROI based on bounding box overlap
  3. Check if that student's hands are within their desk ROI
  4. If hands are missing from the ROI for a sustained threshold
     (default 3 seconds), flag it as suspicious behavior

Workflow:
  1. File picker dialog opens to select a video
  2. First frame shown — user draws polygon ROIs for each desk
  3. Detection runs, associating students to desks and monitoring hands
  4. On sustained detection: saves a timestamped screenshot to ./evidence_hands/

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
    pip install ultralytics opencv-python
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

OBJ_MODEL_PATH = REPO_ROOT / "models" / "front_node" / "my_model.pt"
EVIDENCE_DIR = SCRIPT_DIR / "evidence_hands"

# ── Detection classes ────────────────────────────────────────
CLASS_STUDENT = "student"
CLASS_HAND = "hand"

CONFIDENCE_THRESHOLDS = {
    "student": 0.5,
    "hand": 0.5,
}

# ── Behavior Thresholds ─────────────────────────────────────
HANDS_MISSING_SUSTAIN_SEC = 3.0    # seconds before flagging
EVENT_COOLDOWN_SEC = 10.0          # cooldown between repeated flags

# ── Colors (BGR) ─────────────────────────────────────────────
COL_STUDENT = (0, 255, 0)          # green
COL_HAND = (255, 200, 0)           # cyan-ish
COL_DESK_ROI = (255, 0, 0)        # blue — desk polygon
COL_DESK_FILL = (255, 0, 0)       # blue — translucent fill
COL_ALERT = (0, 0, 255)           # red
COL_WARNING = (0, 165, 255)       # orange
COL_HUD = (0, 255, 0)             # green


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


def log_alert(student_idx: int, desk_idx: int, ts_sec: float, detail: str = ""):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.RED}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{TC.RED}HANDS UNDER TABLE - Student near Desk #{desk_idx + 1}{TC.RESET}"
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


def save_evidence(frame, video_name, desk_idx, ts_sec):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"{video_name}_hands_under_desk{desk_idx + 1}_{ts_str}.jpg"
    path = EVIDENCE_DIR / fname
    cv2.imwrite(str(path), frame)
    log_info(f"Evidence saved: {fname}")


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

    polygons = []            # completed desk polygons (original-res coords)
    current_points = []      # vertices of the polygon being drawn

    def on_mouse(event, mx, my, flags, param):
        # Convert display coords to original frame coords
        ox = int(mx / scale)
        oy = int(my / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((ox, oy))

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Close current polygon
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
            # Label at centroid
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(display, f"Desk {i + 1}", (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_DESK_ROI, 2)
        cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        # Draw current polygon being drawn
        if len(current_points) > 0:
            pts = np.array(current_points, dtype=np.int32)
            for j in range(len(current_points) - 1):
                cv2.line(display, current_points[j], current_points[j + 1],
                         (0, 255, 0), 2, cv2.LINE_AA)
            for pt in current_points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
            # Show vertex count
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

        # Desk count
        cv2.putText(display, f"Desks defined: {len(polygons)}", (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Scale for display
        if scale < 1.0:
            show = cv2.resize(display, (int(fw * scale), int(fh * scale)))
        else:
            show = display

        cv2.imshow(win, show)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            # Close any in-progress polygon first
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


def bbox_in_polygon(bbox, polygon):
    """Check if the center of a bounding box is inside the polygon."""
    cx, cy = bbox_center(bbox)
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0



def find_desk_for_student(student_bbox, desk_polygons):
    """
    Find the desk ROI that best matches this student.

    Strategy: "closest desk below the student" — picks the desk whose
    top edge is closest below (or at) the student's center-y, within the
    student's horizontal range.  This handles perspective overlap where
    front-row students' bboxes overlap with back-row desk ROIs.

    In image coordinates, "below" means larger y values (closer to camera).

    Returns (desk_index, polygon) or (None, None).
    """
    sx1, sy1, sx2, sy2 = student_bbox
    s_cx = (sx1 + sx2) / 2.0
    s_cy = (sy1 + sy2) / 2.0

    best_idx = None
    best_dist = float("inf")

    for i, poly in enumerate(desk_polygons):
        # Desk top edge = minimum y of the polygon vertices
        desk_top_y = float(poly[:, 1].min())
        desk_min_x = float(poly[:, 0].min())
        desk_max_x = float(poly[:, 0].max())

        # Student must horizontally overlap with the desk
        margin_x = (sx2 - sx1) * 0.2
        if s_cx < desk_min_x - margin_x or s_cx > desk_max_x + margin_x:
            continue

        # The desk top edge must be at or below the student's center
        # (desk_top_y >= s_cy in image coords)
        if desk_top_y < s_cy:
            continue

        dist = desk_top_y - s_cy
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is not None:
        return best_idx, desk_polygons[best_idx]
    return None, None


def find_hands_in_roi(hand_boxes, desk_polygon):
    """
    Check how many hand bounding boxes have their center inside the desk ROI.
    Returns the count of hands inside the ROI.
    """
    count = 0
    for hbox in hand_boxes:
        if bbox_in_polygon(hbox, desk_polygon):
            count += 1
    return count


# ══════════════════════════════════════════════════════════════
#  PER-DESK TRACKING STATE
# ══════════════════════════════════════════════════════════════

class DeskState:
    """Tracks the hands-missing state for a single desk."""

    def __init__(self, desk_idx: int):
        self.desk_idx = desk_idx
        self.hands_missing_start = -1.0   # timestamp when hands first went missing
        self.last_flagged_at = -999.0     # last time an alert was fired
        self.total_alerts = 0

    def can_flag(self, now: float) -> bool:
        return (now - self.last_flagged_at) > EVENT_COOLDOWN_SEC


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_desk_rois(img, desk_polygons, desk_states):
    """Draw all desk ROI polygons with status coloring."""
    overlay = img.copy()
    for i, poly in enumerate(desk_polygons):
        state = desk_states[i]
        # Color based on state
        if state.hands_missing_start > 0:
            color = COL_WARNING
        else:
            color = COL_DESK_ROI

        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)

        # Label at centroid
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(img, f"Desk {i + 1}", (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Hands Under Table Detection Test (PC)")
    print("  Detects: hands missing from desk ROI")
    print(f"  Sustained threshold: {HANDS_MISSING_SUSTAIN_SEC}s")
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

    # Show model classes
    print(f"\n{TC.BOLD}Model classes:{TC.RESET}")
    for idx, name in model.names.items():
        marker = "  << USED" if name in (CLASS_STUDENT, CLASS_HAND) else ""
        thresh = CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){marker}")
    print()

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
    print(f"  Video      : {Path(video_path).name}")
    print(f"  Resolution : {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Desk ROIs  : {len(desk_polygons)}")
    print(f"  Threshold  : hands missing for {HANDS_MISSING_SUSTAIN_SEC}s")
    print(f"  Cooldown   : {EVENT_COOLDOWN_SEC}s")
    print(f"  Evidence   : {EVIDENCE_DIR}")
    print("=" * 60)
    print()

    # Reset video to start (frame 1 was already read for calibration)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Detection loop ────────────────────────────────────────
    frame_idx = 0
    paused = False
    total_alerts = 0
    win_name = "AISENTINEL - Hands Under Table Detection"

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

        # ── Run detection ─────────────────────────────────────
        results = model(frame, verbose=False, imgsz=640)
        boxes = results[0].boxes
        annotated = frame.copy()

        # Separate student and hand detections
        student_boxes = []   # list of [x1, y1, x2, y2]
        hand_boxes = []      # list of [x1, y1, x2, y2]

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, f"class_{cls_id}")

                if label == CLASS_STUDENT:
                    min_conf = CONFIDENCE_THRESHOLDS.get(CLASS_STUDENT, 0.5)
                    if conf >= min_conf:
                        coords = list(map(int, box.xyxy[0].tolist()))
                        student_boxes.append(coords)
                        # Draw student box
                        x1, y1, x2, y2 = coords
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      COL_STUDENT, 2)
                        draw_label(annotated, f"student {conf:.0%}",
                                   x1, y1 - 2, COL_STUDENT)

                elif label == CLASS_HAND:
                    min_conf = CONFIDENCE_THRESHOLDS.get(CLASS_HAND, 0.5)
                    if conf >= min_conf:
                        coords = list(map(int, box.xyxy[0].tolist()))
                        hand_boxes.append(coords)
                        # Draw hand box
                        x1, y1, x2, y2 = coords
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      COL_HAND, 2)
                        draw_label(annotated, f"hand {conf:.0%}",
                                   x1, y1 - 2, COL_HAND)

        # ── Associate students to desks & check hands ─────────
        frame_events = []  # desk indices that triggered alerts this frame

        # Track which desks have a student and whether hands are present
        desk_has_student = [False] * len(desk_polygons)
        desk_hands_count = [0] * len(desk_polygons)

        # Associate students to desks
        for s_bbox in student_boxes:
            desk_idx, desk_poly = find_desk_for_student(s_bbox, desk_polygons)
            if desk_idx is None:
                continue
            desk_has_student[desk_idx] = True

        # Check if ANY hand is inside each desk ROI (no need to associate
        # hands to specific students — if a hand is on the desk, it counts)
        for i, poly in enumerate(desk_polygons):
            desk_hands_count[i] = find_hands_in_roi(hand_boxes, poly)

        # ── Update desk states ────────────────────────────────
        for i, state in enumerate(desk_states):
            if not desk_has_student[i]:
                # No student at this desk — reset timer
                state.hands_missing_start = -1.0
                continue

            hands_present = desk_hands_count[i] > 0

            if not hands_present:
                # Hands are missing from the ROI
                if state.hands_missing_start < 0:
                    state.hands_missing_start = ts_sec
                elapsed = ts_sec - state.hands_missing_start

                if elapsed >= HANDS_MISSING_SUSTAIN_SEC and state.can_flag(ts_sec):
                    state.last_flagged_at = ts_sec
                    state.total_alerts += 1
                    total_alerts += 1
                    log_alert(0, i, ts_sec,
                              f"sustained {elapsed:.1f}s, no hands in desk ROI")
                    frame_events.append(i)
            else:
                # Hands are present — reset timer
                state.hands_missing_start = -1.0

        # ── Draw desk ROIs ────────────────────────────────────
        draw_desk_rois(annotated, desk_polygons, desk_states)

        # Draw alert indicators on flagged desks
        for i, state in enumerate(desk_states):
            if state.hands_missing_start > 0 and desk_has_student[i]:
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

        # ── Save evidence ─────────────────────────────────────
        for desk_idx in frame_events:
            save_evidence(annotated, video_name, desk_idx, ts_sec)

        # ── HUD ───────────────────────────────────────────────
        ts_text = fmt_ts(ts_sec)
        hud1 = f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}"
        hud2 = (f"Students: {len(student_boxes)} | Hands: {len(hand_boxes)} "
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
                txt = f"ALERT: Hands missing from Desk #{desk_idx + 1}"
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_ALERT, 2,
                            cv2.LINE_AA)
                banner_y -= 35

        # Timestamp watermark bottom-right
        (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
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
            print(f"    Desk #{i + 1:2d}        : {state.total_alerts} alerts")
    if total_alerts > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No hands-under-table detected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
