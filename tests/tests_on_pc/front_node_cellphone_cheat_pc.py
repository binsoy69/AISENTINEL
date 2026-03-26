#!/usr/bin/env python3
"""
Cellphone / Cheat Sheet Detection Test - PC
=============================================
Runs the front node YOLO object detection model on a video file,
filtering for only cellphone and cheat_sheet detections.

Workflow:
  1. File picker dialog opens to select a video
  2. Model runs on each frame, drawing bounding boxes for cellphone/cheat_sheet
  3. On detection: saves a timestamped screenshot to ./evidence_obj/

Controls:
    q / ESC  - Quit
    SPACE    - Pause / Resume

Requirements:
    pip install ultralytics opencv-python
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OBJ_MODEL_PATH = REPO_ROOT / "models" / "sentinel-yolov11n" / "my_model2.pt"
EVIDENCE_DIR = SCRIPT_DIR / "evidence_obj"

# ── Only these classes matter ────────────────────────────────
TARGET_CLASSES = {"cellphone", "cheat_sheet"}

CONFIDENCE_THRESHOLDS = {
    "cellphone": 0.6,
    "cheat_sheet": 0.5,
}

# ── Colors (BGR) ─────────────────────────────────────────────
COL_CELLPHONE = (0, 0, 255)       # red
COL_CHEAT_SHEET = (0, 165, 255)   # orange
COL_HUD = (0, 255, 0)             # green

CLASS_COLORS = {
    "cellphone": COL_CELLPHONE,
    "cheat_sheet": COL_CHEAT_SHEET,
}


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


def log_alert(label: str, conf: float, ts_sec: float):
    ts = fmt_ts(ts_sec)
    print(
        f"{TC.RED}{TC.BOLD}[ALERT @ {ts}]{TC.RESET} "
        f"{TC.RED}{label.upper()} detected (conf={conf:.0%}){TC.RESET}"
    )


def log_info(msg: str):
    print(f"{TC.CYAN}[INFO]{TC.RESET} {msg}")


# ── Drawing helpers ──────────────────────────────────────────
def draw_label(img, text, x, y, bg, fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)


def save_evidence(frame, video_name, label, conf, ts_sec):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts_str = fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"{video_name}_{label}_{conf:.0f}pct_{ts_str}.jpg"
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


# ── Main ─────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  AISENTINEL - Cellphone / Cheat Sheet Detection Test (PC)")
    print("  Detects: cellphone | cheat_sheet")
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

    # Show which classes the model knows about
    print(f"\n{TC.BOLD}Model classes:{TC.RESET}")
    for idx, name in model.names.items():
        marker = "  << TARGET" if name in TARGET_CLASSES else ""
        thresh = CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){marker}")
    print()

    # ── Open video ────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{TC.RED}[ERROR] Cannot open video: {video_path}{TC.RESET}")
        sys.exit(1)

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0

    print("=" * 60)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {fmt_ts(duration)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Evidence dir : {EVIDENCE_DIR}")
    print("=" * 60)
    print()

    # ── Detection loop ────────────────────────────────────────
    frame_idx = 0
    paused = False
    stats = defaultdict(int)
    alert_count = 0
    win_name = "AISENTINEL - Cellphone / Cheat Sheet Detection"

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
        frame_detections = []  # (label, conf) for this frame

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, f"class_{cls_id}")

                # Only care about target classes
                if label not in TARGET_CLASSES:
                    continue

                min_conf = CONFIDENCE_THRESHOLDS.get(label, 0.5)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                stats[label] += 1
                color = CLASS_COLORS.get(label, (0, 0, 255))

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                draw_label(annotated, f"{label} {conf:.0%}", x1, y1 - 2, color)

                frame_detections.append((label, conf, x1, y1, x2, y2))

        # ── Alerts + evidence ─────────────────────────────────
        for label, conf, *_ in frame_detections:
            alert_count += 1
            log_alert(label, conf, ts_sec)
            save_evidence(annotated, video_name, label, conf, ts_sec)

        # ── HUD ───────────────────────────────────────────────
        ts_text = fmt_ts(ts_sec)
        hud1 = f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}"
        hud2 = f"Alerts: {alert_count}"

        cv2.putText(annotated, hud1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, hud1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_HUD, 2, cv2.LINE_AA)

        hud_color = COL_CELLPHONE if alert_count > 0 else COL_HUD
        cv2.putText(annotated, hud2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, hud2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)

        # Alert banner
        if frame_detections:
            banner_y = h - 30
            for label, conf, *_ in frame_detections:
                txt = f"DETECTED: {label.upper()} ({conf:.0%})"
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated, txt, (10, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_CELLPHONE, 2, cv2.LINE_AA)
                banner_y -= 35

        # Timestamp watermark bottom-right
        (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, ts_text, (w - tw - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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
    print(f"  Total alerts     : {alert_count}")
    for label, count in sorted(stats.items()):
        print(f"    {label:20s}: {count} detections")
    if alert_count > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No cellphone/cheat_sheet detected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
