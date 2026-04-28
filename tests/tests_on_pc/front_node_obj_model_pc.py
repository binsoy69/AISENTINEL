#!/usr/bin/env python3
"""
Front Node Object Detection Model - PC Test Program
=====================================================
Runs the custom-trained front node YOLO object detection model on video
files for testing on PC (not Raspberry Pi). Uses the standard Ultralytics
YOLO API (PyTorch .pt weights) instead of Hailo HEF.

Features:
  - Accepts video files via CLI or interactive menu
  - Class labels are read directly from the loaded model
  - Alerts in terminal when cellphone or cheat_sheet is detected
  - Prints the video timestamp of each detection
  - Saves a screenshot of the frame on cellphone / cheat_sheet detection

Usage:
    python front_node_obj_model_pc.py --video path/to/exam.mp4
    python front_node_obj_model_pc.py --video vid1.mp4 vid2.mp4
    python front_node_obj_model_pc.py                          # Interactive

Controls (display window):
    q / ESC  - Quit
    SPACE    - Pause / Resume

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import cv2
from ultralytics import YOLO

from front_node_pc_common import (
    FRONT_NODE_OBJECT_MODEL_CANDIDATES,
    SENTINEL_MODEL_CANDIDATES,
    canonical_label,
    first_existing,
)

# ── Paths (relative to repo root) ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Default model path
OBJ_MODEL_PATH = first_existing(
    FRONT_NODE_OBJECT_MODEL_CANDIDATES + SENTINEL_MODEL_CANDIDATES
) or (REPO_ROOT / "models" / "front_node" / "my_model.pt")

# Output directory for detection screenshots
DETECTION_OUTPUT_DIR = SCRIPT_DIR / "detection_output"

# ── Alert classes (these trigger terminal alerts + screenshots) ──
ALERT_CLASSES = {"cellphone", "phone", "cheat_sheet"}

# ── Detection thresholds ─────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "student": 0.5,
    "cellphone": 0.6,
    "phone": 0.6,
    "paper": 0.5,
    "hand": 0.5,
    "calculator": 0.5,
    "cheat_sheet": 0.5,
}


# ── Terminal color helpers ───────────────────────────────────
class TermColor:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def fmt_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm."""
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    hrs, rem = divmod(total_sec, 3600)
    mins, secs = divmod(rem, 60)
    millis = int((seconds - total_sec) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def alert(msg: str, timestamp_sec: float, color: str = TermColor.RED):
    """Print a colored alert to the terminal with video timestamp."""
    ts = fmt_timestamp(timestamp_sec)
    print(
        f"{color}{TermColor.BOLD}[ALERT @ {ts}]{TermColor.RESET} "
        f"{color}{msg}{TermColor.RESET}"
    )


def info(msg: str):
    print(f"{TermColor.CYAN}[INFO]{TermColor.RESET} {msg}")


def save_detection_frame(
    frame, video_name: str, timestamp_sec: float, label: str, confidence: float
):
    """Save a screenshot of the frame where a detection occurred."""
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    ts_str = fmt_timestamp(timestamp_sec).replace(":", "").replace(".", "_")
    filename = f"{video_name}_{ts_str}_{label}_{confidence:.0%}.jpg".replace("%", "pct")
    filepath = DETECTION_OUTPUT_DIR / filename
    cv2.imwrite(str(filepath), frame)
    info(f"Screenshot saved: {filepath.name}")


# ─────────────────────────────────────────────────────────────
#  Main processing
# ─────────────────────────────────────────────────────────────
def process_video(
    video_path: str,
    obj_model: YOLO,
    show_display: bool = True,
):
    """Run front-node object detection on a single video."""

    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"{TermColor.RED}[ERROR] Cannot open video: {video_path}{TermColor.RESET}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print()
    print("=" * 65)
    print(f"  Processing: {Path(video_path).name}")
    print(f"  Resolution: {width}x{height}  |  FPS: {fps:.1f}  |  Duration: {fmt_timestamp(duration)}")
    print(f"  Total frames: {total_frames}")
    print("=" * 65)

    # Scale for display
    disp_scale = 1.0
    if width > 1280:
        disp_scale = 1280 / width

    frame_idx = 0
    paused = False

    # Stats
    stats = defaultdict(int)
    alert_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                info("End of video reached.")
                break
            frame_idx += 1
        else:
            # While paused, just wait for key
            key = cv2.waitKey(100) & 0xFF
            if key == ord(" "):
                paused = False
            elif key in (ord("q"), 27):
                break
            continue

        timestamp_sec = frame_idx / fps

        # ── Object Detection ─────────────────────────────────
        obj_results = obj_model(frame, verbose=False, imgsz=640)
        obj_boxes = obj_results[0].boxes

        annotated = frame.copy()

        if obj_boxes is not None and len(obj_boxes) > 0:
            for box in obj_boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = canonical_label(obj_model.names.get(cls_id, f"class_{cls_id}"))

                # Apply per-class threshold
                min_conf = CONFIDENCE_THRESHOLDS.get(label, 0.5)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                stats[label] += 1

                # Draw box
                color = (0, 0, 255) if label in ALERT_CLASSES else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )

                # ── Alert for cellphone / cheat_sheet ────────
                if label in ALERT_CLASSES:
                    alert_count += 1
                    alert(
                        f"🚨 {label.upper()} detected! (conf={conf:.0%})",
                        timestamp_sec,
                    )
                    save_detection_frame(
                        frame, video_name, timestamp_sec, label, conf
                    )

        # ── HUD overlay ──────────────────────────────────────
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}/{total_frames}  |  Time: {fmt_timestamp(timestamp_sec)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Alerts: {alert_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if alert_count > 0 else (0, 255, 0),
            2,
        )

        # ── Display ──────────────────────────────────────────
        if show_display:
            if disp_scale < 1.0:
                disp = cv2.resize(
                    annotated,
                    (int(width * disp_scale), int(height * disp_scale)),
                )
            else:
                disp = annotated

            cv2.imshow("AISENTINEL - Front Node Test", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                info("Quit requested.")
                break
            elif key == ord(" "):
                paused = True
                info("Paused. Press SPACE to resume.")

        # Progress every 500 frames
        if frame_idx % 500 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            info(f"Progress: {pct:.1f}%  ({frame_idx}/{total_frames} frames)")

    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  Summary for: {Path(video_path).name}")
    print("-" * 65)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Total alerts     : {alert_count}")
    for label, count in sorted(stats.items()):
        print(f"    {label:20s}: {count} detections")
    print(f"  Screenshots saved to: {DETECTION_OUTPUT_DIR}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def select_video_interactive() -> list[str]:
    """Interactive prompt to enter video file path(s)."""
    print()
    print("=" * 55)
    print("  AISENTINEL - Front Node Object Detection Test (PC)")
    print("=" * 55)
    print()
    paths = []
    while True:
        p = input("  Enter video file path (or 'done' to start): ").strip().strip('"').strip("'")
        if p.lower() in ("done", "d", ""):
            if paths:
                break
            print("  Please provide at least one video file.")
            continue
        if not os.path.isfile(p):
            print(f"  [ERROR] File not found: {p}")
            continue
        paths.append(p)
        print(f"  ✓ Added: {Path(p).name}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="AISENTINEL Front Node - PC Test (Object Detection Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python front_node_obj_model_pc.py --video exam_clip.mp4
  python front_node_obj_model_pc.py --video clip1.mp4 clip2.mp4
  python front_node_obj_model_pc.py --video exam.mp4 --no-display
  python front_node_obj_model_pc.py  # interactive mode
        """,
    )
    parser.add_argument(
        "--video", "-v",
        nargs="+",
        help="Path(s) to video file(s) to process",
    )
    parser.add_argument(
        "--obj-model",
        type=str,
        default=str(OBJ_MODEL_PATH),
        help=f"Path to object detection model (default: {OBJ_MODEL_PATH.name})",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a display window (headless)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve video paths
    if args.video:
        video_paths = args.video
    else:
        video_paths = select_video_interactive()

    # Validate video files
    for vp in video_paths:
        if not os.path.isfile(vp):
            print(f"{TermColor.RED}[ERROR] Video not found: {vp}{TermColor.RESET}")
            sys.exit(1)

    # Validate model
    if not os.path.isfile(args.obj_model):
        print(f"{TermColor.RED}[ERROR] Object detection model not found: {args.obj_model}{TermColor.RESET}")
        print("  Expected a front-node or sentinel .pt model under models/archive/")
        sys.exit(1)

    # Load model
    info(f"Loading object detection model: {args.obj_model}")
    obj_model = YOLO(args.obj_model)
    info("Object detection model loaded.")

    # Print class map read from the loaded model
    print()
    print(f"{TermColor.BOLD}Classes loaded from model:{TermColor.RESET}")
    for idx, name in obj_model.names.items():
        marker = " ⚠️  ALERT" if name in ALERT_CLASSES else ""
        print(f"  [{idx}] {name} (conf ≥ {CONFIDENCE_THRESHOLDS.get(name, 0.5):.0%}){marker}")
    print()

    # Process each video
    for vp in video_paths:
        process_video(vp, obj_model, show_display=not args.no_display)

    info("All videos processed. Done!")


if __name__ == "__main__":
    main()
