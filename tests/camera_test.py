#!/usr/bin/env python3
"""Local camera preview that mirrors the main webcam runtime capture path."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUNTIME_DIR = REPO_ROOT / "runtime" / "front_node_pi"

if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from runtime_config import DEFAULT_WEBCAM_CONFIG_PATH, load_runtime_config
from runtime_support import (
    describe_webcam_capture,
    get_webcam_source_label,
    open_webcam_capture,
    read_webcam_frame,
)


WINDOW_NAME = "AISENTINEL Camera Preview"
MAX_CONSECUTIVE_READ_FAILURES = 30
HUD_TEXT_COLOR = (255, 255, 255)
HUD_SHADOW_COLOR = (0, 0, 0)
HUD_STATUS_COLOR = (40, 220, 120)


class PreviewLogger:
    """Minimal logger interface expected by runtime_support.open_webcam_capture."""

    @staticmethod
    def log_info(message: str) -> None:
        print(f"[INFO] {message}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open the webcam using the same capture logic as the main front-node "
            "runtime, but show it only in a local preview window with no models, "
            "no web server, and no setup profile overlays."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/camera_test.py
  python tests/camera_test.py --camera 1
  python tests/camera_test.py --width 1920 --height 1080 --fps 30
  python tests/camera_test.py --config runtime/front_node_pi/config_webcam.ini
        """,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional webcam runtime INI file. Defaults to "
            f"{DEFAULT_WEBCAM_CONFIG_PATH.relative_to(REPO_ROOT)}."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Override the configured camera index.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Override the configured source label.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override capture height.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override target capture FPS.",
    )
    return parser


def _apply_cli_overrides(config, args):
    webcam_cfg = replace(
        config.webcam_source,
        camera_index=(
            args.camera
            if args.camera is not None
            else config.webcam_source.camera_index
        ),
        camera_name=(
            args.camera_name
            if args.camera_name is not None
            else config.webcam_source.camera_name
        ),
        capture_width=(
            args.width
            if args.width is not None
            else config.webcam_source.capture_width
        ),
        capture_height=(
            args.height
            if args.height is not None
            else config.webcam_source.capture_height
        ),
        capture_fps=(
            args.fps
            if args.fps is not None
            else config.webcam_source.capture_fps
        ),
        default_setup_profile=None,
        auto_use_saved_setup=False,
    )
    return replace(config, webcam_source=webcam_cfg)


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _is_reasonable_fps(value: float) -> bool:
    return 0.0 < value <= 120.0


def _draw_hud_line(frame, text: str, x: int, y: int, color, scale: float = 0.62) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        HUD_SHADOW_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_fps_badge(frame, fps_value: float, color) -> None:
    badge_text = f"FPS {fps_value:.1f}"
    (badge_w, badge_h), _ = cv2.getTextSize(
        badge_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2,
    )
    frame_h, frame_w = frame.shape[:2]
    badge_x1 = frame_w - badge_w - 28
    badge_y1 = 12
    badge_x2 = frame_w - 10
    badge_y2 = badge_y1 + badge_h + 16
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)
    cv2.putText(
        frame,
        badge_text,
        (badge_x1 + 10, badge_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def _annotate_preview_frame(
    frame,
    source_label: str,
    capture_description: str,
    source_fps: float,
    preview_fps: float,
    frame_idx: int,
    elapsed_sec: float,
):
    annotated = frame.copy()
    hud_lines = [
        "AISENTINEL Preview | Models OFF",
        f"Source: {source_label}",
        f"Capture: {capture_description}",
        f"Frame: {frame_idx} | Live time: {_format_elapsed(elapsed_sec)}",
        f"Camera FPS: {source_fps:.1f} | Preview FPS: {preview_fps:.1f}",
    ]

    for index, line in enumerate(hud_lines):
        color = HUD_STATUS_COLOR if index == 0 else HUD_TEXT_COLOR
        _draw_hud_line(annotated, line, 12, 30 + (index * 28), color)

    frame_h = annotated.shape[0]
    _draw_hud_line(
        annotated,
        "Press Q or ESC to exit",
        12,
        frame_h - 16,
        HUD_TEXT_COLOR,
        scale=0.56,
    )
    _draw_fps_badge(annotated, max(0.0, preview_fps), HUD_STATUS_COLOR)
    return annotated


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config, DEFAULT_WEBCAM_CONFIG_PATH)
    config = _apply_cli_overrides(config, args)
    logger = PreviewLogger()
    source_label = get_webcam_source_label(config)
    cap = None

    print()
    print("=" * 78)
    print("  AISENTINEL - Local Camera Preview")
    print(f"  Config          : {config.config_path}")
    print(f"  Source          : {source_label} (index {config.webcam_source.camera_index})")
    print("  Mode            : local OpenCV preview only")
    print("  Models          : disabled")
    print("  Setup profile   : disabled")
    print("  Web dashboard   : disabled")
    print("=" * 78)
    print()

    try:
        cap = open_webcam_capture(config, logger)
        first_frame = read_webcam_frame(cap, attempts=5, pause_sec=0.02)
        if first_frame is None:
            raise RuntimeError("Cannot read the first webcam frame after opening the camera.")

        capture_description = describe_webcam_capture(cap)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not _is_reasonable_fps(source_fps):
            source_fps = config.webcam_source.capture_fps
        if not _is_reasonable_fps(source_fps):
            source_fps = 30.0

        logger.log_info(f"Preview capture mode: {capture_description}")
        logger.log_info("Preview window ready. Press Q or ESC to stop.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        frame_idx = 0
        consecutive_read_failures = 0
        started_at = time.perf_counter()
        last_loop_at = started_at
        smoothed_fps = source_fps

        while True:
            frame = read_webcam_frame(cap, attempts=1, pause_sec=0.0)
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError("Camera feed stopped returning frames.")
                time.sleep(0.03)
                continue

            consecutive_read_failures = 0
            frame_idx += 1

            now = time.perf_counter()
            elapsed = now - started_at
            delta = now - last_loop_at
            last_loop_at = now
            instant_fps = (1.0 / delta) if delta > 0 else 0.0
            if instant_fps > 0:
                smoothed_fps += (instant_fps - smoothed_fps) * 0.2

            annotated = _annotate_preview_frame(
                frame,
                source_label=source_label,
                capture_description=capture_description,
                source_fps=source_fps,
                preview_fps=smoothed_fps,
                frame_idx=frame_idx,
                elapsed_sec=elapsed,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        logger.log_info("Interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
