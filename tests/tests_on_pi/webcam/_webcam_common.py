#!/usr/bin/env python3
"""Shared webcam helpers for Pi behavior test launchers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2


PI_TEST_DIR = Path(__file__).resolve().parents[1]
if str(PI_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(PI_TEST_DIR))


def add_webcam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to open (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Requested capture FPS (default: 30)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames to discard before the calibration frame (default: 10)",
    )
    parser.add_argument(
        "--no-mjpg",
        action="store_true",
        help="Do not request MJPG capture from the webcam",
    )


def webcam_source_label(camera_index: int) -> str:
    return f"webcam_{camera_index}"


def _iter_open_sources(camera_index: int):
    device_path = Path(f"/dev/video{camera_index}")
    if device_path.exists():
        yield str(device_path), str(device_path)
    yield camera_index, f"index {camera_index}"


def _iter_backends():
    if hasattr(cv2, "CAP_V4L2"):
        yield "V4L2", cv2.CAP_V4L2
    yield "default", None


def configure_camera(cap, width: int, height: int, fps: float, use_mjpg: bool) -> None:
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def read_latest_frame(cap, attempts: int = 30, pause_sec: float = 0.04):
    latest = None
    for _ in range(max(1, attempts)):
        ret, frame = cap.read()
        if ret and frame is not None:
            latest = frame
        elif latest is not None:
            break
        time.sleep(pause_sec)
    return latest


def read_warmup_frame(cap, warmup_frames: int):
    for _ in range(max(0, warmup_frames)):
        cap.read()
    return read_latest_frame(cap)


def open_webcam(camera_index: int, width: int, height: int, fps: float, use_mjpg: bool):
    errors = []
    for source, source_name in _iter_open_sources(camera_index):
        for backend_name, backend_id in _iter_backends():
            cap = (
                cv2.VideoCapture(source, backend_id)
                if backend_id is not None else cv2.VideoCapture(source)
            )
            if not cap.isOpened():
                errors.append(f"{source_name} via {backend_name}: open failed")
                cap.release()
                continue

            configure_camera(cap, width, height, fps, use_mjpg)
            frame = read_latest_frame(cap, attempts=8, pause_sec=0.03)
            if frame is not None:
                return cap, f"{source_name} via {backend_name}"

            errors.append(f"{source_name} via {backend_name}: no readable frame")
            cap.release()

    detail = "\n  ".join(errors) if errors else "no open attempts were made"
    raise RuntimeError(f"Cannot open webcam {camera_index}:\n  {detail}")


def capture_fps(cap, requested_fps: float) -> float:
    actual = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if actual <= 0 or actual > 120:
        return requested_fps if requested_fps > 0 else 30.0
    return actual


def require_file(path: Path, label: str, color) -> None:
    if not path.exists():
        print(f"{color.RED}[ERROR] {label} not found: {path}{color.RESET}")
        raise SystemExit(1)


def require_hailo(available: bool, color) -> None:
    if not available:
        print(f"{color.RED}[ERROR] hailo_platform is required.{color.RESET}")
        print("Install: sudo apt install hailo-all")
        raise SystemExit(1)


def require_flask(available: bool, color) -> None:
    if not available:
        print(f"{color.RED}[ERROR] Flask is required for web streaming.{color.RESET}")
        print("Install: pip install flask")
        raise SystemExit(1)
