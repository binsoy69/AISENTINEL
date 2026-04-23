#!/usr/bin/env python3
"""Calibrate the front node video setup profile using the configured video."""

from _launcher_common import run_node_video_calibration


if __name__ == "__main__":
    run_node_video_calibration("node_front_video.ini")

