#!/usr/bin/env python3
"""Calibrate the mid node KY-037 sound sensor."""

from _launcher_common import run_node_sound_calibration


if __name__ == "__main__":
    run_node_sound_calibration("mid_node.ini")
