#!/usr/bin/env python3
"""Run the KY-037 ADS1015 raw-value hardware test."""

from _launcher_common import run_sound_sensor_raw_test


if __name__ == "__main__":
    run_sound_sensor_raw_test("front_node.ini")
