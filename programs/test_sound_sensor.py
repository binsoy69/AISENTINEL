#!/usr/bin/env python3
"""Run the KY-037 ADS1015 sound-threshold hardware test."""

from _launcher_common import run_sound_sensor_test


if __name__ == "__main__":
    run_sound_sensor_test("front_node.ini")
