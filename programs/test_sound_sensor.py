#!/usr/bin/env python3
"""Run the KY-037 ADS1015 sound-threshold hardware test."""

from _launcher_common import REPO_ROOT, run_script


if __name__ == "__main__":
    run_script(REPO_ROOT / "tests" / "tests_on_pi" / "ky037_sound_threshold_test.py")

