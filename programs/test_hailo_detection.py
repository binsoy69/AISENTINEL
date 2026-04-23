#!/usr/bin/env python3
"""Run the standalone Hailo object-detection hardware test."""

from _launcher_common import REPO_ROOT, run_script


if __name__ == "__main__":
    run_script(REPO_ROOT / "tests" / "tests_on_pi" / "hailo_detection_test.py")

