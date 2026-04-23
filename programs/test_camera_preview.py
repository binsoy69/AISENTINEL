#!/usr/bin/env python3
"""Run the no-model webcam preview test."""

from _launcher_common import REPO_ROOT, run_script


if __name__ == "__main__":
    run_script(REPO_ROOT / "tests" / "camera_test.py")

