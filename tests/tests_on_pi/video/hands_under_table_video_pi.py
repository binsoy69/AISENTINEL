#!/usr/bin/env python3
"""Video launcher for the Pi hands-under-table test."""

from pathlib import Path
import sys


PI_TEST_DIR = Path(__file__).resolve().parents[1]
if str(PI_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(PI_TEST_DIR))

from front_node_hands_under_table_pi import main  # noqa: E402


if __name__ == "__main__":
    main()
