#!/usr/bin/env python3
"""Run the front Raspberry Pi node agent against the configured video file."""

from _launcher_common import run_node_agent


if __name__ == "__main__":
    run_node_agent("node_front_video.ini", require_video=True)

