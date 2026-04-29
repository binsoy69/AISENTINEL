#!/usr/bin/env python3
"""Run the mid Raspberry Pi node agent in webcam deployment mode."""

from _launcher_common import run_node_agent


if __name__ == "__main__":
    run_node_agent("mid_node.ini", source_mode="webcam")
