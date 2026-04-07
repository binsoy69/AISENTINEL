#!/usr/bin/env python3
"""Run a standalone node agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from central_dashboard.node_agent.app import create_app
from central_dashboard.node_agent.config import load_node_agent_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AISENTINEL node agent.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "node_front.ini"),
        help="Path to the node agent INI file.",
    )
    args = parser.parse_args()

    config = load_node_agent_config(args.config)
    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
