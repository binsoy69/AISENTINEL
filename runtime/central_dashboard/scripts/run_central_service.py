#!/usr/bin/env python3
"""Run the standalone central dashboard service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from central_dashboard.central_service.app import create_app
from central_dashboard.central_service.config import load_central_service_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AISENTINEL central dashboard service.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "central_service.ini"),
        help="Path to the central service INI file.",
    )
    args = parser.parse_args()

    config = load_central_service_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
