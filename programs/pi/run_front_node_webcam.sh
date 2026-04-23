#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=programs/pi/_common.sh
source "${SCRIPT_DIR}/_common.sh"
run_python_launcher "programs/run_front_node_webcam.py" "front_node_webcam"

