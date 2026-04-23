#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=programs/pi/_common.sh
source "${SCRIPT_DIR}/_common.sh"
run_python_launcher "programs/calibrate_front_webcam.py" "calibrate_front_webcam"

