#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

run_python_launcher() {
    local launcher_path="$1"
    local launcher_name="$2"

    cd "${REPO_ROOT}"
    mkdir -p runtime/central_dashboard/data/logs

    local log_file="runtime/central_dashboard/data/logs/${launcher_name}_$(date +%Y%m%d).log"
    exec > >(tee -a "${log_file}") 2>&1

    echo "=============================================================================="
    echo "AISENTINEL launcher: ${launcher_name}"
    echo "Started: $(date)"
    echo "Repo: ${REPO_ROOT}"
    echo "Log: ${REPO_ROOT}/${log_file}"
    echo "=============================================================================="

    if [[ -f "venv/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "venv/bin/activate"
    elif [[ -f ".venv/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source ".venv/bin/activate"
    fi

    python3 "${launcher_path}"
}

