#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if command -v xdg-user-dir >/dev/null 2>&1; then
    DESKTOP_DIR="$(xdg-user-dir DESKTOP)"
else
    DESKTOP_DIR="${HOME}/Desktop"
fi

mkdir -p "${DESKTOP_DIR}"

write_launcher() {
    local filename="$1"
    local name="$2"
    local comment="$3"
    local script_relpath="$4"
    local script_path="${REPO_ROOT}/${script_relpath}"
    local desktop_path="${DESKTOP_DIR}/${filename}"

    if [[ ! -f "${script_path}" ]]; then
        echo "[ERROR] Missing script: ${script_path}" >&2
        return 1
    fi

    cat > "${desktop_path}" <<EOF
[Desktop Entry]
Type=Application
Name=${name}
Comment=${comment}
Exec=env AISENTINEL_KEEP_TERMINAL=1 bash "${script_path}"
Path=${REPO_ROOT}
Terminal=true
Categories=Utility;
EOF

    chmod +x "${desktop_path}"

    if command -v gio >/dev/null 2>&1; then
        gio set "${desktop_path}" metadata::trusted true >/dev/null 2>&1 || true
    fi

    echo "[OK] Installed: ${desktop_path}"
}

write_launcher \
    "AISENTINEL Front Node Webcam.desktop" \
    "AISENTINEL Front Node Webcam" \
    "Run the front node with its configured webcam" \
    "programs/pi/run_front_node_webcam.sh"

write_launcher \
    "AISENTINEL Mid Node Webcam.desktop" \
    "AISENTINEL Mid Node Webcam" \
    "Run the mid node with its configured webcam" \
    "programs/pi/run_mid_node_webcam.sh"

write_launcher \
    "AISENTINEL Front Node Video.desktop" \
    "AISENTINEL Front Node Video" \
    "Run the front node with its configured video source" \
    "programs/pi/run_front_node_video.sh"

write_launcher \
    "AISENTINEL Mid Node Video.desktop" \
    "AISENTINEL Mid Node Video" \
    "Run the mid node with its configured video source" \
    "programs/pi/run_mid_node_video.sh"

write_launcher \
    "AISENTINEL Calibrate Front Webcam.desktop" \
    "AISENTINEL Calibrate Front Webcam" \
    "Create or update the front node webcam setup profile" \
    "programs/pi/calibrate_front_webcam.sh"

write_launcher \
    "AISENTINEL Calibrate Mid Webcam.desktop" \
    "AISENTINEL Calibrate Mid Webcam" \
    "Create or update the mid node webcam setup profile" \
    "programs/pi/calibrate_mid_webcam.sh"

write_launcher \
    "AISENTINEL Calibrate Front Video.desktop" \
    "AISENTINEL Calibrate Front Video" \
    "Create or update the front node video setup profile" \
    "programs/pi/calibrate_front_video.sh"

write_launcher \
    "AISENTINEL Calibrate Mid Video.desktop" \
    "AISENTINEL Calibrate Mid Video" \
    "Create or update the mid node video setup profile" \
    "programs/pi/calibrate_mid_video.sh"

write_launcher \
    "AISENTINEL Calibrate Front Sound.desktop" \
    "AISENTINEL Calibrate Front Sound" \
    "Create or update the front node KY-037 sound calibration" \
    "programs/pi/calibrate_front_sound_sensor.sh"

write_launcher \
    "AISENTINEL Calibrate Mid Sound.desktop" \
    "AISENTINEL Calibrate Mid Sound" \
    "Create or update the mid node KY-037 sound calibration" \
    "programs/pi/calibrate_mid_sound_sensor.sh"

echo
echo "Desktop launchers installed."
echo "If Raspberry Pi Desktop asks whether to trust/execute them, choose Allow Launching."
