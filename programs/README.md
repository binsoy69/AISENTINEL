# AISENTINEL Programs

This folder contains no-argument launchers intended for IDE Run buttons,
double-click workflows, and beginner-friendly terminal usage.

## Central Dashboard

- `run_central_dashboard.py`: starts the central Flask dashboard using
  `config/central.ini`.
- `build_central_dashboard_exe.py`: builds the Windows one-folder central
  dashboard EXE with PyInstaller.
- `central_service_exe.ini`: editable config template copied beside the EXE.

## Raspberry Pi Node Agents

- `run_front_node_webcam.py`: front node live webcam deployment.
- `run_mid_node_webcam.py`: mid node live webcam deployment.
- `run_front_node_video.py`: front node video replay using a file picker.
- `run_mid_node_video.py`: mid node video replay using a file picker.

Video replay opens a file picker, saves the selected file to
`[video_source] default_video` in the matching node INI, then starts the node:

- `config/front_node.ini`
- `config/mid_node.ini`

## Calibration

- `calibrate_front_webcam.py`: save/update the front webcam setup profile.
- `calibrate_mid_webcam.py`: save/update the mid webcam setup profile.
- `calibrate_front_video.py`: pick a front video and save/update its setup
  profile.
- `calibrate_mid_video.py`: pick a mid video and save/update its setup profile.

## Testing

- `test_central_dashboard.py`: runs the central-dashboard unit/integration tests.
- `test_camera_preview.py`: opens the webcam using the runtime capture path,
  without AI models.
- `test_hailo_detection.py`: runs the standalone Hailo object-detection test.
- `test_sound_sensor.py`: runs the KY-037 ADS1015 sound-threshold test.

## Raspberry Pi Double-Click Files

The `pi/` folder contains `.sh` scripts, source `.desktop` launchers, and an
installer:

```bash
bash programs/pi/install_desktop_launchers.sh
```

Run that installer on the Raspberry Pi. It writes absolute-path launchers to the
Pi Desktop, marks them executable, and tries to mark them trusted. The generated
`.desktop` files call the `.sh` scripts, and the scripts call the Python
launchers above.

Logs are appended under:

```text
runtime/central_dashboard/data/logs/
```

Config changes apply on the next launcher run. Real local configs live under
`config/*.ini`; tracked examples live beside them as `*.ini.example`.
