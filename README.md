# AISENTINEL Universal Run Guide

AISENTINEL is a dual-node exam monitoring system. A central dashboard runs on a
laptop or host PC, and two Raspberry Pi nodes stream camera feeds, run local
detection, save evidence, and upload incidents live to the dashboard.

## What To Run

### Central dashboard host

Run this on the laptop or PC that will show the browser dashboard:

```bash
python programs/run_central_dashboard.py
```

Open the dashboard at:

```text
http://<central-host-ip>:8090
```

For local testing on the same machine, use:

```text
http://127.0.0.1:8090
```

### Front Raspberry Pi, webcam deployment

```bash
python programs/run_front_node_webcam.py
```

### Mid Raspberry Pi, webcam deployment

```bash
python programs/run_mid_node_webcam.py
```

### Front and mid nodes, configured video playback

```bash
python programs/run_front_node_video.py
python programs/run_mid_node_video.py
```

The video launchers read `[video_source] default_video` from
`config/front_node.ini` and `config/mid_node.ini`.

## Main Program Groups

- Deployment: `programs/run_central_dashboard.py`,
  `programs/run_front_node_webcam.py`, `programs/run_mid_node_webcam.py`
- Calibration: `programs/calibrate_front_webcam.py`,
  `programs/calibrate_mid_webcam.py`, `programs/calibrate_front_video.py`,
  `programs/calibrate_mid_video.py`
- Video replay: `programs/run_front_node_video.py`,
  `programs/run_mid_node_video.py`
- Testing: `programs/test_central_dashboard.py`,
  `programs/test_camera_preview.py`, `programs/test_hailo_detection.py`,
  `programs/test_sound_sensor.py`
- Windows packaging: `programs/build_central_dashboard_exe.py`

## Important Config Files

- `config/central.ini`: dashboard host, port,
  browser login, known node API keys, central database/evidence paths.
- `config/front_node.ini`: front Pi node agent, capture source, models,
  thresholds, webcam setup profile, video default, evidence path, sound sensor.
- `config/mid_node.ini`: mid Pi node agent, capture source, models,
  thresholds, webcam setup profile, video default, evidence path.

Only examples are tracked: `config/*.ini.example`. Copy them to `.ini` files
and edit local values; real `config/*.ini` files are ignored.

For real multi-machine deployment, keep each process `host = 0.0.0.0`, then set
each node `central_base_url` to the central dashboard machine IP, for example:

```ini
central_base_url = http://192.168.1.50:8090
```

## Double-Click Launchers On Raspberry Pi

Raspberry Pi launchers are in `programs/pi/`.

Recommended first-time setup on each Pi:

```bash
bash programs/pi/install_desktop_launchers.sh
```

This creates absolute-path launchers on the Pi Desktop and marks them
executable/trusted where the desktop environment allows it. Then double-click
the desktop launcher for the node program you need.

The launcher opens a terminal, runs the matching Python program from the repo
root, keeps the terminal open after errors, and writes logs to:

```text
runtime/central_dashboard/data/logs/
```

Changing the local `config/*.ini` files takes effect the next time you run the
launcher.

If a copied `.desktop` file still does not open, right-click it and choose
`Allow Launching` if Raspberry Pi Desktop offers that option.

## Windows Central Dashboard EXE

Build a one-folder Windows EXE:

```bash
python -m pip install flask pyinstaller
python programs/build_central_dashboard_exe.py
```

The output folder is:

```text
dist/AISENTINEL Central Dashboard/
```

Run:

```text
AISENTINEL Central Dashboard.exe
```

The packaged app reads the editable external config beside the EXE:

```text
dist/AISENTINEL Central Dashboard/central_service.ini
```

If you edit that config, restart the EXE for changes to apply.

## Live Preview Behavior

The central dashboard has raw and annotated streams for each node. The annotated
stream is intentionally evidence-style: it only boxes confirmed cheating
students and phone or cheat_sheet objects during incidents. Diagnostic overlays
such as skeletons, hand boxes, ROI lines, and FPS HUD are not shown in the live
annotated dashboard feed.

## Automated Tests

Run the central-dashboard test suite:

```bash
python programs/test_central_dashboard.py
```

or:

```bash
python -m unittest discover runtime/central_dashboard/tests
```

Hardware test launchers are also available:

```bash
python programs/test_camera_preview.py
python programs/test_hailo_detection.py
python programs/test_sound_sensor.py
```
