# AISENTINEL Project Overview

AISENTINEL is a dual-node real-time exam proctoring system. The current
implementation has a central dashboard service plus two Raspberry Pi node
agents. Each node runs detection locally, streams raw/annotated previews,
records evidence, and syncs incident manifests/assets to the central dashboard.

## Current Architecture

- Central dashboard: `runtime/central_dashboard/central_service/`
- Node agent: `runtime/central_dashboard/node_agent/`
- Shared DTOs/HTTP helpers: `runtime/central_dashboard/shared/`
- Real Hailo-backed detector runtime: `runtime/front_node_pi/`
- User-facing launchers: `programs/`
- Legacy/reference scripts: `tests/tests_on_pi/` and `tests/tests_on_pc/`

The front and mid nodes both use the same Hailo-backed runtime logic through the
central-dashboard node agent. The node identity, camera label, source mode, and
runtime config are selected by INI files.

## Main Programs

Deployment:

- `programs/run_central_dashboard.py`
- `programs/run_front_node_webcam.py`
- `programs/run_mid_node_webcam.py`

Video replay:

- `programs/run_front_node_video.py`
- `programs/run_mid_node_video.py`

Calibration:

- `programs/calibrate_front_webcam.py`
- `programs/calibrate_mid_webcam.py`
- `programs/calibrate_front_video.py`
- `programs/calibrate_mid_video.py`

Testing:

- `programs/test_central_dashboard.py`
- `programs/test_camera_preview.py`
- `programs/test_hailo_detection.py`
- `programs/test_sound_sensor.py`

Packaging:

- `programs/build_central_dashboard_exe.py`

## Runtime Configs

- `runtime/central_dashboard/central_service.ini`: central dashboard bind
  address, browser auth, central DB/evidence storage, and registered node keys.
- `runtime/central_dashboard/node_front.ini`: front node webcam deployment.
- `runtime/central_dashboard/node_mid.ini`: mid node webcam deployment.
- `runtime/central_dashboard/node_front_video.ini`: front node video replay.
- `runtime/central_dashboard/node_mid_video.ini`: mid node video replay.
- `runtime/central_dashboard/node_front_runtime.ini`: front node models,
  thresholds, setup profiles, video defaults, evidence output, and KY-037 sound.
- `runtime/central_dashboard/node_mid_runtime.ini`: mid node models,
  thresholds, setup profiles, video defaults, and evidence output.

Paths in repo INI files are resolved relative to the repository root. The
Windows packaged central dashboard EXE uses its external `central_service.ini`
beside the EXE and resolves relative data/evidence paths from that EXE folder.

## Detection Coverage

The Hailo-backed node runtime detects:

- head tilt
- shoulder turn
- passing papers
- hands under table
- phone
- cheat_sheet
- front-node KY-037 noise threshold incidents when enabled

The live annotated preview is intentionally evidence-style. It only draws boxes
around confirmed incident students and phone/cheat_sheet objects. Diagnostic
overlays stay out of the dashboard stream.

## Operational Flow

1. Start the central dashboard on the host machine.
2. Start the front and mid node agents on their Raspberry Pis.
3. Create an exam session in the central browser dashboard.
4. Start both nodes from the dashboard.
5. Each node runs local detection and streams raw/annotated feeds.
6. Incidents and evidence sync to the central dashboard.
7. Review, verify, filter, and export evidence from the dashboard.

## Raspberry Pi Double-Click Support

`programs/pi/` contains `.desktop` launchers and `.sh` scripts. They run the
current repo configs and append logs under:

```text
runtime/central_dashboard/data/logs/
```

Make them executable once on the Pi:

```bash
chmod +x programs/pi/*.sh "programs/pi/"*.desktop
```

## Windows EXE Support

Build the central dashboard EXE on Windows with:

```bash
python -m pip install flask pyinstaller
python programs/build_central_dashboard_exe.py
```

The output is a one-folder app under `dist/AISENTINEL Central Dashboard/`.
Edit `central_service.ini` beside the EXE and restart the EXE for config changes
to apply.

## Verification

Run:

```bash
python -m unittest discover runtime/central_dashboard/tests
```

Full Hailo detection validation still requires Raspberry Pi OS with `hailo-all`,
the HEF model files, and calibrated setup profiles.
