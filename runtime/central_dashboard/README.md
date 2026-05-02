# AISENTINEL Central Dashboard Runtime

This folder contains the dual-node monitoring stack used by the shared central
dashboard. The central service runs on a laptop or host PC. Each Raspberry Pi
node agent runs local detection, serves raw/annotated MJPEG streams, and uploads
incidents/evidence live to the central service.

## Recommended Programs

Use the no-argument launchers in `programs/` from the repository root.

Central dashboard host:

```bash
python programs/run_central_dashboard.py
```

Front and mid webcam deployment:

```bash
python programs/run_front_node_webcam.py
python programs/run_mid_node_webcam.py
```

Front and mid configured video replay:

```bash
python programs/run_front_node_video.py
python programs/run_mid_node_video.py
```

Tests:

```bash
python programs/test_central_dashboard.py
```

Hardware checks:

```bash
python programs/test_camera_preview.py
python programs/test_hailo_detection.py
python programs/test_sound_sensor.py
```

## Layout

- `central_service/`: Flask dashboard, SQLite persistence, evidence storage,
  browser UI, and node stream proxy.
- `node_agent/`: Raspberry Pi agent API, local runtime controller, preview
  stream, live uploads, and detector backend selection.
- `../../config/*.ini.example`: tracked operator config examples. Copy these to
  `.ini` files under `config/` and edit local values; real `.ini` files are
  intentionally ignored.
- `shared/`: DTOs, JSON helpers, and HTTP client helpers shared by both apps.
- `scripts/`: lower-level argparse entrypoints still available for advanced
  manual use.
- `data/`: runtime-created databases, evidence, logs, and queue state.
- `tests/`: unit and integration tests for this stack.

## Configs To Edit

Central service:

```text
config/central.ini
```

Important fields:

- `[service] host`
- `[service] port`
- `[browser_auth] username`
- `[browser_auth] password`
- `[node:front] api_key`
- `[node:mid] api_key`

Node agents:

```text
config/front_node.ini
config/mid_node.ini
```

Important fields:

- `[agent] host`
- `[agent] port`
- `[agent] central_base_url`
- `[capture] source_mode`
- `[capture] camera_index`
- `[models] pose`
- `[models] hand`
- `[models] object`
- `[webcam_source] default_setup_profile`
- `[video_source] default_video`
- `[video_source] default_setup_profile`
- `[evidence] root`

## Network Setup

Keep local bind hosts set to `0.0.0.0` for most deployments. Set each node
`central_base_url` to the actual LAN IP of the dashboard host.

Example:

```ini
; central-service laptop
[service]
host = 0.0.0.0
port = 8090

; front-node Pi
[agent]
host = 0.0.0.0
port = 8091
central_base_url = http://192.168.1.50:8090

; mid-node Pi
[agent]
host = 0.0.0.0
port = 8092
central_base_url = http://192.168.1.50:8090
```

With that setup:

1. Start the central service on the host.
2. Start each node agent on its own Pi.
3. Open `http://192.168.1.50:8090` in a browser.

Use `127.0.0.1` only when the central service and node agent are on the same
machine for local testing.

## Live Uploads

The node agent no longer keeps a durable evidence backlog. Each incident
manifest and evidence asset is uploaded immediately, retried once immediately
on failure, then dropped and counted if central is still unavailable. Startup
and new-session startup clear stale local queue rows from older versions.

Dashboard system status exposes `Dropped Uploads`, `Last Drop`, and the latest
drop reason per node. Valid evidence already stored on central remains until an
operator clears records, deletes a session, or deletes a subject.

## Alert And GIF Timing

The central dashboard shows an alert as soon as a node confirms a behavior
event and starts recording evidence. The Records tab shows that row as
`Evidence processing` until media arrives.

Visual incidents save `poster.jpg` plus a compact `evidence.gif` with exactly
five frames: two pre-event frames, the event frame, and two post-event frames.
The GIF is capped at 640px wide and encoded at 4 FPS. Noise incidents remain
poster-only.

## Calibration

Webcam calibration:

```bash
python programs/calibrate_front_webcam.py
python programs/calibrate_mid_webcam.py
```

Video calibration uses each node INI `[video_source] default_video`:

```bash
python programs/calibrate_front_video.py
python programs/calibrate_mid_video.py
```

Calibration saves setup profiles and updates the matching node INI default
profile path.

Sound calibration for KY-037 + ADS1015:

```bash
python programs/calibrate_front_sound_sensor.py
python programs/calibrate_mid_sound_sensor.py
```

The sound calibrator captures quiet and loud references, writes the JSON under
`runtime/central_dashboard/data/node_<id>/sound/`, updates `[sound_sensor]
calibration_config` in the matching node INI, and enables sound monitoring once
both references are present.

## Video Replay

Set:

```ini
[video_source]
default_video = test-videos/Frontcam-set1-001.mp4
```

If the default video is blank or missing, the launcher exits with a clear
message naming the field to edit.

## Live Preview

Each node exposes raw, annotated, and debug streams. For `front_runtime`, the
annotated stream is evidence-style: it only boxes confirmed incident students
and phone/cheat_sheet objects. The debug stream shows the full diagnostic
overlay, including skeletons, detection boxes, ROI lines, desk lines, FPS HUD,
and diagnostic labels.

## Raspberry Pi Double-Click Launchers

Raspberry Pi launchers live in:

```text
programs/pi/
```

Recommended first-time setup on each Pi:

```bash
bash programs/pi/install_desktop_launchers.sh
```

This creates launchers on the Pi Desktop with absolute paths. Double-click one
of those launchers, or run the `.sh` scripts directly from a terminal. Output
appears in the terminal, the terminal stays open after failures, and logs are
appended to:

```text
runtime/central_dashboard/data/logs/
```

Config changes apply on the next launch because the scripts always read the
current local `config/*.ini` files.

If Raspberry Pi Desktop asks whether to trust or execute the launcher, choose
`Allow Launching`.

## Windows Central Dashboard EXE

Build on Windows:

```bash
python -m pip install flask pyinstaller
python programs/build_central_dashboard_exe.py
```

Output:

```text
dist/AISENTINEL Central Dashboard/
```

Run:

```text
AISENTINEL Central Dashboard.exe
```

Edit this file beside the EXE:

```text
central_service.ini
```

Restart the EXE after config changes.

## Detector Backends

`front_runtime`

- Uses the real Hailo-backed all-behavior pipeline from `runtime/edge_node_runtime`.
- Produces student-number incidents, evidence frames/GIFs, and live previews.
- On the front node, can report KY-037 classroom-noise telemetry and upload noise
  incidents when `[sound_sensor] enabled = true`.

`motion`

- Lightweight fallback used for hardware-free automated tests.
- Still available through node-agent config for development.

## Troubleshooting

`Cannot assign requested address` means the configured local `host` is not an IP
address on that machine. Use `host = 0.0.0.0` unless you intentionally bind to a
specific interface.

`Address already in use` means the port is already occupied. Change the matching
`port` value or stop the other process.

If nodes cannot reach the central dashboard, check:

- node `central_base_url`
- host firewall rules for ports `8090`, `8091`, and `8092`
- both machines are on the same reachable network
