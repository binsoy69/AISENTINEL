# AISENTINEL Central Dashboard Runtime

This folder contains the multi-node monitoring stack used by the shared
central dashboard. The central service API stays stable while each node agent
can run either:

- `front_runtime`: the real Hailo-backed classroom detector reused from
  `runtime/front_node_pi`
- `motion`: the original lightweight fallback backend used for tests and
  hardware-free runs

## Layout

- `central_service/`: laptop-hosted Flask app, SQLite persistence, evidence
  storage, browser UI, and MJPEG feed proxy.
- `node_agent/`: Raspberry Pi agent that serves local feeds, runs local frame
  processing, stores evidence, and syncs incidents/evidence to central.
- `node_front.ini` / `node_mid.ini`: node launch configs for networking,
  capture mode, sync, preview, and detector selection.
- `node_front_runtime.ini` / `node_mid_runtime.ini`: per-node Hailo detector
  runtime configs for models, thresholds, evidence roots, and saved setup
  profile paths.
- `shared/`: DTOs, JSON helpers, and HTTP client helpers shared by both apps.
- `scripts/`: entrypoint scripts for running the central service and node
  agents plus node calibration helpers.
- `data/`: runtime-created databases, evidence, and queue state.
- `tests/`: unit and integration tests for the new stack.

## Default Runtime Model

- The central service owns the shared exam session and browser-facing dashboard.
- Nodes self-register with the central service using per-node API keys.
- Each node keeps detection local and exposes both raw and annotated preview
  streams.
- Evidence is saved locally first, then synced to the central service with
  retry-on-failure queueing.
- The shipped `front` and `mid` node configs default to `detector.mode =
  front_runtime`.

## Running

Run the central service:

```bash
python runtime/central_dashboard/scripts/run_central_service.py
```

Run a node agent:

```bash
python runtime/central_dashboard/scripts/run_node_agent.py --config runtime/central_dashboard/node_front.ini
python runtime/central_dashboard/scripts/run_node_agent.py --config runtime/central_dashboard/node_mid.ini
```

## Calibration

Each node keeps calibration as a JSON setup profile. The detector/runtime INI
stores the default profile path that the node agent should load on startup.

Calibrate a webcam-backed node:

```bash
python runtime/central_dashboard/scripts/calibrate_node_webcam.py --config runtime/central_dashboard/node_front.ini
python runtime/central_dashboard/scripts/calibrate_node_webcam.py --config runtime/central_dashboard/node_mid.ini
```

Calibrate against a test video and update the runtime INI defaults:

```bash
python runtime/central_dashboard/scripts/calibrate_node_video.py --config runtime/central_dashboard/node_front.ini --video test-videos/front.mp4
python runtime/central_dashboard/scripts/calibrate_node_video.py --config runtime/central_dashboard/node_mid.ini --video test-videos/mid.mp4
```

## Detector Backends

`front_runtime`
- Uses the real Hailo-backed all-behavior pipeline from `runtime/front_node_pi`
- Produces real student-number incidents, poster/gif evidence, and live
  annotated previews

`motion`
- Keeps the original motion-anomaly fallback backend available
- Remains the default backend for automated integration tests
