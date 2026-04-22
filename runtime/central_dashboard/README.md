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
- The front node can also publish live KY-037 classroom-noise telemetry through
  its heartbeat when `sound_sensor.enabled = true` in the front-node runtime
  INI.
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

## Network Setup

The network settings that usually matter are:

- `runtime/central_dashboard/central_service.ini`:
  - `[service] host`
  - `[service] port`
- `runtime/central_dashboard/node_front.ini`:
  - `[agent] host`
  - `[agent] port`
  - `[agent] central_base_url`
- `runtime/central_dashboard/node_mid.ini`:
  - `[agent] host`
  - `[agent] port`
  - `[agent] central_base_url`

What each field means:

- `host`: the local bind address for that process on that machine
- `port`: the local TCP port that process listens on
- `central_base_url`: the remote URL that a node agent uses to reach the
  central service

In most real deployments, keep every local `host` set to `0.0.0.0`. That
means "listen on all network interfaces on this machine". Other devices should
then connect to the machine's actual LAN IP, not to `0.0.0.0`.

Example:

- central-service laptop IP: `192.168.1.50`
- front-node Pi IP: `192.168.1.61`
- mid-node Pi IP: `192.168.1.62`

Use these settings:

```ini
; runtime/central_dashboard/central_service.ini
[service]
host = 0.0.0.0
port = 8090

; runtime/central_dashboard/node_front.ini
[agent]
host = 0.0.0.0
port = 8091
central_base_url = http://192.168.1.50:8090

; runtime/central_dashboard/node_mid.ini
[agent]
host = 0.0.0.0
port = 8092
central_base_url = http://192.168.1.50:8090
```

With that setup:

- start the central service on the laptop
- start each node agent on its own Pi
- open the dashboard in a browser at `http://192.168.1.50:8090`

If you are running the central service and a node agent on the same machine
for local testing only, `central_base_url = http://127.0.0.1:8090` is valid.
For real multi-machine runs, `127.0.0.1` is wrong because it points back to
the node itself, not to the laptop hosting the central dashboard.

### `Cannot Assign Requested Address`

`Cannot assign requested address` usually means the process tried to bind to an
IP address that does not exist on the current machine.

Common causes:

- setting `host` to the laptop IP inside a Pi config
- setting `host` to the Pi IP inside the laptop config
- using an old Wi-Fi IP after the machine moved to a different network
- binding to a specific address before that interface is up

Typical fix:

1. Find the actual IP of each machine.
2. Keep local `host = 0.0.0.0` unless you intentionally need to bind to one
   specific interface.
3. Set each node's `central_base_url` to the central-service machine's real
   reachable IP and port.
4. Open the browser against the central-service machine IP, for example
   `http://192.168.1.50:8090`.

Useful commands:

```bash
# Linux / Raspberry Pi OS
hostname -I
ip addr

# Windows
ipconfig
```

Related notes:

- If you set `host` to a specific IP, it must be an address already assigned
  to that same machine.
- `0.0.0.0` is only for binding. Do not type `http://0.0.0.0:8090` into
  another device and expect that to be the routable address.
- If startup fails with a different message such as `Address already in use`,
  the IP is fine but the port is already occupied. Change the matching `port`
  value instead.
- If nodes cannot reach the service even with the right IPs, check firewall
  rules for ports `8090`, `8091`, and `8092`.

## Front-Node Runtime INIs

`runtime/central_dashboard/node_front_runtime.ini` and
`runtime/central_dashboard/node_mid_runtime.ini` do not contain the central
service host IP. Their `[runtime] port` is only the local detector/dashboard
port used inside the node runtime path. Usually you do not need to change it
unless there is a port conflict on that same machine.

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
- On the front node, can also report estimated dB from the KY-037 + ADS1015,
  show live threshold warnings in the central dashboard, and sync saved
  `noise` incidents without media attachments

`motion`
- Keeps the original motion-anomaly fallback backend available
- Remains the default backend for automated integration tests
