# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AISENTINEL is a dual-node real-time exam proctoring system using YOLO pose estimation and object detection to detect cheating behaviors. It targets two Raspberry Pi 5 units:

- **Front Node**: Hailo-8 AI HAT+ (26 TOPS) — covers all 20 students, runs pose + detection
- **Mid Node**: CPU-only (NCNN) — supplemental coverage of back 12 students, desk zone monitoring

## Running Tests

All test scripts are in `tests/`. Run from the repo root on a PC with a GPU or on the target Raspberry Pi:

```bash
# Head behavior detection (head tilt, look-at-neighbor, shoulder turn)
python tests/front_node_head_behavior_pc.py

# Interactive threshold calibration tool (real-time slider adjustments)
python tests/calibrate_head_behavior.py

# Pose estimation test
python tests/front_node_pose_test_pc.py

# Hands-under-table detection
python tests/front_node_hands_under_table_pc.py

# Cellphone / cheat sheet object detection
python tests/front_node_cellphone_cheat_pc.py

# USB camera diagnostics (list cameras, benchmark FPS, capture test images)
python tests/camera_test.py

# Mid Node inference test (.pt or NCNN)
python tests/BackNodeTest.py
python tests/BackNodeTest.py --export-ncnn   # also export to NCNN format
```

Each PC test script opens a **file dialog** to select a video file, then an interactive student assignment window before starting detection.

## Dependencies

```bash
pip install ultralytics opencv-python numpy lap
```

For Hailo hardware tests (Raspberry Pi only): `hailo-platform`, `hailort`. For Mid Node: `ncnn`.

## Architecture

### Behavioral Detection Pipeline (Front Node)

Each `front_node_*_pc.py` test follows the same pattern:
1. **Video selection** → file dialog (`tkinter`)
2. **Assignment phase** → `run_assignment_phase()`: click persons on first frame, assign student numbers, keyed by ByteTrack ID
3. **Detection loop** → `run_detection()`: `model.track(persist=True)` with ByteTrack maintains consistent track IDs; per-`StudentState` timers track sustained behavior durations
4. **Alerting**: behavior must exceed threshold AND be sustained ≥ `SUSTAINED_SEC` (3s) before flagging; then `EVENT_COOLDOWN_SEC` (10s) between repeated flags
5. **Evidence**: annotated screenshots saved to `tests/evidence/`

### Key Thresholds (in `front_node_head_behavior_pc.py`)

| Behavior | Parameter | Value |
|---|---|---|
| Head tilt (roll) | `HEAD_TILT_ANGLE_DEG` | 30° (ear-to-ear atan2) |
| Head tilt (yaw) | `HEAD_TURN_RATIO` | 0.26 (nose offset / shoulder width) |
| Shoulder turn (overhead) | `SHOULDER_TURN_ANGLE_DEG` | 20° |
| Sustained duration | `SUSTAINED_SEC` | 3.0s |
| Alert cooldown | `EVENT_COOLDOWN_SEC` | 10.0s |
| Keypoint confidence | `KP_CONF_THRESH` | 0.3 |

Use `calibrate_head_behavior.py` with live trackbars to find optimal values for a new camera angle before hardcoding.

### COCO Keypoint Indices Used

```
KP_NOSE=0, KP_LEFT_EAR=3, KP_RIGHT_EAR=4, KP_LEFT_SHOULDER=5, KP_RIGHT_SHOULDER=6
```

### Model Files

- `yolo26s-pose.pt` — repo root, pose model for front node (auto-downloaded by Ultralytics if missing)
- `yolo11n.pt` — object detection model (auto-downloaded)
- Hailo `.hef` files — Hailo Model Zoo pre-compiled models, referenced in Hailo test scripts

### Tracking Configuration

`tests/bytetrack_front.yaml` — ByteTrack settings: `track_high_thresh=0.5`, `track_buffer=60`, `match_thresh=0.8`. The `persist=True` argument to `model.track()` keeps tracker state alive across frames so track IDs remain stable after the assignment phase.

### Desk Zones

`tests/Frontcam-set1-001_desk_zones.json` — 20 calibrated (p1, p2) bounding boxes for student desk surfaces, used by the Mid Node for presence-then-absence hand monitoring.

### Evidence Merge (Post-Exam)

After exam completion, evidence from both nodes is merged chronologically with a ±15s deduplication window. The merge tool (not yet in repo) produces a unified incident report.

## Project Documentation

- `PROJECT.md` — master spec: full algorithm formulas, detection logic, dual-node architecture
- `CAMERA_SETUP_GUIDE.md` — USB camera setup and testing on Raspberry Pi
- `POSE_MODEL_SETUP.md` — Hailo pose model compilation and API
- `BACK_NODE_SETUP.md` — Mid Node environment setup (NCNN, venv)
