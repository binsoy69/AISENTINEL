# AISENTINEL Project Implementation Guide

This document reflects the current repository state after the front-node Pi runtime was moved into a self-contained runtime package.

## Current Decision

- `runtime/front_node_pi` is now the executable source of truth for the front-node Raspberry Pi runtime.
- The runtime no longer imports behavior logic from `tests/tests_on_pi`.
- `tests/tests_on_pi` remains the reference/prototype lineage that the runtime was ported from.
- `tests/tests_on_pc` remains a PC-side experiment and calibration area.
- The front-node Pi runtime is the current packaged implementation baseline.
- Mid-node packaging and evidence merging are still planned work, not completed runtime modules in this repository.

## Current Repository Status

### Implemented now

- A self-contained front-node Pi runtime under `runtime/front_node_pi`
- Local runtime copies of the Pi behavior modules for:
  - head tilt and shoulder turn
  - passing papers
  - hands under table
  - phone and cheat-sheet detection
  - combined all-behavior orchestration
  - setup profile save/load support
- Config-driven model selection and behavior-threshold tuning
- Supporting model artifacts under `models/`
- Supporting setup guides for camera, pose model, microphone, and Hailo deployment

### Not yet implemented as a finalized runtime

- A packaged `runtime/mid_node_pi` equivalent
- A repo-backed evidence merge utility
- A packaged mid-node runtime equivalent to the front-node split video/webcam runtime

## Canonical Front-Node Code Locations

The front-node runtime is now fully contained in:

- `runtime/front_node_pi/main.py`
- `runtime/front_node_pi/save_setup.py`
- `runtime/front_node_pi/main_video.py`
- `runtime/front_node_pi/main_webcam.py`
- `runtime/front_node_pi/calibrate_video.py`
- `runtime/front_node_pi/calibrate_webcam.py`
- `runtime/front_node_pi/config_video.ini`
- `runtime/front_node_pi/config_webcam.ini`
- `runtime/front_node_pi/runtime_config.py`
- `runtime/front_node_pi/runtime_support.py`
- `runtime/front_node_pi/front_node_all_behavior_pi.py`
- `runtime/front_node_pi/front_node_all_behavior_setup_io.py`
- `runtime/front_node_pi/front_node_head_behavior_pi.py`
- `runtime/front_node_pi/front_node_passing_papers_pi.py`
- `runtime/front_node_pi/front_node_hands_under_table_pi.py`
- `runtime/front_node_pi/front_node_cellphone_cheat_pi.py`

These files now hold the runnable front-node logic without importing the test scripts.

## Runtime Structure

```text
runtime/
  front_node_pi/
    config.ini
    config_video.ini
    config_webcam.ini
    main.py
    main_video.py
    main_webcam.py
    save_setup.py
    calibrate_video.py
    calibrate_webcam.py
    runtime_config.py
    runtime_support.py
    front_node_all_behavior_pi.py
    front_node_all_behavior_setup_io.py
    front_node_head_behavior_pi.py
    front_node_passing_papers_pi.py
    front_node_hands_under_table_pi.py
    front_node_cellphone_cheat_pi.py
    data/
      evidence_combined/
      setup_profiles/
```

## Front-Node Runtime Architecture

### Entry points

- `runtime/front_node_pi/main_video.py`
  - front-node entrypoint for video-file processing
- `runtime/front_node_pi/main_webcam.py`
  - front-node entrypoint for live webcam processing
- `runtime/front_node_pi/calibrate_video.py`
  - calibration/setup saver for video mode
- `runtime/front_node_pi/calibrate_webcam.py`
  - calibration/setup saver for webcam mode
- `runtime/front_node_pi/main.py`
  - compatibility wrapper that forwards to `main_video.py`
- `runtime/front_node_pi/save_setup.py`
  - compatibility wrapper that forwards to `calibrate_video.py`

### Local runtime modules

- `front_node_all_behavior_pi.py`
  - combined orchestrator
  - shared pose estimator
  - tracker reacquisition logic
  - evidence burst generation
  - MJPEG stream handling
- `front_node_head_behavior_pi.py`
  - head tilt and shoulder-turn analysis
- `front_node_passing_papers_pi.py`
  - row-neighbor logic and pair interaction tracking
- `front_node_hands_under_table_pi.py`
  - hand association and per-student table-edge monitoring
- `front_node_cellphone_cheat_pi.py`
  - phone/cheat-sheet detection and student-object association
- `front_node_all_behavior_setup_io.py`
  - saved setup profile serialization and first-frame remapping

## Final Front-Node Program Flow

The packaged runtime preserves the same operational sequence as the original Pi flow, but now entirely inside `runtime/front_node_pi` with separate video and webcam entrypoints.

1. Load either `runtime/front_node_pi/config_video.ini` or `runtime/front_node_pi/config_webcam.ini`
2. Resolve models, thresholds, tracker settings, and output directories
3. Apply config values to the local behavior modules
4. Open the source
   - video mode: CLI `--video`, config `default_video`, or file dialog
   - webcam mode: configured webcam source from `config_webcam.ini`
5. Create one shared Hailo VDevice for all models
6. Load:
   - pose model
   - hand model
   - object model
7. Read the first frame
8. Resolve setup state in this order:
   - explicit `--calibration-file`
   - source-specific default setup profile from the active config
   - auto-matched saved profile for the active source
   - manual setup flow
9. If manual setup is needed, run:
   - ROI polygon setup
   - student assignment
   - per-student table-edge line drawing
10. Start the web stream
11. Run unified detection:
   - head tilt
   - shoulder turn
   - passing papers
   - hands under table
   - phone
   - cheat sheet
12. Save evidence under `runtime/front_node_pi/data/evidence_combined/`

## Runtime Configuration

The front-node runtime now uses separate config files:

- `runtime/front_node_pi/config_video.ini`
- `runtime/front_node_pi/config_webcam.ini`

Each file controls model selection and per-behavior thresholds for its source mode.

### Config sections

- `[models]`
  - pose
  - hand
  - object
- `[inference]`
  - pose_confidence
  - object_confidence
- `[runtime]`
  - port
- `[video_source]`
  - default_video
  - default_setup_profile
  - auto_use_saved_setup
- `[webcam_source]`
  - camera_index
  - camera_name
  - capture_width
  - capture_height
  - capture_fps
  - warmup_frames
  - default_setup_profile
  - auto_use_saved_setup
- `[outputs]`
  - evidence_root
  - setup_profile_dir
- `[tracking]`
  - iou_threshold
  - max_lost
- `[head_behavior]`
  - head_tilt_angle_deg
  - head_turn_ratio
  - shoulder_turn_angle_deg
  - sustained_sec
  - event_cooldown_sec
  - keypoint_confidence
- `[passing_papers]`
  - event_cooldown_sec
  - keypoint_confidence
  - row_tolerance_px
  - reference_bbox_height
  - wrist_proximity_px
  - min_interaction_sec
- `[hands_under_table]`
  - hand_confidence
  - person_confidence
  - hands_missing_sustain_sec
  - event_cooldown_sec
  - min_visible_hands
  - hand_assoc_margin_px
  - smooth_window_frames
  - smooth_missing_ratio
  - student_absent_reset_sec
  - table_edge_near_px
  - edge_disappear_arm_sec
- `[object_detection]`
  - person_confidence
  - phone_confidence
  - cheat_sheet_confidence
  - event_cooldown_sec
  - assoc_iou_thresh
- `[evidence]`
  - pre_event_frames
  - post_event_frames

### Config path behavior

- Paths inside both config files are resolved relative to the repository root.
- You can override the config file with:
  - `python runtime/front_node_pi/main_video.py --config <path>`
  - `python runtime/front_node_pi/main_webcam.py --config <path>`
- You can also point to another config with:
  - `AISENTINEL_FRONT_NODE_CONFIG`

## Behavior Coverage in the Current Front-Node Runtime

| Behavior | Runtime module | Status |
|---|---|---|
| Head tilt | `front_node_head_behavior_pi.py` | Implemented |
| Shoulder turn | `front_node_head_behavior_pi.py` | Implemented |
| Passing papers | `front_node_passing_papers_pi.py` | Implemented |
| Hands under table | `front_node_hands_under_table_pi.py` | Implemented |
| Phone | `front_node_cellphone_cheat_pi.py` | Implemented |
| Cheat sheet | `front_node_cellphone_cheat_pi.py` | Implemented |

## Actual Project Structure

```text
AISENTINEL/
  models/
  runtime/
    front_node_pi/
  tests/
    tests_on_pi/      # reference/prototype Pi scripts
    tests_on_pc/      # PC experiments and calibration tools
  CAMERA_SETUP_GUIDE.md
  MID_NODE_SETUP.md
  POSE_MODEL_SETUP.md
  PROJECT.md
```

## Recommended Commands

### Run the front-node video runtime

```bash
python runtime/front_node_pi/main_video.py
```

### Run the front-node webcam runtime

```bash
python runtime/front_node_pi/main_webcam.py
```

### Save a reusable video setup profile

```bash
python runtime/front_node_pi/calibrate_video.py
```

### Save a reusable webcam setup profile

```bash
python runtime/front_node_pi/calibrate_webcam.py
```

## Deployment Notes

### Front node

- Target device: Raspberry Pi 5
- Accelerator: Raspberry Pi AI HAT+ using Hailo runtime
- Runtime assumptions:
  - `hailo-all` installed
  - Python dependencies installed
  - Flask available for MJPEG streaming
  - model files present in `models/`

### Current operating mode

- The runtime now supports both video mode and webcam mode through separate entrypoints.
- Video mode remains file-dialog capable.
- Webcam mode uses a dedicated webcam config and calibration flow.
- Evidence and setup profiles now live under the runtime folder instead of under `tests/`.

## Limitations That Still Apply

- The current packaged runtime is front-node focused.
- The dual-node architecture remains a project goal, not a fully packaged repo implementation.
- Real-time cross-node correlation is not implemented.
- Evidence merging is still planned work.
- Manual first-frame setup is still required unless a saved profile exists.
- Webcam mode still depends on the same manual first-frame calibration flow unless a saved profile exists.

## Next Implementation Priorities

1. Build `runtime/mid_node_pi` with the same self-contained structure.
2. Reduce duplication across the split video/webcam entrypoints where it no longer adds clarity.
3. Add an evidence merge utility to the repository.
4. Add smoke-test and validation scripts for Pi deployment.
5. Reduce duplication between front-node modules after the runtime stabilizes.

## Summary

AISENTINEL now has a self-contained front-node runtime under `runtime/front_node_pi`.

- The runtime no longer depends on `tests/tests_on_pi`.
- The runtime now exposes separate video and webcam entrypoints plus separate calibration tools.
- Model paths and per-behavior thresholds are configurable through `config_video.ini` and `config_webcam.ini`.
- The packaged runtime preserves the original all-behavior flow while keeping deployment files, evidence, and setup profiles inside the runtime folder.
