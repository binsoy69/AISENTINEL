# AISENTINEL Project Implementation Guide

> A Strategic Roadmap for Implementing a Real-Time Automated Exam Proctoring System Using Deep Learning on Edge Devices

---

## 📋 Table of Contents

1. [Project Overview & Objectives](#project-overview--objectives)
2. [Scope and Limitations](#scope-and-limitations)
3. [System Architecture](#system-architecture)
4. [Hardware Configuration](#hardware-configuration)
5. [Software & Algorithm Design](#software--algorithm-design)
6. [Implementation Phases](#implementation-phases)
7. [Testing & Validation Strategy](#testing--validation-strategy)

---

## Project Overview & Objectives

**AISENTINEL** is designed to strengthen the integrity of academic assessments by providing automated, real-time support to human proctors. The project focuses on three key objectives:

1. **Detect Suspicious Behaviors**: Specifically identifying unauthorized actions during examinations such as passing papers, head tilting, hands under the table, using a cellphone, and accessing cheat sheets.
2. **Implement Real-Time Proctoring**: Deploying a dual-node system using two independent Raspberry Pi 5 units — both equipped with a Raspberry Pi AI HAT+ (26 TOPS) for high-speed inference — for continuous, multi-angle monitoring of 20 students. The Front Node covers the front 8 students; the Mid Node covers the back 12 students.
3. **Generate Evidence**: Automatically recording short video clips with timestamps whenever cheating behavior is detected to serve as evidence for administrative review. A post-exam merge process unifies evidence from both nodes into a single, chronological timeline.

---

## Scope and Limitations

### Scope

The study is centered on a visual detection system deployed within a controlled simulation environment.

- **Behaviors Detected**: The system is trained to identify specific visual cues associated with cheating:
  - Abnormal head tilting (both nodes — pose estimation).
  - Hands under the table (both nodes — object-detection-based hand presence-then-absence monitoring per desk zone).
  - Presence and usage of cellphones (both nodes — object detection).
  - Usage of cheat sheets (both nodes — object detection).
  - Passing papers between side-by-side neighbors (both nodes — wrist keypoint lateral exit from tracked student's bounding box via pose estimation + ByteTrack).
- **Environment**: A simulation room designed to approximate a standard Philippine classroom (7m × 9m).
  - **Capacity**: The simulation involves 20 students. The Front Node covers the front 8 students and the Mid Node covers the back 12 students, both running the full detection pipeline (object detection + pose-based behavioral analysis).
  - **Layout**: Seats arranged with at least 1 meter of spacing, following post-pandemic DepEd and DOH guidelines to reduce occlusions and improve camera coverage.
  - **Conditions**: Average exam duration of 1 hour and 30 minutes.
- **Evidence**: Each detected event generates a locally stored video clip with a precise timestamp for review. Evidence from both nodes is merged post-exam into a unified, deduplicated timeline.

### Limitations

The system operates within defined constraints:

- **Visual Only**: It covers behaviors detectable within the cameras' fields of view, particularly around individual student tables.
- **Contextual Blind Spots**:
  - It cannot reliably distinguish cheat sheets that mimic the official questionnaire in size or appearance.
  - It cannot capture subtle, non-object-based behaviors such as hand signals or whispering.
  - Passing papers detection covers only direct side-to-side hand-to-hand transfers between neighbors in the same row. Front-to-back passing, tossing, or sliding papers is not detected.
- **Independent Nodes**: Because the two nodes operate independently, real-time cross-camera fusion is not available. Correlated evidence is linked only post-exam via timestamp alignment.
- **Environment**: Results gathered in the controlled simulation may differ from those in larger, more chaotic real-world classroom settings.

---

## System Architecture

The monitoring setup employs an **Independent Dual-Node Architecture**. Each node operates autonomously during the exam — running its own detection model, behavioral analysis, and evidence recording — with no network dependency between them. This design maximizes fault tolerance: if one node fails, the other continues operating unaffected. Evidence from both nodes is correlated post-exam using synchronized timestamps.

Both nodes run identical hardware (Raspberry Pi 5 + AI HAT+) and the same full detection pipeline (object detection, pose estimation, ByteTrack tracking). They differ only in camera position and student coverage.

### 1. Front Node (Front View — Hailo-Accelerated)

The Front Node is positioned at the front of the room, covering the front 8 students with both object detection and pose-based behavioral analysis.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: Raspberry Pi AI HAT+ (26 TOPS).
- **Models**:
  - **YOLOv11n (Object Detection)** — Runs on Hailo at 25–30 FPS, 640×640 resolution. Detects: `person`, `hand`, `cell_phone`, `cheat_sheet`.
  - **YOLOv11n-pose (Pose Estimation)** — Runs on Hailo or CPU at 7–8 FPS (alternating with detection model). Provides facial and upper-body keypoints (nose, eyes, ears, shoulders, wrists) for head tilt and hands-under-table analysis.
- **Tracking**: ByteTrack for persistent student identification across frames.
- **Export Format**: HEF (Hailo Executable Format).
- **Student Coverage**: Front 8 students (students 1–8).
- **Detection Responsibilities**: Cellphone usage, cheat sheet usage, head tilting, hands under the table (object-detection-based hand presence-then-absence monitoring per desk zone), passing papers (wrist keypoint lateral exit from student's person bounding box toward a side-by-side neighbor).

### 2. Mid Node (Mid-Room View — Hailo-Accelerated)

The Mid Node is positioned in the middle of the room, facing the back half. It provides a front-facing view of the back 12 students with the full detection pipeline — identical to the Front Node.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: Raspberry Pi AI HAT+ (26 TOPS).
- **Models**:
  - **YOLOv11n (Object Detection)** — Runs on Hailo at 25–30 FPS, 640×640 resolution. Detects: `person`, `hand`, `cell_phone`, `cheat_sheet`.
  - **YOLOv11n-pose (Pose Estimation)** — Runs on Hailo or CPU at 7–8 FPS (alternating with detection model). Provides facial and upper-body keypoints (nose, eyes, ears, shoulders, wrists) for head tilt and hands-under-table analysis.
- **Tracking**: ByteTrack for persistent student identification across frames.
- **Export Format**: HEF (Hailo Executable Format).
- **Student Coverage**: Back 12 students (students 9–20).
- **Detection Responsibilities**: Cellphone usage, cheat sheet usage, head tilting, hands under the table (object-detection-based hand presence-then-absence monitoring per desk zone), passing papers (wrist keypoint lateral exit from student's person bounding box toward a side-by-side neighbor).

### 3. Post-Exam Evidence Merge

After the exam concludes, evidence from both nodes is collected (via microSD or local network transfer) and processed through a merge script that:

- Combines all flagged events into a single chronological timeline.
- Deduplicates events that were captured by both cameras within a configurable time window (e.g., 15 seconds).
- Outputs a unified evidence report with linked video clips from each perspective.

---

## Hardware Configuration

### Positioning Strategy

To maximize coverage in the 7m × 9m room:

- **Front Camera**:
  - **Location**: Mounted at the front of the room.
  - **Height**: 2.5 meters.
  - **Angle**: Angled downwards at 15 degrees.
  - **Purpose**: Covers the front 8 students (students 1–8). Runs the full detection pipeline: object detection (cellphones, cheat sheets, hands), pose-based behavioral analysis (head tilt, hands under table, passing papers).

- **Mid Camera**:
  - **Location**: Mounted in the middle of the room (ceiling or elevated mount), facing the back half.
  - **Height**: 2.5 meters.
  - **Angle**: Angled downwards to cover the back 12 student desks.
  - **Purpose**: Covers the back 12 students (students 9–20). Runs the same full detection pipeline as the Front Node: object detection, pose-based behavioral analysis, and passing papers detection.

### Component List

- **Compute Modules**: 2× Raspberry Pi 5 (8GB RAM).
- **AI Accelerators**: 2× Raspberry Pi AI HAT+ (26 TOPS) — one for each node.
- **Imaging**: 2× High-resolution Wide-Angle Cameras (USB or MIPI-CSI) capable of clear video at 2.5m distance.
- **Storage**: High-endurance microSD cards (64GB+) for local video clip storage on each node.
- **Power**: Official USB-C Power Supplies (27W) for both units to support sustained compute loads.
- **Cooling**: Active coolers (heatsink + fan) for **both** Pis to handle the AI HAT+ workload over the 1.5-hour exam duration.

---

## Software & Algorithm Design

### Detection Pipeline Architecture

Both nodes follow the same modular pipeline structure and run identical software — differing only in `config.yaml` settings (student coverage, camera calibration, desk zone definitions):

```
aisentinel/
├── config.yaml                 # Thresholds, timers, camera settings, zone definitions (per-node)
├── main.py                     # Orchestrator (selects front or mid mode)
├── capture/
│   ├── camera.py               # Frame capture from camera feed
│   └── buffer.py               # Circular video ring buffer (last N seconds)
├── inference/
│   ├── detector.py             # YOLOv11 object detection (Hailo)
│   └── pose.py                 # YOLOv11-pose for head tilt and keypoint analysis
├── tracking/
│   └── tracker.py              # ByteTrack wrapper for persistent student IDs
├── analysis/
│   ├── behavior.py             # Rule engine (tracker + pose-based, shared by both nodes)
│   └── zones.py                # Desk zone boundary definitions per seat
├── evidence/
│   └── recorder.py             # Save clips + metadata on trigger
├── merge/
│   └── merge_evidence.py       # Post-exam evidence merging and deduplication
└── models/
    ├── detect.hef              # Detection model (Hailo format, shared)
    └── pose.hef                # Pose model (Hailo format, shared)
```

---

### Shared Models & Detection Logic (Both Nodes)

Both nodes use the same models and detection logic. The sections below describe the shared pipeline. The only per-node differences are the student coverage (Front: students 1–8, Mid: students 9–20) and desk zone calibration.

### Front Node: Models & Detection Logic

#### Model 1 — YOLOv11n Object Detection (Hailo)

- **Resolution**: 640×640
- **Target FPS**: 25–30
- **Export Format**: HEF (Hailo Executable Format)
- **Classes**:

| Class Index | Class Name    | Description                                      |
|-------------|---------------|--------------------------------------------------|
| 0           | `person`      | Student (pretrained from COCO, fine-tuned). Used for ByteTrack person tracking. |
| 1           | `hand`        | Visible hands on or near the desk; also used to detect reaching gestures |
| 2           | `cell_phone`  | Unauthorized device                              |
| 3           | `cheat_sheet` | Unauthorized reference material                  |

#### Model 2 — YOLOv11n-pose Pose Estimation (Hailo or CPU)

- **Resolution**: 640×640
- **Target FPS**: 7–8 (runs on every 4th frame, alternating with detection model)
- **Purpose**: Provides keypoints (nose, eyes, ears, shoulders, **wrists**) for head tilt and hands-under-table analysis.
- **No custom training required** — uses pretrained YOLOv11n-pose weights.
- **Desk Zone Boundary**: A horizontal line is defined per student seat representing the desk edge. When a tracked student's wrist keypoints drop below this boundary for a sustained period, a "hands under table" flag is raised.

#### Tracking: ByteTrack

ByteTrack assigns persistent IDs to each detected person across frames, enabling per-student behavioral state tracking.

**Key ByteTrack parameters to tune**:
- `track_high_thresh`: 0.5 (confidence threshold for first-pass matching)
- `track_low_thresh`: 0.1 (for second-pass matching of low-confidence detections)
- `match_thresh`: 0.8 (IoU threshold for matching)
- `track_buffer`: 60+ frames (keep lost tracks alive longer since students are mostly stationary)

#### Front Node Behavioral Logic

Each tracked student maintains an independent state machine:

| Behavior                   | Detection Method                                                                                                  | Trigger Condition                          |
|----------------------------|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| **Cellphone usage**        | Direct detection of `cell_phone` at high confidence                                                               | Confidence ≥ 0.6, immediate flag           |
| **Cheat sheet usage**      | Direct detection of `cheat_sheet` at high confidence                                                              | Confidence ≥ 0.5, immediate flag           |
| **Head tilting**           | Ear-to-ear angle from pose keypoints: `atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)`                | Angle > 25–30° sustained for 3–5 seconds   |
| **Hands under table**      | Object-detection-based: Each tracked student (`student` class via ByteTrack) is assigned to a calibrated desk polygon ROI via bounding-box/polygon intersection area. `hand` detections are associated to the nearest student (center inside student bbox or within configurable pixel margin). A desk is flagged when the assigned student's hands are absent from the desk ROI. A sliding-window majority vote (e.g., 12 frames, ≥ 60% missing) smooths intermittent missed detections before the sustained timer begins. Student absence > 2 seconds resets the desk state to prevent false alerts. | Majority-vote-smoothed absence sustained for ≥ 3 seconds (configurable), with 10-second cooldown between repeated flags |
| **Passing papers**         | A tracked student's wrist keypoint exits their own `person` bounding box laterally (left or right edge). The exit direction determines which side-by-side neighbor is involved. The nearest tracked neighbor in that lateral direction (matched by similar y-center within a row tolerance) is identified. Both students are flagged as a linked event. Confidence is boosted if a `hand` or `cheat_sheet` detection overlaps the wrist's exit region. | Immediate flag: base confidence 0.7, reinforced confidence 0.85 if `hand`/`cheat_sheet` overlaps exit region. |

---

### Mid Node: Models & Detection Logic

The Mid Node runs the **same models, tracking, and behavioral logic** as the Front Node. It differs only in camera position (mid-room, facing back) and student coverage (back 12 students, students 9–20).

#### Models

- **YOLOv11n (Object Detection)** — Runs on Hailo at 25–30 FPS, 640×640 resolution. Same model and classes as the Front Node.
- **YOLOv11n-pose (Pose Estimation)** — Runs on Hailo or CPU at 7–8 FPS. Same pretrained weights as the Front Node.

#### Tracking & Behavioral Logic

- **ByteTrack** for persistent student identification across frames.
- **Behavioral state machine** identical to the Front Node (see table above).
- **Desk zone calibration** is performed independently for the mid camera's perspective, covering the back 12 desk positions.

#### Mid Node Behavioral Logic

| Behavior                      | Detection Method                                                                                        | Trigger Condition                              |
|-------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **Cellphone usage**           | Direct detection of `cell_phone` at high confidence                                                     | Confidence ≥ 0.6, immediate flag               |
| **Cheat sheet usage**         | Direct detection of `cheat_sheet` at high confidence                                                    | Confidence ≥ 0.5, immediate flag               |
| **Head tilting**              | Ear-to-ear angle from pose keypoints: `atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)`      | Angle > 25–30° sustained for 3–5 seconds       |
| **Hands under table**         | Object-detection-based: same sliding-window majority vote logic as Front Node                           | Majority-vote-smoothed absence sustained for ≥ 3 seconds, with 10-second cooldown |
| **Passing papers**            | Wrist keypoint lateral exit from student's person bounding box toward a side-by-side neighbor            | Base confidence 0.7, reinforced 0.85 if overlap |

---

### Evidence Generation (Both Nodes)

Both nodes implement identical evidence recording logic:

- **Circular Buffer**: Each node maintains a rolling video buffer of the last 10–15 seconds of footage at full camera frame rate (30 FPS), independent of the inference frame rate.
- **Trigger**: When a behavioral flag is raised.
- **Action**: The system saves the buffered footage (10 seconds before the event) plus records an additional 10 seconds after the trigger, producing a ~20-second evidence clip.
- **Metadata**: Each clip's filename encodes: `{node}_{timestamp}_{behavior_type}_{confidence}.mp4`
  - Example: `front_20260207_143022_cellphone_087.mp4`
  - Example: `mid_20260207_143055_hands_under_table_072.mp4`
  - Example: `front_20260207_143110_passing_papers_085.mp4`
- **Storage**: Clips are stored locally on each node's microSD card in an `evidence/` directory.

---

### Post-Exam Evidence Merge

After the exam, evidence from both nodes is collected and processed:

1. **Collection**: Transfer evidence directories from both microSD cards to a review workstation (via USB reader, local network, or direct copy).
2. **Merging**: The `merge_evidence.py` script:
   - Loads all event metadata from both nodes.
   - Sorts events chronologically.
   - Deduplicates events that occur within a configurable time window (default: 15 seconds) and share the same behavior category.
   - Outputs a unified evidence report (CSV or HTML) linking to the relevant video clips from each perspective.
3. **Review**: The proctor or administrator reviews the merged timeline, with access to video evidence from one or both camera angles per incident.

---

### Clock Synchronization

Accurate timestamp alignment between nodes is **critical** for the post-exam merge process.

- **Method**: Both Raspberry Pi units synchronize their clocks via NTP before each exam session.
- **Setup**: If the exam room has internet access, both Pis sync to the same public NTP server. If offline, designate one Pi as the NTP server and have the other sync to it over a direct Ethernet connection.
- **Verification**: A pre-exam check script validates that the time difference between both nodes is less than 1 second.

```bash
# Ensure NTP sync is active on both Pis
sudo timedatectl set-ntp true

# Verify sync status
timedatectl status
```

---

## Implementation Phases

### Phase 1: Environment & Hardware Setup

1. **Room Configuration**: Set up the 7m × 9m simulation room. Arrange 20 chairs with precise 1-meter spacing.
2. **Mounting**: Install the Front Camera mount at 2.5m on the front wall (15-degree downward tilt covering the front 8 desks). Install the Mid Camera mount at 2.5m in the middle of the room (ceiling or elevated mount, angled to cover the back 12 desks).
3. **Device Assembly**:
   - Attach the AI HAT+ to the Front Raspberry Pi 5. Install active cooling.
   - Attach the AI HAT+ to the Mid Raspberry Pi 5. Install active cooling.
4. **Clock Sync**: Configure NTP synchronization on both Pis and verify alignment.

### Phase 2: Data Collection & Model Preparation

1. **Dataset Gathering**: Record footage from **both camera positions** (front and mid) of actors performing the specific cheating behaviors in the simulation room.
2. **Annotation**:
   - Label images for all 4 classes (`person`, `hand`, `cell_phone`, `cheat_sheet`) using Roboflow or CVAT from both camera perspectives.
   - Target: 1,500–2,000 annotated images per class.
3. **Training** (on GPU machine, not on the Pi):
   - **Detection model**: Fine-tune YOLOv11n from COCO-pretrained weights with 4 classes (`person`, `hand`, `cell_phone`, `cheat_sheet`). The same model is deployed on both nodes.
   - **Pose model**: No training needed — use pretrained YOLOv11n-pose weights directly.
4. **Optimization & Export**:
   - Export both models to HEF format using the Hailo Dataflow Compiler for AI HAT+ deployment on both nodes.

### Phase 3: System Integration

1. **Front Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install Hailo runtime (`hailo-all`).
   - Deploy the HEF detection and pose models.
   - Implement the ByteTrack-based tracking and behavioral analysis pipeline.
   - Implement evidence recording with circular buffer.

2. **Mid Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install Hailo runtime (`hailo-all`).
   - Deploy the HEF detection and pose models (same as Front Node).
   - Implement the ByteTrack-based tracking and behavioral analysis pipeline (same as Front Node).
   - Run the desk zone calibration tool to define desk edge boundaries for each of the back 12 seats.
   - Implement evidence recording with circular buffer.

3. **Evidence Merge Tool**:
   - Develop and test the `merge_evidence.py` script on the review workstation.

### Phase 4: Testing & Simulation

1. **Thermal Stability Test**: Run both nodes continuously for 1.5 hours with active inference to verify no CPU/NPU throttling occurs. Monitor temperatures:
   ```bash
   vcgencmd measure_temp  # Check during operation; target < 80°C
   ```
2. **Zone Calibration Verification**: Confirm that the Mid Node's desk zone boundaries accurately map to each of the back 12 seats from the mid camera's perspective.
3. **Pilot Run**: Conduct a mock exam with 20 participants.
4. **Scenario Testing**: Instruct participants to act out specific scenarios:
   - *Scenario A*: Student in the front 8 tilts head to look at a neighbor's paper (Front Node — pose estimation).
   - *Scenario B*: Student in the back 12 tilts head to look at a neighbor's paper (Mid Node — pose estimation).
   - *Scenario C*: Student hides hands under the table for extended period (both nodes — object-detection-based hand absence from desk zone).
   - *Scenario D*: Student uses a cellphone on or near desk (both nodes — object detection).
   - *Scenario E*: Student uses a cheat sheet on or near desk (both nodes — object detection).
   - *Scenario F*: Student directly hands a paper/note to the neighbor sitting beside them in the same row (both nodes — wrist keypoint exits person bounding box laterally toward neighbor; both students flagged as a linked event).
5. **Evidence Merge Test**: Run the merge script on the collected evidence. Verify that:
   - Events from both nodes appear in the correct chronological order.
   - Duplicate detections of the same event are properly deduplicated.
   - All video clips are playable and correctly linked.
6. **Threshold Tuning**: Adjust detection confidence thresholds, sustained-duration timers, and zone boundaries based on pilot results.

---

## Testing & Validation Strategy

The system will be evaluated based on:

1. **Detection Accuracy**: The percentage of true positive cheating events captured versus missed events (False Negatives), measured **per node** and **combined**.
2. **False Alarm Rate**: The frequency of normal behaviors (e.g., stretching, dropping a pen, briefly looking around) being flagged as cheating, measured per node.
3. **System Latency**: Time taken between the behavior occurring and the system flagging it, measured per node. Expected: both nodes < 1 second (Hailo-accelerated inference).
4. **Evidence Integrity**: Verifying that recorded clips are playable, correctly timestamped, and clearly show the flagged behavior from the appropriate camera angle.
5. **Evidence Merge Accuracy**: Verifying that the post-exam merge correctly combines and deduplicates events from both nodes without losing genuine separate incidents.
6. **Thermal Stability**: Ensuring both Raspberry Pi 5 units can run their respective inference workloads for the full 1.5-hour exam duration without throttling.
   - **Front Node**: Pi 5 + AI HAT+ running dual models (detection + pose).
   - **Mid Node**: Pi 5 + AI HAT+ running dual models (detection + pose).
7. **Coverage Completeness**: Evaluating whether the dual-camera setup eliminates significant blind spots. The Front Node covers the front 8 students and the Mid Node covers the back 12 students, both running the full detection pipeline (object detection + pose-based behavioral analysis). The mid-room camera provides close-range front-facing coverage of the back students, improving detection reliability for students farther from the front camera.
