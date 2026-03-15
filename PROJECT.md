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
2. **Implement Real-Time Proctoring**: Deploying a dual-node system using two independent Raspberry Pi 5 units — one with a Raspberry Pi AI HAT+ (26 TOPS) for high-speed inference (Front Node) and one running a lightweight CPU-optimized object detection model (Mid Node) — for continuous, multi-angle monitoring of 20 students.
3. **Generate Evidence**: Automatically recording short video clips with timestamps whenever cheating behavior is detected to serve as evidence for administrative review. A post-exam merge process unifies evidence from both nodes into a single, chronological timeline.

---

## Scope and Limitations

### Scope

The study is centered on a visual detection system deployed within a controlled simulation environment.

- **Behaviors Detected**: The system is trained to identify specific visual cues associated with cheating:
  - Abnormal head tilting (Front Node — pose estimation).
  - Hands under the table (Front Node — wrist keypoint disappearance from pose estimation; Mid Node — hand presence-then-absence monitoring per desk zone).
  - Presence and usage of cellphones (Front Node — object detection on all 20 students; Mid Node — object detection on the back 12 students).
  - Usage of cheat sheets (Front Node — object detection on all 20 students; Mid Node — object detection on the back 12 students).
  - Passing papers between side-by-side neighbors (Front Node — wrist keypoint lateral exit from tracked student's bounding box via pose estimation + ByteTrack).
- **Environment**: A simulation room designed to approximate a standard Philippine classroom (7m × 9m).
  - **Capacity**: The simulation involves 20 students. The Front Node covers all 20 students (object detection + pose-based behavioral analysis), while the Mid Node covers the back 12 students (object detection only for cellphones, cheat sheets, and hands under table).
  - **Layout**: Seats arranged with at least 1 meter of spacing, following post-pandemic DepEd and DOH guidelines to reduce occlusions and improve camera coverage.
  - **Conditions**: Average exam duration of 1 hour and 30 minutes.
- **Evidence**: Each detected event generates a locally stored video clip with a precise timestamp for review. Evidence from both nodes is merged post-exam into a unified, deduplicated timeline.

### Limitations

The system operates within defined constraints:

- **Visual Only**: It covers behaviors detectable within the cameras' fields of view, particularly around individual student tables.
- **Contextual Blind Spots**:
  - It cannot reliably distinguish cheat sheets that mimic the official questionnaire in size or appearance.
  - It cannot capture subtle, non-object-based behaviors such as hand signals or whispering.
  - Passing papers detection covers only direct side-to-side hand-to-hand transfers between neighbors in the same row (Front Node only). Front-to-back passing, tossing, or sliding papers is not detected.
- **Frame Rate Disparity**: The Mid Node operates at a lower frame rate (8–12 FPS) than the Front Node (25–30 FPS) due to its CPU-only inference, which means very fast, transient movements may be missed by the mid camera.
- **Independent Nodes**: Because the two nodes operate independently, real-time cross-camera fusion is not available. Correlated evidence is linked only post-exam via timestamp alignment.
- **Environment**: Results gathered in the controlled simulation may differ from those in larger, more chaotic real-world classroom settings.

---

## System Architecture

The monitoring setup employs an **Independent Dual-Node Architecture**. Each node operates autonomously during the exam — running its own detection model, behavioral analysis, and evidence recording — with no network dependency between them. This design maximizes fault tolerance: if one node fails, the other continues operating unaffected. Evidence from both nodes is correlated post-exam using synchronized timestamps.

### 1. Front Node (Front View — Hailo-Accelerated)

The Front Node handles the computationally intensive workload, leveraging the AI HAT+ for high-speed inference. It covers all 20 students with both object detection and pose-based behavioral analysis.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: Raspberry Pi AI HAT+ (26 TOPS).
- **Models**:
  - **YOLOv11n (Object Detection)** — Runs on Hailo at 25–30 FPS, 640×640 resolution. Detects: `person`, `hand`, `cell_phone`, `cheat_sheet`.
  - **YOLOv11n-pose (Pose Estimation)** — Runs on Hailo or CPU at 7–8 FPS (alternating with detection model). Provides facial and upper-body keypoints (nose, eyes, ears, shoulders, wrists) for head tilt, head turn, and hands-under-table analysis.
- **Tracking**: ByteTrack for persistent student identification across frames.
- **Export Format**: HEF (Hailo Executable Format).
- **Detection Responsibilities**: Cellphone usage, cheat sheet usage, head tilting, hands under the table (wrist keypoints from pose estimation disappear or drop below desk zone boundary), passing papers (wrist keypoint lateral exit from student's person bounding box toward a side-by-side neighbor).

### 2. Mid Node (Mid-Room View — CPU-Only)

The Mid Node is positioned in the middle of the room, facing the back half. It provides a front-facing view of the back 12 students with a streamlined object-detection-only pipeline optimized for CPU inference. No pose estimation runs on this node.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: None (CPU-only inference).
- **Model**:
  - **YOLOv11n (Object Detection, Narrowed)** — Runs on CPU via NCNN at 8–12 FPS, 320×320 resolution. Detects a reduced class set: `hand`, `cell_phone`, `cheat_sheet`.
- **Tracking**: None. Uses **per-seat desk zone monitoring** — a desk surface region is defined per student seat during calibration. Hands-under-table is inferred when `hand` detections that were previously present in a zone stop appearing (presence-then-absence).
- **Export Format**: NCNN.
- **Detection Responsibilities**: Hands under table (hand absence monitoring per desk zone), cellphone usage, cheat sheet usage. Covers the back 12 of 20 students only.

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
  - **Purpose**: Covers all 20 students. Handles object detection (cellphones, cheat sheets, hands) and pose-based behavioral analysis (head tilt, hands under table via wrist keypoints below desk zone).

- **Mid Camera**:
  - **Location**: Mounted in the middle of the room (ceiling or elevated mount), facing the back half.
  - **Height**: 2.5 meters.
  - **Angle**: Angled downwards to cover the back 12 student desks.
  - **Purpose**: Provides a front-facing view of the back 12 students for object detection of cellphones, cheat sheets, and hands below the desk zone. Complements the Front Camera by detecting small objects that are difficult to resolve from the front wall distance.

### Component List

- **Compute Modules**: 2× Raspberry Pi 5 (8GB RAM).
- **AI Accelerator**: 1× Raspberry Pi AI HAT+ (26 TOPS) for the Front Node only.
- **Imaging**: 2× High-resolution Wide-Angle Cameras (USB or MIPI-CSI) capable of clear video at 2.5m distance.
- **Storage**: High-endurance microSD cards (64GB+) for local video clip storage on each node.
- **Power**: Official USB-C Power Supplies (27W) for both units to support sustained compute loads.
- **Cooling**: Active coolers (heatsink + fan) for **both** Pis. The Front Node requires cooling for the AI HAT+ workload; the Mid Node requires cooling for sustained CPU-based inference, which generates significant heat over the 1.5-hour exam duration.

---

## Software & Algorithm Design

### Detection Pipeline Architecture

Both nodes follow a modular pipeline structure, but with different components active on each:

```
aisentinel/
├── config.yaml                 # Thresholds, timers, camera settings, zone definitions
├── main.py                     # Orchestrator (selects front or mid mode)
├── capture/
│   ├── camera.py               # Frame capture from camera feed
│   └── buffer.py               # Circular video ring buffer (last N seconds)
├── inference/
│   ├── detector.py             # YOLOv11 object detection (Hailo or NCNN)
│   └── pose.py                 # YOLOv11-pose for head tilt (Front Node only)
├── tracking/
│   └── tracker.py              # ByteTrack wrapper (Front Node only)
├── analysis/
│   ├── behavior_front.py       # Front Node rule engine (tracker + pose-based)
│   ├── behavior_mid.py         # Mid Node rule engine (zone-based object detection)
│   └── zones.py                # Desk zone boundary definitions per seat (Mid Node)
├── evidence/
│   └── recorder.py             # Save clips + metadata on trigger
├── merge/
│   └── merge_evidence.py       # Post-exam evidence merging and deduplication
└── models/
    ├── front_detect.hef        # Front detection model (Hailo format)
    ├── front_pose.hef          # Front pose model (Hailo format)
    └── mid_detect_ncnn/        # Mid detection model (NCNN format)
```

---

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
| **Hands under table**      | Wrist keypoints (left/right wrist) from pose drop below the per-seat desk zone line, or become undetected (low confidence) after being consistently tracked above the desk | Sustained for 5–8 seconds |
| **Passing papers**         | A tracked student's wrist keypoint exits their own `person` bounding box laterally (left or right edge). The exit direction determines which side-by-side neighbor is involved. The nearest tracked neighbor in that lateral direction (matched by similar y-center within a row tolerance) is identified. Both students are flagged as a linked event. Confidence is boosted if a `hand` or `cheat_sheet` detection overlaps the wrist's exit region. | Immediate flag: base confidence 0.7, reinforced confidence 0.85 if `hand`/`cheat_sheet` overlaps exit region. Front Node only — Mid Node does not attempt this detection. |

---

### Mid Node: Model & Detection Logic

#### Model — YOLOv11n Object Detection (NCNN, Narrowed)

- **Resolution**: 320×320
- **Target FPS**: 8–12
- **Export Format**: NCNN (optimized for ARM CPU)
- **Coverage**: Back 12 of 20 students (camera positioned in the middle of the room, facing toward the back wall)
- **Classes** (reduced set for faster inference):

| Class Index | Class Name    | Description                                          |
|-------------|---------------|------------------------------------------------------|
| 0           | `hand`        | Visible hands on the desk surface; absence from desk zone infers hands under table |
| 1           | `cell_phone`  | Unauthorized device visible on or near desk          |
| 2           | `cheat_sheet` | Unauthorized reference material on or near desk      |

#### Per-Seat Desk Zone Monitoring (No Tracker)

Instead of tracking individual students, the Mid Node uses **per-seat desk zone regions** mapped during calibration. Since the camera is fixed, these zones remain stable throughout the session. No pose estimation or tracker runs on this node.

**Hands-under-table logic (presence-then-absence)**:

Since hands under the desk are occluded and physically undetectable, the Mid Node infers this behavior by monitoring whether `hand` detections *stop appearing* in each student's desk zone after having been present:

```
Per desk zone, each frame:
  if hand detected in zone → reset absence_timer, set hand_seen = True
  if no hand in zone AND hand_seen is True → increment absence_timer
  if absence_timer ≥ threshold (5–8 sec) → flag "hands under table"
```

The `hand_seen` flag prevents false positives at exam start before hands have ever been visible. Normal exam behavior (student writing) produces consistent `hand` detections in the zone; hands moving below the desk surface cause those detections to vanish.

**Object presence**:
- `cell_phone` and `cheat_sheet` detections anywhere in frame are flagged immediately — no zone logic required.

**Calibration process**: Before the exam, a calibration script captures a reference frame of the empty room. The operator draws a bounding region per seat representing the desk surface area using a simple GUI tool. These zones are saved to `config.yaml`.

#### Mid Node Behavioral Logic

| Behavior                      | Detection Method                                                                                        | Trigger Condition                              |
|-------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **Hands under table**         | `hand` previously detected in desk zone, then absent — `absence_timer` incremented each frame with no detection | Absence sustained for 5–8 seconds             |
| **Cellphone usage**           | `cell_phone` detected anywhere in frame                                                                 | Immediate flag at confidence ≥ 0.6             |
| **Cheat sheet usage**         | `cheat_sheet` detected anywhere in frame                                                                | Immediate flag at confidence ≥ 0.5             |

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
2. **Mounting**: Install the Front Camera mount at 2.5m on the front wall (15-degree downward tilt covering all 20 desks). Install the Mid Camera mount at 2.5m in the middle of the room (ceiling or elevated mount, angled to cover the back 12 desks).
3. **Device Assembly**:
   - Attach the AI HAT+ to the Front Raspberry Pi 5. Install active cooling.
   - Set up the Mid Raspberry Pi 5 with active cooling (heatsink + fan).
4. **Clock Sync**: Configure NTP synchronization on both Pis and verify alignment.

### Phase 2: Data Collection & Model Preparation

1. **Dataset Gathering**: Record footage from **both camera positions** (front and mid) of actors performing the specific cheating behaviors in the simulation room.
2. **Annotation**:
   - **Front model dataset**: Label images for all 4 classes (`person`, `hand`, `cell_phone`, `cheat_sheet`) using Roboflow or CVAT from the front camera perspective.
   - **Mid model dataset**: Label images for the 3-class set (`hand`, `cell_phone`, `cheat_sheet`) from the mid camera perspective (front-facing view of back 12 students).
   - Target: 1,500–2,000 annotated images per class, per model.
3. **Training** (on GPU machine, not on the Pi):
   - **Front detection model**: Fine-tune YOLOv11n from COCO-pretrained weights on the front dataset with 4 classes (`person`, `hand`, `cell_phone`, `cheat_sheet`).
   - **Mid detection model**: Fine-tune YOLOv11n from COCO-pretrained weights on the mid dataset with 3 classes (`hand`, `cell_phone`, `cheat_sheet`).
   - **Pose model**: No training needed — use pretrained YOLOv11n-pose weights directly.
4. **Optimization & Export**:
   - **Front models**: Export to HEF format using the Hailo Dataflow Compiler for AI HAT+ deployment.
   - **Mid model**: Export to NCNN format for CPU inference on the Mid Pi.

### Phase 3: System Integration

1. **Front Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install Hailo runtime (`hailo-all`).
   - Deploy the HEF detection and pose models.
   - Implement the ByteTrack-based tracking and behavioral analysis pipeline.
   - Implement evidence recording with circular buffer.

2. **Mid Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install NCNN runtime and Python dependencies.
   - Deploy the NCNN detection model.
   - Run the desk zone calibration tool to define desk edge boundaries for each of the back 12 seats.
   - Implement the desk-zone-based behavioral analysis pipeline.
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
   - *Scenario A*: Student tilts head to look at a neighbor's paper (Front Node — pose estimation).
   - *Scenario B*: Student hides hands under the table for extended period (Front Node — wrist keypoints disappear from pose; Mid Node — hand absence from desk zone triggers flag).
   - *Scenario C*: Student uses a cellphone on or near desk (Front Node — object detection for all 20; Mid Node — object detection for back 12).
   - *Scenario D*: Student uses a cheat sheet on or near desk (Front Node — object detection for all 20; Mid Node — object detection for back 12).
   - *Scenario E*: Student in the front 8 seats hides hands under table (Front Node only — verifies front coverage without Mid Node overlap).
   - *Scenario F*: Student directly hands a paper/note to the neighbor sitting beside them in the same row (Front Node only — wrist keypoint exits person bounding box laterally toward neighbor; both students flagged as a linked event).
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
3. **System Latency**: Time taken between the behavior occurring and the system flagging it, measured per node. Expected: Front Node < 1 second, Mid Node 1–3 seconds.
4. **Evidence Integrity**: Verifying that recorded clips are playable, correctly timestamped, and clearly show the flagged behavior from the appropriate camera angle.
5. **Evidence Merge Accuracy**: Verifying that the post-exam merge correctly combines and deduplicates events from both nodes without losing genuine separate incidents.
6. **Thermal Stability**: Ensuring both Raspberry Pi 5 units can run their respective inference workloads for the full 1.5-hour exam duration without throttling.
   - **Front Node**: Pi 5 + AI HAT+ running dual models (detection + pose).
   - **Mid Node**: Pi 5 CPU running continuous NCNN inference.
7. **Coverage Completeness**: Evaluating whether the dual-camera setup eliminates significant blind spots. The Front Node covers all 20 students for both object detection (cellphones, cheat sheets, hands) and pose-based behavioral analysis (head tilt, looking at neighbor, hands under table). The Mid Node provides supplemental close-range coverage of the back 12 students for small-object detection (cellphones, cheat sheets, hands below desk zone), improving detection reliability for students farther from the front camera.
