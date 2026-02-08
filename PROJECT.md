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
2. **Implement Real-Time Proctoring**: Deploying a dual-node system using two independent Raspberry Pi 5 units — one with a Raspberry Pi AI HAT+ (26 TOPS) for high-speed inference (Front Node) and one running a lightweight CPU-optimized model (Back Node) — for continuous, multi-angle monitoring.
3. **Generate Evidence**: Automatically recording short video clips with timestamps whenever cheating behavior is detected to serve as evidence for administrative review. A post-exam merge process unifies evidence from both nodes into a single, chronological timeline.

---

## Scope and Limitations

### Scope

The study is centered on a visual detection system deployed within a controlled simulation environment.

- **Behaviors Detected**: The system is trained to identify specific visual cues associated with cheating:
  - Passing of papers between students (above and below desk level).
  - Abnormal head tilting or sustained head turning towards a neighbor.
  - Hands hidden under the table.
  - Presence and usage of cellphones (on desk, near face, or hidden under desk).
  - Usage of cheat sheets.
- **Environment**: A simulation room designed to approximate a standard Philippine classroom (7m × 9m).
  - **Capacity**: The simulation involves 20 or fewer students (approximating a full class of 40–50).
  - **Layout**: Seats arranged with at least 1 meter of spacing, following post-pandemic DepEd and DOH guidelines to reduce occlusions and improve camera coverage.
  - **Conditions**: Average exam duration of 1 hour and 30 minutes.
- **Evidence**: Each detected event generates a locally stored video clip with a precise timestamp for review. Evidence from both nodes is merged post-exam into a unified, deduplicated timeline.

### Limitations

The system operates within defined constraints:

- **Visual Only**: It covers behaviors detectable within the cameras' fields of view, particularly around individual student tables.
- **Contextual Blind Spots**:
  - It cannot reliably distinguish cheat sheets that mimic the official questionnaire in size or appearance.
  - It cannot capture subtle, non-object-based behaviors such as hand signals or whispering.
- **Frame Rate Disparity**: The Back Node operates at a lower frame rate (8–12 FPS) than the Front Node (25–30 FPS) due to its CPU-only inference, which means very fast, transient movements may be missed by the rear camera.
- **Independent Nodes**: Because the two nodes operate independently, real-time cross-camera fusion is not available. Correlated evidence is linked only post-exam via timestamp alignment.
- **Environment**: Results gathered in the controlled simulation may differ from those in larger, more chaotic real-world classroom settings.

---

## System Architecture

The monitoring setup employs an **Independent Dual-Node Architecture**. Each node operates autonomously during the exam — running its own detection model, behavioral analysis, and evidence recording — with no network dependency between them. This design maximizes fault tolerance: if one node fails, the other continues operating unaffected. Evidence from both nodes is correlated post-exam using synchronized timestamps.

### 1. Front Node (Front View — Hailo-Accelerated)

The Front Node handles the computationally intensive workload, leveraging the AI HAT+ for high-speed inference.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: Raspberry Pi AI HAT+ (26 TOPS).
- **Models**:
  - **YOLOv11n (Object Detection)** — Runs on Hailo at 25–30 FPS, 640×640 resolution. Detects: `person`, `cell_phone`, `paper`, `hand`, `cheat_sheet`.
  - **YOLOv11n-pose (Pose Estimation)** — Runs on Hailo or CPU at 7–8 FPS (alternating with detection model). Provides facial and upper-body keypoints for head tilt and head turn analysis.
- **Tracking**: ByteTrack for persistent student identification across frames.
- **Export Format**: HEF (Hailo Executable Format).
- **Detection Responsibilities**: Cellphone usage (on/near desk), head tilting, looking at a neighbor's paper, cheat sheet on desk, above-desk paper passing.

### 2. Back Node (Back View — CPU-Only)

The Back Node provides a rear perspective with a streamlined, purpose-built detection pipeline optimized for CPU inference.

- **Device**: Raspberry Pi 5 (8GB RAM).
- **Accelerator**: None (CPU-only inference).
- **Model**:
  - **YOLOv11n (Object Detection, Narrowed)** — Runs on CPU via NCNN at 8–12 FPS, 320×320 resolution. Detects a reduced class set: `hand`, `cell_phone`, `paper`.
- **Tracking**: None. Uses **static zone-based detection** instead, with predefined regions mapped to each seat during calibration.
- **Export Format**: NCNN.
- **Detection Responsibilities**: Hands under table, hidden phones (under desk/on lap), below-desk paper passing, reaching behind towards a neighbor.

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
  - **Purpose**: Captures facial expressions, head movements, head orientation, objects on desks, and above-desk interactions between students.

- **Back Camera**:
  - **Location**: Mounted at the rear of the room.
  - **Height**: 2.5 meters.
  - **Angle**: Angled downwards at 15 degrees.
  - **Purpose**: Monitors hands under tables, items on laps, passing of papers behind backs or below desks, and screens of unauthorized devices hidden from the front view.

### Component List

- **Compute Modules**: 2× Raspberry Pi 5 (8GB RAM).
- **AI Accelerator**: 1× Raspberry Pi AI HAT+ (26 TOPS) for the Front Node only.
- **Imaging**: 2× High-resolution Wide-Angle Cameras (USB or MIPI-CSI) capable of clear video at 2.5m distance.
- **Storage**: High-endurance microSD cards (64GB+) for local video clip storage on each node.
- **Power**: Official USB-C Power Supplies (27W) for both units to support sustained compute loads.
- **Cooling**: Active coolers (heatsink + fan) for **both** Pis. The Front Node requires cooling for the AI HAT+ workload; the Back Node requires cooling for sustained CPU-based inference, which generates significant heat over the 1.5-hour exam duration.

---

## Software & Algorithm Design

### Detection Pipeline Architecture

Both nodes follow a modular pipeline structure, but with different components active on each:

```
aisentinel/
├── config.yaml                 # Thresholds, timers, camera settings, zone definitions
├── main.py                     # Orchestrator (selects front or back mode)
├── capture/
│   ├── camera.py               # Frame capture from camera feed
│   └── buffer.py               # Circular video ring buffer (last N seconds)
├── inference/
│   ├── detector.py             # YOLOv11 object detection (Hailo or NCNN)
│   └── pose.py                 # YOLOv11-pose for head tilt (Front Node only)
├── tracking/
│   └── tracker.py              # ByteTrack wrapper (Front Node only)
├── analysis/
│   ├── behavior_front.py       # Front Node rule engine (tracker-based)
│   ├── behavior_back.py        # Back Node rule engine (zone-based)
│   └── zones.py                # Static zone definitions per seat (Back Node)
├── evidence/
│   └── recorder.py             # Save clips + metadata on trigger
├── merge/
│   └── merge_evidence.py       # Post-exam evidence merging and deduplication
└── models/
    ├── front_detect.hef        # Front detection model (Hailo format)
    ├── front_pose.hef          # Front pose model (Hailo format)
    └── back_detect_ncnn/       # Back detection model (NCNN format)
```

---

### Front Node: Models & Detection Logic

#### Model 1 — YOLOv11n Object Detection (Hailo)

- **Resolution**: 640×640
- **Target FPS**: 25–30
- **Export Format**: HEF (Hailo Executable Format)
- **Classes**:

| Class Index | Class Name   | Description                              |
|-------------|--------------|------------------------------------------|
| 0           | `person`     | Student (pretrained from COCO, fine-tuned)|
| 1           | `cell_phone` | Unauthorized device                      |
| 2           | `paper`      | Exam paper or cheat sheet                |
| 3           | `hand`       | For tracking paper passing gestures      |
| 4           | `cheat_sheet`| Distinct from normal exam paper          |

#### Model 2 — YOLOv11n-pose Pose Estimation (Hailo or CPU)

- **Resolution**: 640×640
- **Target FPS**: 7–8 (runs on every 4th frame, alternating with detection model)
- **Purpose**: Provides keypoints (nose, eyes, ears, shoulders) for head orientation analysis.
- **No custom training required** — uses pretrained YOLOv11n-pose weights.

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
| **Head tilting**           | Ear-to-ear angle from pose keypoints: `atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)`                | Angle > 25–30° sustained for 3–5 seconds   |
| **Looking at neighbor**    | Nose position offset from shoulder midpoint: `abs(nose.x - shoulder_center.x)`                                    | Offset exceeds threshold for 3–5 seconds   |
| **Cheat sheet on desk**    | `paper` or `cheat_sheet` detected in unusual desk position (not centered where exam should be)                    | Sustained presence, confidence ≥ 0.5       |
| **Paper passing (above desk)** | `paper` + `hand` bounding boxes overlapping between two different `person` tracks within a 1–2 second window | Spatial overlap across two tracked persons |

---

### Back Node: Model & Detection Logic

#### Model — YOLOv11n Object Detection (NCNN, Narrowed)

- **Resolution**: 320×320
- **Target FPS**: 8–12
- **Export Format**: NCNN (optimized for ARM CPU)
- **Classes** (reduced set for faster inference):

| Class Index | Class Name   | Description                                    |
|-------------|--------------|------------------------------------------------|
| 0           | `hand`       | Primary class — detecting hands below desk     |
| 1           | `cell_phone` | Phones held under desk, on lap, screen visible |
| 2           | `paper`      | Papers passed below desk or behind backs       |

#### Zone-Based Detection (No Tracker)

Instead of tracking individual students, the Back Node uses **predefined static zones** mapped during a one-time calibration step before each exam. Since the camera is fixed, these zones remain stable throughout the session.

**Zone types per seat**:
- **`desk_area`**: The visible desk surface region for that seat.
- **`under_desk`**: The region below the desk line for that seat.
- **`between_seats`**: The gap region between adjacent students.
- **`lap_area`**: The region corresponding to the student's lap.

**Calibration process**: Before the exam, a calibration script captures a reference frame of the empty room. The operator draws bounding regions for each seat's zones using a simple GUI tool, and these are saved to `config.yaml`.

#### Back Node Behavioral Logic

| Behavior                      | Detection Method                                                       | Trigger Condition                              |
|-------------------------------|------------------------------------------------------------------------|------------------------------------------------|
| **Hands under table**         | `hand` detected in `under_desk` zone                                   | Sustained for 5–8 seconds                      |
| **Hidden phone (under desk)** | `cell_phone` detected in `under_desk` or `lap_area` zone              | Immediate flag at confidence ≥ 0.6             |
| **Paper passing (below desk)**| `paper` detected in `between_seats` zone                               | Any detection at confidence ≥ 0.5              |
| **Reaching behind**           | `hand` detected in `between_seats` zone                                | Sustained for 3–5 seconds                      |

---

### Evidence Generation (Both Nodes)

Both nodes implement identical evidence recording logic:

- **Circular Buffer**: Each node maintains a rolling video buffer of the last 10–15 seconds of footage at full camera frame rate (30 FPS), independent of the inference frame rate.
- **Trigger**: When a behavioral flag is raised.
- **Action**: The system saves the buffered footage (10 seconds before the event) plus records an additional 10 seconds after the trigger, producing a ~20-second evidence clip.
- **Metadata**: Each clip's filename encodes: `{node}_{timestamp}_{behavior_type}_{confidence}.mp4`
  - Example: `front_20260207_143022_cellphone_087.mp4`
  - Example: `back_20260207_143055_hands_under_table_072.mp4`
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
2. **Mounting**: Install camera mounts at 2.5m height on front and back walls. Verify the 15-degree tilt covers all desks from both angles.
3. **Device Assembly**:
   - Attach the AI HAT+ to the Front Raspberry Pi 5. Install active cooling.
   - Set up the Back Raspberry Pi 5 with active cooling (heatsink + fan).
4. **Clock Sync**: Configure NTP synchronization on both Pis and verify alignment.

### Phase 2: Data Collection & Model Preparation

1. **Dataset Gathering**: Record footage from **both camera positions** (front and back) of actors performing the specific cheating behaviors in the simulation room.
2. **Annotation**:
   - **Front model dataset**: Label images for all 5 classes (`person`, `cell_phone`, `paper`, `hand`, `cheat_sheet`) using Roboflow or CVAT.
   - **Back model dataset**: Label images for the reduced 3-class set (`hand`, `cell_phone`, `paper`) from the rear camera perspective.
   - Target: 1,500–2,000 annotated images per class, per model.
3. **Training** (on GPU machine, not on the Pi):
   - **Front detection model**: Fine-tune YOLOv11n from COCO-pretrained weights on the front dataset.
   - **Back detection model**: Fine-tune YOLOv11n from COCO-pretrained weights on the back dataset with reduced classes.
   - **Pose model**: No training needed — use pretrained YOLOv11n-pose weights directly.
4. **Optimization & Export**:
   - **Front models**: Export to HEF format using the Hailo Dataflow Compiler for AI HAT+ deployment.
   - **Back model**: Export to NCNN format for CPU inference on the Back Pi.

### Phase 3: System Integration

1. **Front Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install Hailo runtime (`hailo-all`).
   - Deploy the HEF detection and pose models.
   - Implement the ByteTrack-based tracking and behavioral analysis pipeline.
   - Implement evidence recording with circular buffer.

2. **Back Node Deployment**:
   - Flash Raspberry Pi OS (64-bit, Bookworm). Install NCNN runtime and Python dependencies.
   - Deploy the NCNN detection model.
   - Run the zone calibration tool to define seat zones for the specific room layout.
   - Implement the zone-based behavioral analysis pipeline.
   - Implement evidence recording with circular buffer.

3. **Evidence Merge Tool**:
   - Develop and test the `merge_evidence.py` script on the review workstation.

### Phase 4: Testing & Simulation

1. **Thermal Stability Test**: Run both nodes continuously for 1.5 hours with active inference to verify no CPU/NPU throttling occurs. Monitor temperatures:
   ```bash
   vcgencmd measure_temp  # Check during operation; target < 80°C
   ```
2. **Zone Calibration Verification**: Confirm that the Back Node's predefined zones accurately map to each seat from the rear camera's perspective.
3. **Pilot Run**: Conduct a mock exam with 20 participants.
4. **Scenario Testing**: Instruct participants to act out specific scenarios:
   - *Scenario A*: Student passes a paper to a neighbor above the desk (Front Node target).
   - *Scenario B*: Student passes a paper below the desk (Back Node target).
   - *Scenario C*: Student checks a phone under the desk (Back Node target).
   - *Scenario D*: Student holds a phone near their face (Front Node target).
   - *Scenario E*: Student holds a cheat sheet (Front Node target).
   - *Scenario F*: Student tilts head to look at a neighbor's paper (Front Node target).
   - *Scenario G*: Student hides hands under the table for extended period (Back Node target).
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
3. **System Latency**: Time taken between the behavior occurring and the system flagging it, measured per node. Expected: Front Node < 1 second, Back Node 1–3 seconds.
4. **Evidence Integrity**: Verifying that recorded clips are playable, correctly timestamped, and clearly show the flagged behavior from the appropriate camera angle.
5. **Evidence Merge Accuracy**: Verifying that the post-exam merge correctly combines and deduplicates events from both nodes without losing genuine separate incidents.
6. **Thermal Stability**: Ensuring both Raspberry Pi 5 units can run their respective inference workloads for the full 1.5-hour exam duration without throttling.
   - **Front Node**: Pi 5 + AI HAT+ running dual models (detection + pose).
   - **Back Node**: Pi 5 CPU running continuous NCNN inference.
7. **Coverage Completeness**: Evaluating whether the dual-camera setup eliminates significant blind spots, particularly for behaviors that are only visible from one angle (e.g., under-desk activity only visible from behind, facial orientation only visible from front).
