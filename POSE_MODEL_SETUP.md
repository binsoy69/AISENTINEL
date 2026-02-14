# Pose Estimation Setup Guide — RPi 5 + Hailo-8 (26 TOPS) + USB Webcam

This guide walks you through getting a **free, open-source pose estimation model** running on your Raspberry Pi 5 with the Hailo-8 AI HAT+ (26 TOPS) using a USB webcam.

## Your Setup

| Component | Details |
|-----------|---------|
| **Board** | Raspberry Pi 5 |
| **AI Accelerator** | Hailo-8 AI HAT+ (26 TOPS) |
| **Architecture** | `hailo8` |
| **Camera** | USB webcam (v4l2) |
| **OS** | Raspberry Pi OS Trixie (64-bit) |
| **Hailo Stack** | `hailo-all` installed and verified |
| **Use Case** | Attention monitoring (head pose, gaze direction) |

## Model Options (Pre-compiled, Free, Open-Source)

Pre-compiled HEF models from the **Hailo Model Zoo** — no conversion needed:

| Model | Accuracy (mAP) | FPS (Hailo-8) | Size | Best For |
|-------|----------------|---------------|------|----------|
| **yolov8s_pose** | 59.2 | ~393 | Small | Real-time, lower accuracy |
| **yolov8m_pose** | 64.3 | ~66 | Medium | Best balance (recommended) |

Both detect **17 COCO keypoints** per person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).

> **Note:** These are YOLOv8 models from Ultralytics (open-source, AGPL-3.0 license). Pre-compiled YOLOv11 pose HEFs are not yet available in the Hailo Model Zoo. YOLOv8 pose works excellently for attention monitoring.

---

## Quick Start (Simplest Path)

### Step 1: Download the Pre-compiled HEF Model

SSH into your Raspberry Pi and run:

```bash
# Create models directory
mkdir -p ~/AISENTINEL/models
cd ~/AISENTINEL/models

# Download yolov8s_pose (recommended — fast, good accuracy)
wget -O yolov8s_pose.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov8s_pose.hef

# OR download yolov8m_pose (higher accuracy, still very fast on Hailo-8)
wget -O yolov8m_pose.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov8m_pose.hef
```

No authentication required — these are publicly hosted by Hailo.

### Step 2: Verify Your USB Webcam

```bash
# List video devices
ls /dev/video*

# Check camera details
v4l2-ctl --list-devices

# Quick capture test (optional, requires ffmpeg)
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test_frame.jpg
```

Your USB webcam will typically be at `/dev/video0` or `/dev/video2`.

### Step 3: Install the Hailo Apps Framework

The **hailo-apps** framework provides ready-to-run pose estimation pipelines with USB webcam support:

```bash
cd ~
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```

### Step 4: Run Pose Estimation

```bash
# Basic run with USB webcam (auto-detects camera and model)
hailo-pose --input usb --show-fps

# Specify your downloaded model explicitly
hailo-pose --input usb --hef-path ~/AISENTINEL/models/yolov8s_pose.hef --show-fps

# Target a specific camera device
hailo-pose --input /dev/video0 --show-fps

# Force Hailo-8 architecture (if auto-detect fails)
hailo-pose --input usb --arch hailo8 --show-fps
```

That's it! You should see a live video window with pose skeletons overlaid on detected persons.

---

## Method 2: Using the AISENTINEL Test Script

The project includes a custom pose estimation script with **behavioral analysis** (head tilt, head turn detection) designed for attention monitoring.

### Step 1: Install Python Dependencies

```bash
pip install opencv-python numpy flask ultralytics
```

### Step 2: Run in CPU Mode (Quick Test, No HEF Needed)

```bash
cd ~/AISENTINEL/tests
python3 pose_usb_cam_hailo.py --cpu --port 8080
```

This auto-downloads `yolo11n-pose.pt` and runs on CPU at 2-4 FPS. Open `http://<pi-ip>:8080` in your browser to view the stream.

### Step 3: Run with Hailo Acceleration

```bash
cd ~/AISENTINEL/tests
python3 pose_usb_cam_hailo.py --model ~/AISENTINEL/models/yolov8s_pose.hef --port 8080
```

> **Note:** The AISENTINEL script currently expects YOLOv11 output format (56 channels). If using the YOLOv8 HEF models, the output format is the same (56 channels: 4 bbox + 1 conf + 17×3 keypoints), so they are compatible.

### Behavioral Analysis Options

The AISENTINEL script includes attention monitoring features:

```bash
python3 pose_usb_cam_hailo.py \
  --model ~/AISENTINEL/models/yolov8s_pose.hef \
  --port 8080 \
  --confidence 0.5 \
  --kpt-threshold 0.3 \
  --head-tilt-threshold 25 \
  --head-turn-threshold 0.4 \
  --sustained-duration 3 \
  --log behaviors.log \
  --save-frames
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence` | 0.5 | Person detection confidence threshold |
| `--kpt-threshold` | 0.3 | Keypoint confidence threshold |
| `--head-tilt-threshold` | 25° | Angle to flag head tilt |
| `--head-turn-threshold` | 0.4 | Ratio to flag head turn |
| `--sustained-duration` | 3s | Duration before triggering alert |
| `--log` | — | Log behavioral alerts to file |
| `--save-frames` | — | Save frames when alerts trigger |

---

## Method 3: Standalone Python with Hailo Runtime (Custom Pipeline)

For full control without the hailo-apps framework:

### Install Hailo Python API

```bash
pip install hailort
```

### Minimal Pose Estimation Script

```python
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, ConfigureParams, \
    InputVStreamParams, OutputVStreamParams, FormatType

# Load model
hef = HEF("~/AISENTINEL/models/yolov8s_pose.hef")

# Configure device
params = VDevice.create_params()
with VDevice(params) as vdevice:
    # Configure network
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, configure_params)[0]

    # Set up streams
    input_vstreams_params = InputVStreamParams.make(network_group)
    output_vstreams_params = OutputVStreamParams.make(network_group)

    # Open camera
    cap = cv2.VideoCapture(0)  # USB webcam

    with network_group.activate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess: resize to 640x640, normalize
            input_data = cv2.resize(frame, (640, 640))
            input_data = input_data.astype(np.float32) / 255.0

            # Run inference
            # ... (send to input vstream, read from output vstream)
            # Parse 56-channel output for keypoints

            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
```

> This is a skeleton — for a complete implementation, refer to `tests/pose_usb_cam_hailo.py` or the hailo-apps standalone examples.

---

## Advanced: Convert Your Own Model to HEF

If you want to use a different model (e.g., YOLOv11-pose), you need the **Hailo Dataflow Compiler** on a development machine (not the Pi).

### Step 1: Export to ONNX (on dev machine)

```bash
pip install ultralytics
python export_pose_model.py
```

Or manually:

```python
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=11)
```

### Step 2: Compile ONNX to HEF (requires Hailo SDK)

```bash
hailomz compile yolo11n-pose.onnx \
    --hw-arch hailo8 \
    --performance \
    --output yolo11n-pose.hef
```

> **Important:** Use `--hw-arch hailo8` (not `hailo8l`). Your AI HAT+ has the full Hailo-8 chip (26 TOPS).

### Step 3: Transfer to Pi

```bash
scp yolo11n-pose.hef pi@<pi-ip>:~/AISENTINEL/models/
```

---

## Model Output Format

Both YOLOv8 and YOLOv11 pose models output **56 channels per detection**:

```
[x, y, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ...]
 └─────┬─────┘ └┬─┘ └──────────────────┬──────────────────────────────────────┘
      bbox     conf              17 keypoints × 3 = 51 values
```

**Keypoint order (COCO format):**

```
 0: nose           5: left_shoulder   11: left_hip
 1: left_eye       6: right_shoulder  12: right_hip
 2: right_eye      7: left_elbow      13: left_knee
 3: left_ear       8: right_elbow     14: right_knee
 4: right_ear      9: left_wrist      15: left_ankle
                  10: right_wrist     16: right_ankle
```

**Keypoints most relevant for attention monitoring:**
- **0 (nose)**: Gaze direction indicator
- **1-4 (eyes, ears)**: Head orientation
- **5-6 (shoulders)**: Body facing direction
- Head tilt = angle between ears (keypoints 3 and 4)
- Head turn = nose offset from shoulder midpoint

---

## Verification & Testing

### 1. Verify Hailo Device

```bash
hailortcli fw-control identify
# Should show: Hailo-8, 26 TOPS
```

### 2. Verify USB Camera

```bash
v4l2-ctl --list-devices
# Should list your USB webcam
```

### 3. Test Pose Estimation

```bash
# Quick test with hailo-apps
cd ~/hailo-apps && source setup_env.sh
hailo-pose --input usb --show-fps

# Or with AISENTINEL script (CPU fallback)
cd ~/AISENTINEL/tests
python3 pose_usb_cam_hailo.py --cpu --port 8080
```

### 4. Expected Performance

| Mode | FPS | Inference Time | Notes |
|------|-----|----------------|-------|
| Hailo-8 (yolov8s_pose) | 30+ | ~3ms | Excellent for real-time |
| Hailo-8 (yolov8m_pose) | 30+ | ~15ms | Best accuracy |
| CPU fallback | 2-4 | ~300ms | For testing only |

### 5. Monitor Temperature

```bash
watch -n 1 vcgencmd measure_temp
# Keep below 80°C — use active cooling if needed
```

---

## Troubleshooting

### USB Camera Not Detected

```bash
# Check if camera is recognized
lsusb
dmesg | tail -20

# Try different video device
ls /dev/video*
# USB webcams are often /dev/video0 or /dev/video2
```

### Hailo Device Not Found

```bash
# Check PCIe connection
lspci | grep Hailo

# Restart Hailo service
sudo systemctl restart hailort.service

# Re-identify
hailortcli fw-control identify
```

### Low FPS

1. **Thermal throttling**: `vcgencmd measure_temp` — add active cooling
2. **Wrong architecture**: Ensure using `hailo8` models, not `hailo8l`
3. **Camera bottleneck**: Try lower resolution: `--width 320 --height 240`

### Model Loading Errors

```bash
# Verify HEF file is valid
hailortcli run yolov8s_pose.hef

# Check file integrity
ls -lh ~/AISENTINEL/models/yolov8s_pose.hef
# yolov8s_pose.hef should be ~10-20 MB
```

### Trixie Compatibility Notes

RPi OS Trixie has known edge cases with Hailo software. If you encounter issues:

1. **Driver version mismatch**: Run `sudo apt update && sudo apt upgrade hailo-all`
2. **Python 3.13 issues**: Some Hailo Python bindings may need `pip install --break-system-packages`
3. **DKMS driver**: If kernel updates break Hailo, run `sudo dkms autoinstall`
4. **Fallback**: Raspberry Pi OS Bookworm is the most battle-tested OS for Hailo

---

## Hailo-Apps CLI Reference

| Flag | Description |
|------|-------------|
| `--input usb` | Auto-detect USB camera |
| `--input /dev/video0` | Specific camera device |
| `--hef-path <path>` | Custom HEF model file |
| `--arch hailo8` | Force Hailo-8 architecture |
| `--show-fps` | Display FPS counter |
| `--frame-rate 30` | Target frame rate |
| `--disable-sync` | Max speed (benchmarking) |
| `--log-level debug` | Verbose logging |

---

## Summary: Fastest Path to Running Pose Estimation

```bash
# 1. Download model (30 seconds)
mkdir -p ~/AISENTINEL/models && cd ~/AISENTINEL/models
wget -O yolov8s_pose.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov8s_pose.hef

# 2. Install hailo-apps (5 minutes)
cd ~ && git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps && sudo ./install.sh && source setup_env.sh

# 3. Run! (immediate)
hailo-pose --input usb --hef-path ~/AISENTINEL/models/yolov8s_pose.hef --show-fps
```

Three commands. That's it.

---

## Additional Resources

- [Hailo Model Zoo — Pose Models](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_pose_estimation.rst)
- [Hailo Apps Framework](https://github.com/hailo-ai/hailo-apps)
- [Ultralytics YOLOv8 Pose Docs](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Hailo Community Forum](https://community.hailo.ai/)
