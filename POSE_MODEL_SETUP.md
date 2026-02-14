# YOLOv11n-pose Model Conversion Guide for Hailo

This guide explains how to convert the YOLOv11n-pose model to Hailo's HEF format for accelerated inference on the Raspberry Pi 5 + Hailo AI HAT+.

## Overview

**Model**: YOLOv11n-pose (Ultralytics)
**Input**: 640×640 RGB image
**Output**: 17 COCO keypoints per person
**Target FPS**: 7-8 FPS on Hailo8L
**Format**: HEF (Hailo Executable Format)

## Prerequisites

### On Development Machine (for conversion)
- Python 3.8+
- Hailo Dataflow Compiler (part of Hailo SDK)
- Ultralytics package: `pip install ultralytics`

### On Raspberry Pi (for inference)
- Hailo AI HAT+ or AI Kit installed
- `hailo-all` package: `sudo apt install hailo-all`
- Python packages: `pip install opencv-python numpy flask ultralytics`

---

## Method 1: Convert from PyTorch to HEF (Recommended)

### Step 1: Export to ONNX

Create a Python script to export the model:

```python
from ultralytics import YOLO

# Load the pretrained YOLOv11n-pose model
model = YOLO('yolov11n-pose.pt')

# Export to ONNX format
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=11,
)

print("✓ Model exported to yolov11n-pose.onnx")
```

Run the script:
```bash
python3 export_pose_model.py
```

This will generate `yolov11n-pose.onnx` in the current directory.

### Step 2: Convert ONNX to HEF

Using the Hailo Dataflow Compiler:

```bash
hailomz compile yolov11n-pose.onnx \
    --hw-arch hailo8l \
    --performance \
    --output yolov11n-pose.hef
```

**Parameters:**
- `--hw-arch hailo8l`: Target architecture (Hailo8L on RPi 5 AI HAT+)
- `--performance`: Optimize for speed over accuracy
- `--output`: Output HEF file name

**Expected output:**
```
Compiling model...
Optimizing for hailo8l...
Writing HEF file: yolov11n-pose.hef
✓ Compilation successful
```

### Step 3: Transfer to Raspberry Pi

```bash
# Create models directory on Pi
ssh pi@<pi-ip> "mkdir -p ~/AISENTINEL/models"

# Transfer the HEF file
scp yolov11n-pose.hef pi@<pi-ip>:~/AISENTINEL/models/

# Verify transfer
ssh pi@<pi-ip> "ls -lh ~/AISENTINEL/models/yolov11n-pose.hef"
```

---

## Method 2: Use Hailo Model Zoo (If Available)

The Hailo Model Zoo may have pre-compiled YOLOv11-pose models.

### Check availability:
```bash
hailomz list | grep -i "pose\|yolo"
```

### Download if available:
```bash
hailomz get yolov11n-pose --hw-arch hailo8l
```

This downloads the pre-compiled .hef file directly, skipping the conversion process.

---

## Method 3: Quick Start with CPU Fallback (No .hef needed)

For immediate testing without HEF conversion, use CPU mode:

```bash
cd ~/AISENTINEL/tests
python3 pose_usb_cam_hailo.py --cpu --port 8080
```

**What happens:**
- Ultralytics automatically downloads `yolov11n-pose.pt`
- Runs on CPU (2-4 FPS, slower but functional)
- Good for testing the pipeline while HEF conversion is in progress

---

## Verification

### Test the HEF Model

Once you have the .hef file on the Raspberry Pi:

```bash
cd ~/AISENTINEL/tests
python3 pose_usb_cam_hailo.py --model ../models/yolov11n-pose.hef --port 8080
```

**Expected output:**
```
============================================================
YOLOv11 Pose Estimation Test - RPi 5 + Hailo
AISENTINEL Project
============================================================
[INFO] Loading HEF model: ../models/yolov11n-pose.hef
[INFO] Model input shape : (640, 640, 3)
[INFO] Model output layer: output0 -> (1, 56, 8400)
[INFO] NMS on-chip       : NO (Python NMS)
[INFO] Hailo device ready.
[INFO] USB camera found at /dev/video0
[INFO] Camera resolution: 640x480 @ 30fps

[WEB] Stream at http://192.168.1.100:8080

[INFO] Starting pose estimation... Press Ctrl+C to quit
```

### Check Performance

Monitor FPS and inference time:
```
[Frame 30] FPS: 7.8, Detections: 1
[Frame 60] FPS: 7.9, Detections: 1
[Frame 90] FPS: 8.0, Detections: 2
```

**Target metrics:**
- FPS: 7-8 (Hailo mode) or 2-4 (CPU mode)
- Inference time: 125-140ms per frame (Hailo)
- All 17 keypoints detected on frontal persons

### Monitor Temperature

During extended testing:
```bash
watch -n 1 vcgencmd measure_temp
```

Keep temperature < 80°C (use active cooling if needed).

---

## Troubleshooting

### Issue: ONNX Export Fails

**Error:** `ModuleNotFoundError: No module named 'onnx'`

**Solution:**
```bash
pip install onnx onnxruntime
```

### Issue: Hailo Compiler Not Found

**Error:** `hailomz: command not found`

**Solution:**
Install the Hailo SDK on your development machine:
1. Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/)
2. Follow installation instructions for your OS
3. Verify: `hailomz --version`

### Issue: HEF Model Not Loading on Pi

**Error:** `[ERROR] HEF model not found: yolov11n-pose.hef`

**Solution:**
```bash
# Check file exists
ls -lh ~/AISENTINEL/models/yolov11n-pose.hef

# Use absolute path
python3 pose_usb_cam_hailo.py --model /home/pi/AISENTINEL/models/yolov11n-pose.hef
```

### Issue: Low FPS (< 5 FPS) with Hailo

**Possible causes:**
1. **Thermal throttling**: Check `vcgencmd measure_temp`, add cooling
2. **Wrong mode**: Verify not using `--cpu` flag
3. **Camera resolution**: Try lower resolution: `--width 320 --height 240`

### Issue: Keypoints Not Detected

**Solutions:**
- Lower keypoint threshold: `--kpt-threshold 0.2`
- Increase person confidence: `--confidence 0.6`
- Ensure good lighting (frontal, even illumination)
- Check camera focus

---

## Model Output Format

YOLOv11-pose outputs **56 channels per detection**:

```
[x, y, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ...]
 └─────┬─────┘ └┬─┘ └──────────────────┬──────────────────────────────────────┘
      bbox     conf              17 keypoints × 3 = 51 values
```

**Keypoint order (COCO format):**
```
0: nose          5: left_shoulder   11: left_hip
1: left_eye      6: right_shoulder  12: right_hip
2: right_eye     7: left_elbow      13: left_knee
3: left_ear      8: right_elbow     14: right_knee
4: right_ear     9: left_wrist      15: left_ankle
                10: right_wrist     16: right_ankle
```

---

## Performance Optimization

### For Better FPS:
1. **Use Hailo mode** (not CPU): 7-8 FPS vs 2-4 FPS
2. **Lower camera resolution**: `--width 320 --height 240`
3. **Reduce confidence threshold**: `--confidence 0.4` (fewer detections)
4. **Disable frame saving**: Don't use `--save-frames` during testing
5. **Active cooling**: Add fan to Raspberry Pi

### For Better Accuracy:
1. **Higher confidence**: `--confidence 0.6`
2. **Higher keypoint threshold**: `--kpt-threshold 0.4`
3. **Better lighting**: Frontal, even, bright
4. **Camera focus**: Ensure subjects are in focus

---

## Next Steps

After successful model conversion and testing:

1. **Integrate with tracking**: Use ByteTrack for persistent person IDs
2. **Tune thresholds**: Adjust head tilt/turn thresholds for your use case
3. **Deploy to production**: Set up systemd service for auto-start
4. **Monitor performance**: Log FPS and detection quality

---

## Additional Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)

---

## Support

For issues with:
- **Model conversion**: Check Hailo Developer Zone forums
- **Test program**: See main README or open GitHub issue
- **Raspberry Pi setup**: See CAMERA_SETUP_GUIDE.md
- **Hailo installation**: See Hailo official documentation
