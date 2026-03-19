# Mid Node Setup Guide (Raspberry Pi 5 + Hailo-8 AI HAT+)

This guide walks you through setting up the Mid Node environment on a Raspberry Pi 5 with the Hailo-8 AI HAT+ (26 TOPS). The Mid Node covers the back 12 students (students 9–20) and runs the same full detection pipeline as the Front Node.

---

## 1. Prerequisites

- **Hardware**: Raspberry Pi 5 (8GB RAM) with Raspberry Pi AI HAT+ (26 TOPS) and active cooling.
- **OS**: Raspberry Pi OS (64-bit, Bookworm or Trixie).
- **Camera**: USB Webcam connected.
- **Internet Access**: Required for installing packages.

---

## 2. Environment Setup

### Step 2.1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2.2: Install Hailo Runtime

```bash
sudo apt install hailo-all -y
```

Verify the Hailo device is recognized:

```bash
hailortcli fw-control identify
# Should show: Hailo-8, 26 TOPS
```

### Step 2.3: Create Virtual Environment

```bash
# Install python3-venv if not already installed
sudo apt install python3-venv -y

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 2.4: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install Ultralytics (includes standard YOLO dependencies)
pip install ultralytics

# Install OpenCV
pip install opencv-python-headless

# Install additional dependencies
pip install numpy lap flask

# hailo_platform is already installed via hailo-all (system package)
```

> **Note**: If you face issues with `opencv-python` on RPi, install system dependencies:
> `sudo apt install libgl1-mesa-glx`

---

## 3. Deploy Models

The Mid Node uses the same HEF models as the Front Node. Copy them from the Front Node or download fresh:

```bash
mkdir -p ~/AISENTINEL/models

# Copy detection model
scp sentinel@aisentinel-1:~/AISENTINEL/models/detect.hef ~/AISENTINEL/models/

# Copy pose model
scp sentinel@aisentinel-1:~/AISENTINEL/models/pose.hef ~/AISENTINEL/models/
```

Or download pre-compiled models from the Hailo Model Zoo:

```bash
cd ~/AISENTINEL/models

# Download pose model (yolov8s_pose — fast, good accuracy)
wget -O yolov8s_pose.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov8s_pose.hef
```

---

## 4. Camera Setup

Verify the USB webcam is recognized:

```bash
# List video devices
ls /dev/video*

# Check camera details
v4l2-ctl --list-devices

# Quick capture test
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test_frame.jpg
```

For detailed camera setup instructions, see `CAMERA_SETUP_GUIDE.md`.

---

## 5. Desk Zone Calibration

Before the exam, calibrate desk zone boundaries for the back 12 seats from the mid camera's perspective:

```bash
cd ~/AISENTINEL
python3 calibrate_desk_zones.py --camera 0 --seats 12
```

This opens a GUI where you draw polygon ROIs for each desk. The zones are saved to `config.yaml`.

---

## 6. Clock Synchronization

Both nodes must have synchronized clocks for post-exam evidence merging:

```bash
# Ensure NTP sync is active
sudo timedatectl set-ntp true

# Verify sync status
timedatectl status
```

If offline, sync to the Front Node over a direct Ethernet connection.

---

## 7. Running the Mid Node

```bash
cd ~/AISENTINEL
source venv/bin/activate

# Start the mid node detection pipeline
python3 main.py --mode mid
```

---

## 8. Troubleshooting

- **Hailo Device Not Found**:
  - Check PCIe connection: `lspci | grep Hailo`
  - Restart Hailo service: `sudo systemctl restart hailort.service`
  - Re-identify: `hailortcli fw-control identify`

- **Camera Not Found**:
  - Check connections and try `--source 1` or `--source 2`.
  - Verify with `ls /dev/video*`.

- **Low FPS**:
  - Check temperature: `vcgencmd measure_temp` (target < 80°C).
  - Ensure using `hailo8` models, not `hailo8l`.
  - Verify active cooling is working.

- **"Illegal Instruction" or Crash**:
  - Ensure 64-bit OS: `uname -m` should say `aarch64`.
  - Update Hailo runtime: `sudo apt update && sudo apt upgrade hailo-all`.

---

## 9. Next Steps

Once the Mid Node is running:

1. **Verify detection** on all 12 back students from the mid camera angle.
2. **Tune thresholds** using `calibrate_head_behavior.py` for the mid camera perspective.
3. **Run a pilot exam** and verify evidence generation.
4. **Test evidence merge** with the Front Node's evidence.
