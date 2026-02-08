# USB Camera Setup and Testing Guide for Raspberry Pi

This guide provides instructions for setting up and testing USB cameras on Raspberry Pi for the AISENTINEL project.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Setup](#software-setup)
- [Installation](#installation)
- [Running the Test Script](#running-the-test-script)
- [Troubleshooting](#troubleshooting)
- [Camera Configuration](#camera-configuration)

---

## Hardware Requirements

- **Raspberry Pi** (3, 4, or 5 recommended)
- **USB Camera** (UVC-compatible webcam)
- **Power Supply** (Official Raspberry Pi power adapter recommended)
- **MicroSD Card** (16GB+ with Raspberry Pi OS installed)
- **Display** (for GUI-based testing) or SSH access

### Supported Cameras

Most USB webcams that support UVC (USB Video Class) protocol work with Raspberry Pi, including:

- Logitech C270, C310, C920, C930e
- Microsoft LifeCam
- Generic USB webcams
- Raspberry Pi Camera Module (via USB adapter)

---

## Software Setup

### 1. Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Required Packages

```bash
# Install Python 3 and pip (usually pre-installed)
sudo apt install python3 python3-pip -y

# Install OpenCV dependencies
sudo apt install -y \
    python3-opencv \
    libopencv-dev \
    python3-picamera2 \
    v4l-utils \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev

# Install OpenCV via pip (alternative method - see note below)
pip3 install opencv-python opencv-contrib-python
```

> [!IMPORTANT]
> **If you get "error: externally-managed-environment"**  
> This error occurs on Raspberry Pi OS Bookworm (Debian 12) and later. Use one of the solutions below instead of pip3.

**Solution 1: Use APT (Recommended for Raspberry Pi)**

```bash
# This is the best approach for Raspberry Pi
sudo apt install python3-opencv -y

# Verify installation
python3 -c "import cv2; print(cv2.__version__)"
```

**Solution 2: Use Virtual Environment**

```bash
# Create virtual environment
python3 -m venv ~/aisentinel-env

# Activate it
source ~/aisentinel-env/bin/activate

# Now pip install works
pip install opencv-python

# Remember to activate this environment before running the script
```

**Solution 3: Use pipx (For Isolated Tools)**

```bash
# Install pipx
sudo apt install pipx -y

# Install opencv in isolated environment
pipx install opencv-python
```

**Solution 4: Override (Not Recommended)**

```bash
# Use --break-system-packages flag (use with caution)
pip3 install opencv-python --break-system-packages
```

### 3. Install Additional Python Dependencies

```bash
# Install numpy (required by OpenCV)
pip3 install numpy

# Optional: Install imutils for easier camera handling
pip3 install imutils
```

### 4. Enable Camera Support

```bash
# Add your user to the video group
sudo usermod -a -G video $USER

# Reboot to apply changes
sudo reboot
```

---

## Installation

### 1. Download the Test Script

Copy the `camera_test.py` script to your Raspberry Pi:

```bash
# If using git clone
cd /home/pi/AISENTINEL/tests
# The script should already be in the directory

# Make the script executable
chmod +x camera_test.py
```

### 2. Verify Camera Detection

```bash
# List all video devices
ls -l /dev/video*

# Get detailed camera information
v4l2-ctl --list-devices

# List all cameras with capabilities
v4l2-ctl --all
```

Expected output:

```
/dev/video0
/dev/video1  (if multiple cameras)
```

---

## Running the Test Script

### Basic Usage

```bash
# Run with default camera (index 0)
python3 camera_test.py

# Run with specific camera index
python3 camera_test.py 1
```

### Interactive Menu

The script provides an interactive menu with the following options:

1. **List available cameras** - Scans and displays all connected USB cameras
2. **Display camera information** - Shows camera properties (resolution, FPS, etc.)
3. **Capture test image** - Takes a snapshot and saves it to `test_captures/` directory
4. **Live preview** - Opens a window showing real-time camera feed
5. **Run performance benchmark** - Tests camera performance and calculates average FPS
6. **Run all tests** - Executes all tests sequentially
7. **Exit** - Closes the application

### Live Preview Controls

When running live preview (option 4):

- **Press 'q'** - Quit preview
- **Press 's'** - Save snapshot
- **Press 'i'** - Display camera information in terminal

### Command Line Examples

```bash
# Quick test - list all cameras
python3 camera_test.py
# Then select option 1

# Full automated test
python3 camera_test.py
# Then select option 6
```

---

## Troubleshooting

### Common Issues

#### 1. No Camera Detected

**Problem:** `✗ No cameras detected!`

**Solutions:**

```bash
# Check if camera is physically connected
lsusb

# Check video devices
ls -l /dev/video*

# Verify permissions
groups $USER  # Should include 'video'

# If not in video group:
sudo usermod -a -G video $USER
sudo reboot
```

#### 2. Permission Denied

**Problem:** `[Error] Cannot open camera: Permission denied`

**Solutions:**

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Fix device permissions (temporary)
sudo chmod 666 /dev/video0

# Reboot
sudo reboot
```

#### 3. OpenCV Import Error

**Problem:** `ModuleNotFoundError: No module named 'cv2'`

**Solutions:**

```bash
# Try pip3 installation
pip3 install opencv-python

# Or use apt (recommended for Raspberry Pi)
sudo apt install python3-opencv -y

# Verify installation
python3 -c "import cv2; print(cv2.__version__)"
```

#### 4. Display Error (Headless Setup)

**Problem:** `Cannot connect to X server`

**Solutions:**

For headless Raspberry Pi (no display), you have two options:

**Option A: Use X11 Forwarding via SSH**

```bash
# On your local machine, connect with X11 forwarding
ssh -X pi@raspberrypi.local

# Then run the script
python3 camera_test.py
```

**Option B: Modify Script for Headless Mode**

Create a headless version that only captures images without display:

```python
# Use options 1, 2, 3, or 5 (avoid option 4 - live preview)
```

#### 5. Low Frame Rate

**Problem:** FPS is very low (< 10 FPS)

**Solutions:**

```bash
# Increase GPU memory allocation
sudo raspi-config
# Navigate to: Advanced Options -> Memory Split
# Set to 256 MB or higher

# Reduce camera resolution
# Edit the script and add after line with cv2.VideoCapture:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

#### 6. Camera Not Releasing

**Problem:** Camera stays busy after script exit

**Solutions:**

```bash
# Kill all Python processes using camera
pkill -9 python3

# Or restart the video service
sudo systemctl restart video
```

---

## Camera Configuration

### Check Camera Capabilities

```bash
# List supported formats and resolutions
v4l2-ctl --list-formats-ext

# Get all camera settings
v4l2-ctl --all

# List available controls
v4l2-ctl --list-ctrls
```

### Adjust Camera Settings

```bash
# Set brightness (0-255)
v4l2-ctl --set-ctrl brightness=150

# Set contrast (0-255)
v4l2-ctl --set-ctrl contrast=128

# Set saturation (0-255)
v4l2-ctl --set-ctrl saturation=128

# Disable auto-exposure
v4l2-ctl --set-ctrl exposure_auto=1

# Set manual exposure
v4l2-ctl --set-ctrl exposure_absolute=100
```

### Optimize for AISENTINEL Project

For exam proctoring, recommended settings:

```bash
# Set resolution to 1280x720 for balance between quality and performance
# Set in script or via v4l2-ctl:
v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=MJPG

# Enable auto-focus (if supported)
v4l2-ctl --set-ctrl focus_auto=1

# Adjust for indoor lighting
v4l2-ctl --set-ctrl brightness=130
v4l2-ctl --set-ctrl contrast=100
```

---

## Advanced Configuration

### Using Multiple Cameras

```bash
# Test each camera separately
python3 camera_test.py 0  # First camera
python3 camera_test.py 1  # Second camera
```

### Performance Optimization

**1. Use GPU Acceleration**

```bash
# Enable OpenGL driver
sudo raspi-config
# Advanced Options -> GL Driver -> GL (Full KMS)
```

**2. Overclock Raspberry Pi** (optional, use with caution)

```bash
sudo raspi-config
# Performance Options -> Overclock
```

**3. Use MJPEG Format**

```python
# In the script, add after VideoCapture:
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
```

### Running as System Service

To run camera monitoring as a background service:

```bash
# Create service file
sudo nano /etc/systemd/system/camera-test.service
```

Add:

```ini
[Unit]
Description=USB Camera Test Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/AISENTINEL/camera_test.py
WorkingDirectory=/home/pi/AISENTINEL
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable camera-test.service
sudo systemctl start camera-test.service
sudo systemctl status camera-test.service
```

---

## Testing Checklist

Use this checklist to verify your camera setup:

- [ ] Camera physically connected to USB port
- [ ] Camera detected by `lsusb`
- [ ] `/dev/video0` (or similar) exists
- [ ] User is in `video` group
- [ ] OpenCV installed and importable
- [ ] Script lists camera successfully (option 1)
- [ ] Camera properties displayed (option 2)
- [ ] Test image captured successfully (option 3)
- [ ] Live preview works without errors (option 4)
- [ ] FPS is acceptable (>15 FPS for 720p)

---

## Output Files

The test script creates the following:

```
AISENTINEL/
├── camera_test.py           # Main test script
├── test_captures/           # Directory for captured images
│   ├── test_image_20260208_132301.jpg
│   └── test_image_20260208_132305.jpg
└── CAMERA_SETUP_GUIDE.md   # This guide
```

---

## Next Steps for AISENTINEL Project

After verifying camera functionality:

1. **Integrate with main application** - Import camera functions into your proctoring system
2. **Add face detection** - Implement face recognition using OpenCV or dlib
3. **Motion detection** - Add algorithms to detect suspicious movements
4. **Multi-camera support** - If using multiple angles for exam monitoring
5. **Recording capabilities** - Add video recording for exam sessions
6. **Edge AI integration** - Deploy TensorFlow Lite or PyTorch models for real-time analysis

---

## Additional Resources

- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [V4L2 Documentation](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html)
- [UVC Camera Protocol](https://www.usb.org/document-library/video-class-v11-document-set)

---

## Support

For issues specific to this script, check:

1. Raspberry Pi system logs: `sudo journalctl -xe`
2. Camera kernel messages: `dmesg | grep -i video`
3. USB device messages: `dmesg | grep -i usb`

**Common Log Commands:**

```bash
# Real-time system logs
sudo journalctl -f

# Camera-related logs
dmesg | grep -i "video\|usb" | tail -20

# Check USB bandwidth
lsusb -t
```
