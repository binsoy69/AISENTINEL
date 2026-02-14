# Back Node Setup Guide (Raspberry Pi 5)

This guide walks you through setting up the environment to run the `BackNodeTest.py` script on a Raspberry Pi 5. The Back Node uses the CPU for inference, leveraging NCNN for optimization.

---

## 1. Prerequisites

- **Hardware**: Raspberry Pi 5 (8GB RAM recommended) with active cooling.
- **OS**: Raspberry Pi OS (64-bit, Bookworm).
- **Camera**: USB Webcam connected.
- **Internet Access**: Required for installing packages.

---

## 2. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

### Step 2.1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2.2: Create Virtual Environment

```bash
# Install python3-venv if not already installed
sudo apt install python3-venv -y

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 2.3: Install Dependencies

Install the required Python libraries. Key libraries are `ultralytics` (for YOLO) and `ncnn` (for optimized inference).

```bash
# Update pip
pip install --upgrade pip

# Install Ultralytics (includes standard YOLO dependencies)
pip install ultralytics

# Install OpenCV (usually included, but ensure correct version)
pip install opencv-python-headless

# Install NCNN (optional, but Ultralytics AutoBackend uses it if exporting)
pip install ncnn
```

> **Note**: If you face issues with `opencv-python` on RPi, you might need to install system dependencies:
> `sudo apt install libgl1-mesa-glx`

---

## 3. Running the Test

Navigate to the `tests` folder where `BackNodeTest.py` is located.

### Option A: Standard PyTorch Mode (Easy, Slow)

This runs the standard `.pt` model on the CPU. It checks if everything works but will be slow (low FPS).

```bash
# Run with default settings (downloads yolo11n.pt automatically)
python BackNodeTest.py --source 0

# Run with a specific model file
python BackNodeTest.py --model ../models/yolo11n.pt --source 0
```

### Option B: NCNN Optimized Mode (Recommended, Fast)

This converts the model to NCNN format, which is highly optimized for ARM CPUs like the RPi 5.

**1. First Run (Export & Test):**
Use the `--export-ncnn` flag. This will:

1. Load `yolo11n.pt`.
2. Export it to a directory named `yolo11n_ncnn_model`.
3. Automatically reload and run the NCNN version.

```bash
python BackNodeTest.py --model yolo11n.pt --export-ncnn --source 0
```

**2. Subsequent Runs:**
Once exported, point directly to the NCNN model directory to skip the export step.

```bash
python BackNodeTest.py --model yolo11n_ncnn_model --source 0
```

---

## 4. Troubleshooting

- **Low FPS**: Ensure you are using the NCNN model (`Option B`). Standard PyTorch on CPU is significantly slower.
- **Camera Not Found**:
  - Check connections.
  - Try changing `--source 0` to `--source 1` or `--source 2`.
  - Verify with `vcgill-ctl` or `ls /dev/video*`.
- **"Illegal Instruction" or Crash**:
  - Ensure you are running a 64-bit OS (`uname -m` should say `aarch64`).
  - Update `ncnn` and `ultralytics`.

---

## 5. Next Steps

Once the test confirms the camera and inference are working:

1.  **Calibrate Zones**: You will need to customize the logic to handle "hands under table" detection using the fixed camera position.
2.  **Integrate Logic**: Move from this simple visualization to the `behavior_back.py` logic defined in `PROJECT.md`.
