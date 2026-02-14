#!/usr/bin/env python3
"""
Export YOLOv11n-pose model to ONNX format for Hailo conversion.

This script:
1. Downloads the pretrained YOLOv11n-pose model (if not present)
2. Exports it to ONNX format
3. Prepares it for Hailo Dataflow Compiler conversion to .hef
"""

from ultralytics import YOLO
import os

print("="*60)
print("YOLOv11n-pose Model Export to ONNX")
print("="*60)

# Model name (correct naming: yolo11n-pose, not yolov11n-pose)
model_name = 'yolo11n-pose.pt'

print(f"\n[1/3] Loading model: {model_name}")
print("      (Will download if not present...)")

try:
    # Load the pretrained YOLOv11n-pose model
    # Ultralytics will auto-download if the model doesn't exist
    model = YOLO(model_name)
    print(f"✓ Model loaded successfully")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection (model needs to download)")
    print("2. Try: pip install --upgrade ultralytics")
    print("3. Manually download from: https://github.com/ultralytics/assets/releases")
    exit(1)

print(f"\n[2/3] Exporting to ONNX format...")
print("      Input size: 640x640")
print("      Simplify: True")
print("      Opset: 11")

try:
    # Export to ONNX format
    result = model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=11,
    )

    onnx_file = result if isinstance(result, str) else f"{model_name.replace('.pt', '.onnx')}"

    print(f"✓ Export successful!")

except Exception as e:
    print(f"✗ Export failed: {e}")
    print("\nTroubleshooting:")
    print("1. Install ONNX: pip install onnx onnxsim")
    print("2. Update ultralytics: pip install --upgrade ultralytics")
    exit(1)

print(f"\n[3/3] Verifying output file...")

if os.path.exists(onnx_file):
    file_size = os.path.getsize(onnx_file) / (1024 * 1024)  # MB
    print(f"✓ ONNX file created: {onnx_file}")
    print(f"  Size: {file_size:.2f} MB")
else:
    print(f"✗ ONNX file not found: {onnx_file}")
    exit(1)

print("\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)
print(f"\nNext steps:")
print(f"1. Transfer {onnx_file} to your Hailo development machine")
print(f"2. Convert to HEF using Hailo Dataflow Compiler:")
print(f"   hailomz compile {onnx_file} --hw-arch hailo8l --output yolov11n-pose.hef")
print(f"3. Transfer the .hef file to your Raspberry Pi")
print(f"4. Test with: python3 pose_usb_cam_hailo.py --model yolov11n-pose.hef")
print("="*60)