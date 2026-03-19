#!/usr/bin/env python3
"""
Standalone Hailo AI Hat Object Detection Test Script
Uses only HailoRT (pyhailort) and OpenCV - NO hailo-apps dependency

This script is designed to run on Raspberry Pi 5 with Hailo AI Hat.

Prerequisites (install via apt on Raspberry Pi OS):
    sudo apt update
    sudo apt install hailo-all python3-opencv

    # Or install Python packages:
    pip install hailort opencv-python numpy

Usage:
    python hailo_detection_test.py                          # USB camera (default)
    python hailo_detection_test.py --input /dev/video0      # Specific camera
    python hailo_detection_test.py --input video.mp4        # Video file
    python hailo_detection_test.py --hef /path/to/model.hef # Custom model
"""

import sys
import os
import time
import argparse
from datetime import datetime

# Check prerequisites before importing heavy libraries
HAILO_MODULE = None  # Will be set after successful import

def check_prerequisites():
    """Check if required packages are available"""
    global HAILO_MODULE
    errors = []
    
    try:
        import cv2
    except ImportError:
        errors.append("OpenCV not installed. Run: pip install opencv-python")
    
    try:
        import numpy
    except ImportError:
        errors.append("NumPy not installed. Run: pip install numpy")
    
    # Try multiple possible import names for HailoRT
    hailo_imported = False
    import_attempts = []
    
    # Attempt 1: hailo_platform (standard name)
    try:
        from hailo_platform import HEF, VDevice, ConfigureParams, HailoSchedulingAlgorithm, FormatType
        HAILO_MODULE = "hailo_platform"
        hailo_imported = True
    except ImportError as e:
        import_attempts.append(f"hailo_platform: {e}")
    
    # Attempt 2: hailort
    if not hailo_imported:
        try:
            from hailort import HEF, VDevice, ConfigureParams, HailoSchedulingAlgorithm, FormatType
            HAILO_MODULE = "hailort"
            hailo_imported = True
        except ImportError as e:
            import_attempts.append(f"hailort: {e}")
    
    # Attempt 3: hailo (older versions)
    if not hailo_imported:
        try:
            import hailo
            HAILO_MODULE = "hailo"
            hailo_imported = True
        except ImportError as e:
            import_attempts.append(f"hailo: {e}")
    
    # Attempt 4: pyhailort
    if not hailo_imported:
        try:
            import pyhailort
            HAILO_MODULE = "pyhailort"
            hailo_imported = True
        except ImportError as e:
            import_attempts.append(f"pyhailort: {e}")
    
    if not hailo_imported:
        errors.append("HailoRT Python bindings not found")
        errors.append("Tried: " + ", ".join([a.split(":")[0] for a in import_attempts]))
        errors.append("Run --diagnose for detailed import error info")
    
    return errors


def run_diagnostics():
    """Run detailed diagnostics for HailoRT installation"""
    print("=" * 60)
    print("Hailo Installation Diagnostics")
    print("=" * 60)
    
    # Check system info
    import platform
    print(f"\n[System]")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Python path: {sys.executable}")
    
    # Check for Hailo device
    print(f"\n[Hailo Device]")
    if os.path.exists("/dev/hailo0"):
        print("  ✓ /dev/hailo0 found")
    else:
        print("  ✗ /dev/hailo0 not found - check if AI Hat is connected")
    
    # Try each import and show detailed errors
    print(f"\n[Python Module Imports]")
    modules_to_try = [
        ("hailo_platform", "from hailo_platform import HEF"),
        ("hailort", "from hailort import HEF"),
        ("hailo", "import hailo"),
        ("pyhailort", "import pyhailort"),
    ]
    
    for module_name, import_stmt in modules_to_try:
        try:
            exec(import_stmt)
            print(f"  ✓ {module_name}: OK")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
        except Exception as e:
            print(f"  ✗ {module_name}: {type(e).__name__}: {e}")
    
    # Check pip packages
    print(f"\n[Installed pip packages with 'hailo']")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "hailo" in line.lower():
                print(f"  {line}")
    except Exception as e:
        print(f"  Error checking pip: {e}")
    
    # Check system packages
    print(f"\n[System packages]")
    try:
        import subprocess
        result = subprocess.run(["dpkg", "-l"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "hailo" in line.lower():
                parts = line.split()
                if len(parts) >= 3:
                    print(f"  {parts[1]}: {parts[2]}")
    except Exception as e:
        print(f"  Error checking dpkg: {e}")
    
    print("\n" + "=" * 60)
    print("Suggested fixes:")
    print("=" * 60)
    print("1. Make sure you're using the system Python (not a venv):")
    print("   python3 hailo_detection_test.py")
    print("\n2. Or install the Python wheel in your environment:")
    print("   pip install /usr/share/hailo/*.whl")
    print("\n3. Check Hailo documentation:")
    print("   https://github.com/hailo-ai/hailo-rpi5-examples")
    print("=" * 60)
    return 0


def print_error_and_exit(errors):
    """Print helpful error message"""
    print("\n" + "=" * 60)
    print("ERROR: Missing dependencies")
    print("=" * 60)
    for error in errors:
        print(f"  ✗ {error}")
    print("\nTry running with --diagnose for detailed info:")
    print("  python hailo_detection_test.py --diagnose")
    print("\nOr try using system Python directly:")
    print("  python3 hailo_detection_test.py")
    print("=" * 60)
    sys.exit(1)


# Handle --diagnose flag early
if "--diagnose" in sys.argv:
    run_diagnostics()
    sys.exit(0)

# Check prerequisites
errors = check_prerequisites()
if errors:
    print_error_and_exit(errors)


# Now import based on what we found
import cv2
import numpy as np

if HAILO_MODULE == "hailo_platform":
    from hailo_platform import HEF, VDevice, ConfigureParams, HailoSchedulingAlgorithm, FormatType
elif HAILO_MODULE == "hailort":
    from hailort import HEF, VDevice, ConfigureParams, HailoSchedulingAlgorithm, FormatType
elif HAILO_MODULE == "hailo":
    import hailo
    # Older API might have different structure
    HEF = hailo.HEF if hasattr(hailo, 'HEF') else None
    VDevice = hailo.VDevice if hasattr(hailo, 'VDevice') else None
elif HAILO_MODULE == "pyhailort":
    import pyhailort
    HEF = pyhailort.HEF if hasattr(pyhailort, 'HEF') else None
    VDevice = pyhailort.VDevice if hasattr(pyhailort, 'VDevice') else None


# COCO class labels (80 classes for YOLO models)
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class HailoDetector:
    """
    Standalone Hailo object detector using HailoRT directly.
    No dependency on hailo-apps.
    """
    
    def __init__(self, hef_path, confidence_threshold=0.5, labels=None):
        """
        Initialize the Hailo detector.
        
        Args:
            hef_path: Path to the HEF model file
            confidence_threshold: Minimum confidence for detections
            labels: List of class labels (defaults to COCO)
        """
        self.confidence_threshold = confidence_threshold
        self.labels = labels or COCO_LABELS
        
        print(f"[INFO] Loading model: {hef_path}")
        
        # Load HEF file
        self.hef = HEF(hef_path)
        
        # Create virtual device
        self.vdevice = VDevice()
        
        # Configure the device
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoSchedulingAlgorithm.ROUND_ROBIN
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        
        # Get input/output info
        self.input_vstreams_info = self.hef.get_input_vstream_infos()
        self.output_vstreams_info = self.hef.get_output_vstream_infos()
        
        # Get input shape
        input_info = self.input_vstreams_info[0]
        self.input_shape = input_info.shape
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]
        
        print(f"[INFO] Model input shape: {self.input_shape}")
        print(f"[INFO] Model loaded successfully")
    
    def preprocess(self, frame):
        """
        Preprocess frame for inference.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Preprocessed frame ready for inference
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def infer(self, preprocessed_frame):
        """
        Run inference on a preprocessed frame.
        
        Args:
            preprocessed_frame: Preprocessed image
            
        Returns:
            Raw inference output
        """
        # Create input/output params
        input_params = self.network_group.make_input_vstream_params(
            quantized=False,
            format_type=FormatType.FLOAT32
        )
        output_params = self.network_group.make_output_vstream_params(
            quantized=False,
            format_type=FormatType.FLOAT32
        )
        
        # Run inference
        with self.network_group.activate():
            input_data = {self.input_vstreams_info[0].name: 
                         np.expand_dims(preprocessed_frame.astype(np.float32) / 255.0, axis=0)}
            
            with self.network_group.create_vstreams(input_params, output_params) as (input_vstreams, output_vstreams):
                # Send input
                for name, data in input_data.items():
                    input_vstreams[name].send(data)
                
                # Get output
                outputs = {}
                for output_vstream in output_vstreams:
                    outputs[output_vstream.name] = output_vstream.recv()
        
        return outputs
    
    def postprocess(self, outputs, original_shape):
        """
        Parse detection outputs.
        
        Args:
            outputs: Raw inference outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detections: [(label, confidence, x1, y1, x2, y2), ...]
        """
        detections = []
        orig_h, orig_w = original_shape[:2]
        
        # Handle different output formats
        # This is a simplified parser - actual format depends on the model
        for name, output in outputs.items():
            # Flatten and reshape as needed
            output = np.squeeze(output)
            
            # Try to parse as YOLO-style output
            # Format varies by model - this handles common cases
            if len(output.shape) == 2:
                # Each row: [x, y, w, h, conf, class_scores...]
                for detection in output:
                    if len(detection) >= 6:
                        # Get confidence
                        conf = detection[4] if len(detection) > 5 else max(detection[5:])
                        
                        if conf < self.confidence_threshold:
                            continue
                        
                        # Get class
                        class_scores = detection[5:] if len(detection) > 5 else detection[4:]
                        class_id = np.argmax(class_scores)
                        
                        # Get bounding box (normalized coordinates)
                        x, y, w, h = detection[:4]
                        
                        # Convert to pixel coordinates
                        x1 = int((x - w/2) * orig_w)
                        y1 = int((y - h/2) * orig_h)
                        x2 = int((x + w/2) * orig_w)
                        y2 = int((y + h/2) * orig_h)
                        
                        # Clip to image bounds
                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))
                        
                        label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                        detections.append((label, float(conf), x1, y1, x2, y2))
        
        return detections
    
    def detect(self, frame):
        """
        Run full detection pipeline on a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detections: [(label, confidence, x1, y1, x2, y2), ...]
        """
        preprocessed = self.preprocess(frame)
        outputs = self.infer(preprocessed)
        detections = self.postprocess(outputs, frame.shape)
        return detections
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'vdevice'):
            self.vdevice.release()


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for label, conf, x1, y1, x2, y2 in detections:
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_text = f"{label}: {conf:.2%}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def find_default_hef():
    """Try to find a default HEF model file"""
    search_paths = [
        "/usr/share/hailo-models",
        "/usr/local/share/hailo-models",
        os.path.expanduser("~/hailo-models"),
        os.path.expanduser("~/hailo-apps/resources/models"),
        os.path.expanduser("~/hailo-rpi5-examples/resources"),
        "/usr/local/hailo/resources",
    ]
    
    # Common YOLO model names
    model_names = [
        "yolov8s.hef",
        "yolov8n.hef",
        "yolov6n.hef",
        "yolov5s.hef",
        "yolov5n.hef",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for model in model_names:
                model_path = os.path.join(path, model)
                if os.path.exists(model_path):
                    return model_path
            # Search recursively
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.hef') and 'yolo' in file.lower():
                        return os.path.join(root, file)
    
    return None


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Standalone Hailo Object Detection Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hailo_detection_test.py                          # USB camera
  python hailo_detection_test.py --input /dev/video0      # Specific camera
  python hailo_detection_test.py --input video.mp4        # Video file
  python hailo_detection_test.py --hef yolov8s.hef        # Custom model
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="0",
        help="Input source: camera index (0), device path (/dev/video0), or video file"
    )
    
    parser.add_argument(
        "--hef", "-m",
        type=str,
        default=None,
        help="Path to HEF model file (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--filter", "-f",
        type=str,
        nargs="+",
        help="Filter specific classes (e.g., --filter person car)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (headless mode)"
    )
    
    parser.add_argument(
        "--log",
        type=str,
        help="Log detections to file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Hailo AI Hat Object Detection Test")
    print("AISENTINEL Project - Standalone Version")
    print("=" * 60)
    
    args = parse_args()
    
    # Find or validate HEF model
    hef_path = args.hef
    if hef_path is None:
        hef_path = find_default_hef()
        if hef_path is None:
            print("\n[ERROR] No HEF model found!")
            print("Please specify a model using --hef /path/to/model.hef")
            print("\nTo download models, run:")
            print("  hailo-download-resources --all")
            return 1
    
    if not os.path.exists(hef_path):
        print(f"\n[ERROR] Model file not found: {hef_path}")
        return 1
    
    print(f"\n[CONFIG]")
    print(f"  Model: {hef_path}")
    print(f"  Input: {args.input}")
    print(f"  Confidence: {args.confidence}")
    if args.filter:
        print(f"  Filter: {', '.join(args.filter)}")
    
    # Initialize detector
    try:
        detector = HailoDetector(hef_path, args.confidence)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize detector: {e}")
        print("\nMake sure:")
        print("  1. Hailo AI Hat is properly connected")
        print("  2. HailoRT is installed: sudo apt install hailo-all")
        return 1
    
    # Open video source
    if args.input.isdigit():
        source = int(args.input)
    else:
        source = args.input
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video source: {args.input}")
        print("\nCheck available cameras: ls /dev/video*")
        detector.close()
        return 1
    
    print(f"\n[INFO] Video source opened successfully")
    print("-" * 60)
    print("Running detection... Press 'q' to quit, 's' to save frame")
    print("-" * 60)
    
    # Set up logging
    log_file = None
    if args.log:
        log_file = open(args.log, 'w')
        log_file.write(f"Hailo Detection Log - {datetime.now()}\n")
        log_file.write("=" * 60 + "\n")
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.startswith('/dev'):
                    print("\n[INFO] End of video file")
                    break
                continue
            
            frame_count += 1
            
            # Run detection
            detections = detector.detect(frame)
            
            # Filter classes if specified
            if args.filter:
                detections = [d for d in detections if d[0] in args.filter]
            
            total_detections += len(detections)
            
            # Log detections
            if detections and log_file:
                for label, conf, x1, y1, x2, y2 in detections:
                    log_file.write(f"Frame {frame_count}: {label} ({conf:.2%})\n")
            
            # Print to console every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[Frame {frame_count}] FPS: {fps:.1f}, Detections this frame: {len(detections)}")
            
            # Draw and display
            if not args.no_display:
                frame = draw_detections(frame, detections)
                
                # Add FPS overlay
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Hailo Detection Test", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit requested")
                    break
                elif key == ord('s'):
                    filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Detection Test Summary")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average FPS: {frame_count / elapsed:.1f}" if elapsed > 0 else "  Duration: 0s")
    print(f"  Duration: {elapsed:.1f} seconds")
    if args.log:
        print(f"  Log saved to: {args.log}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
