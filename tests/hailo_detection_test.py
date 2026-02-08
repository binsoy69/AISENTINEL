#!/usr/bin/env python3
"""
Hailo AI Hat Object Detection Test Script for Raspberry Pi 5
Tests object detection using Hailo accelerator with USB camera input

This script is designed to run on Raspberry Pi 5 with Hailo AI Hat installed.
It will NOT work on Windows or systems without Hailo hardware.

Prerequisites:
    1. Hailo AI Hat connected to Raspberry Pi 5
    2. hailo-apps repository installed (https://github.com/hailo-ai/hailo-apps)
    3. Environment sourced: source setup_env.sh

Usage:
    python hailo_detection_test.py                    # Use USB camera (default)
    python hailo_detection_test.py --input usb        # Use USB camera
    python hailo_detection_test.py --input /dev/video0  # Use specific device
    python hailo_detection_test.py --input video.mp4  # Use video file
    python hailo_detection_test.py --help             # Show all options
"""

import sys
import os
import argparse
from datetime import datetime

# Check if running on compatible system
def check_prerequisites():
    """Check if Hailo dependencies are available"""
    errors = []
    
    # Check for GStreamer
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
    except (ImportError, ValueError) as e:
        errors.append(f"GStreamer not available: {e}")
    
    # Check for Hailo runtime
    try:
        import hailo
    except ImportError:
        errors.append("Hailo runtime not installed. Please install hailo-apps package.")
    
    # Check for hailo_apps
    try:
        from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    except ImportError:
        errors.append("hailo_apps package not found. Please clone and install hailo-apps repository.")
    
    return errors


def print_banner():
    """Print script banner"""
    print("=" * 60)
    print("Hailo AI Hat Object Detection Test")
    print("AISENTINEL Project")
    print("=" * 60)


def print_prerequisites_error(errors):
    """Print helpful error message for missing prerequisites"""
    print("\n" + "=" * 60)
    print("ERROR: Prerequisites not met")
    print("=" * 60)
    print("\nThis script requires Hailo AI Hat and hailo-apps package.")
    print("\nMissing dependencies:")
    for error in errors:
        print(f"  ✗ {error}")
    
    print("\nTo set up on Raspberry Pi 5:")
    print("  1. Install Hailo AI Hat hardware")
    print("  2. Clone hailo-apps repository:")
    print("     git clone https://github.com/hailo-ai/hailo-apps.git")
    print("  3. Install dependencies:")
    print("     cd hailo-apps && sudo ./install.sh")
    print("  4. Source environment before running:")
    print("     source setup_env.sh")
    print("\nFor more info: https://github.com/hailo-ai/hailo-apps")
    print("=" * 60)


# Only import Hailo dependencies if prerequisites are met
prerequisites_errors = check_prerequisites()
if prerequisites_errors:
    # Define minimal argument parser for help message
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    print_prerequisites_error(prerequisites_errors)
    sys.exit(1)

# Import Hailo dependencies (only reached if prerequisites pass)
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

# Try to import the simple detection pipeline first (lightweight)
try:
    from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp
    DETECTION_TYPE = "simple"
except ImportError:
    # Fall back to full detection pipeline
    from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
    DETECTION_TYPE = "full"


class DetectionTestCallback(app_callback_class):
    """
    Custom callback class for object detection testing.
    Extends app_callback_class to track detections and performance.
    """
    
    def __init__(self, log_file=None, target_classes=None):
        super().__init__()
        self.detection_count = 0
        self.frame_start_time = datetime.now()
        self.log_file = log_file
        self.target_classes = target_classes  # Filter specific classes (e.g., ['person'])
        
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(f"Hailo Detection Test Log - {self.frame_start_time}\n")
                f.write("=" * 60 + "\n")
    
    def log_detection(self, frame_num, label, confidence, bbox=None):
        """Log a detection to file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = f"[{timestamp}] Frame {frame_num}: {label} ({confidence:.2%})"
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + "\n")
        
        return msg
    
    def get_elapsed_time(self):
        """Get elapsed time since start"""
        return (datetime.now() - self.frame_start_time).total_seconds()
    
    def get_fps(self):
        """Calculate current FPS"""
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            return self.get_count() / elapsed
        return 0


def detection_callback(pad, info, user_data):
    """
    Callback function called for each frame processed by the pipeline.
    Parses detections from the GStreamer buffer and prints results.
    
    Args:
        pad: GStreamer pad
        info: Probe info containing the buffer
        user_data: DetectionTestCallback instance
    """
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Increment frame counter
    user_data.increment()
    frame_num = user_data.get_count()
    
    # Get detections from buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Build output string
    output_lines = []
    
    # Add FPS info every 30 frames
    if frame_num % 30 == 0:
        fps = user_data.get_fps()
        output_lines.append(f"\n[Frame {frame_num}] FPS: {fps:.1f}")
    
    # Parse each detection
    frame_detections = 0
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        
        # Filter by target classes if specified
        if user_data.target_classes and label not in user_data.target_classes:
            continue
        
        # Get bounding box
        bbox = detection.get_bbox()
        
        # Log detection
        msg = user_data.log_detection(frame_num, label, confidence, bbox)
        output_lines.append(f"  ✓ {label}: {confidence:.2%}")
        
        frame_detections += 1
        user_data.detection_count += 1
    
    # Print output if there are detections
    if output_lines:
        print("\n".join(output_lines))
    
    return Gst.PadProbeReturn.OK


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Hailo AI Hat Object Detection Test for AISENTINEL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hailo_detection_test.py                     # USB camera (default)
  python hailo_detection_test.py --input usb         # USB camera auto-detect
  python hailo_detection_test.py --input /dev/video0 # Specific device
  python hailo_detection_test.py --input video.mp4   # Video file
  python hailo_detection_test.py --filter person     # Only detect persons
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="usb",
        help="Input source: 'usb' (default), 'rpi', '/dev/videoX', or video file path"
    )
    
    parser.add_argument(
        "--filter", "-f",
        type=str,
        nargs="+",
        help="Filter specific object classes (e.g., --filter person car)"
    )
    
    parser.add_argument(
        "--log", "-l",
        type=str,
        help="Log detections to specified file"
    )
    
    parser.add_argument(
        "--use-frame",
        action="store_true",
        help="Enable frame processing (higher CPU usage)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Input source: {args.input}")
    print(f"  Detection type: {DETECTION_TYPE}")
    if args.filter:
        print(f"  Filter classes: {', '.join(args.filter)}")
    if args.log:
        print(f"  Log file: {args.log}")
    
    print("\n" + "-" * 60)
    print("Starting object detection pipeline...")
    print("Press Ctrl+C to stop")
    print("-" * 60 + "\n")
    
    # Set up environment
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)
    
    # Create callback instance
    user_data = DetectionTestCallback(
        log_file=args.log,
        target_classes=args.filter
    )
    
    if args.use_frame:
        user_data.use_frame = True
    
    try:
        # Create and run detection app
        # Pass input argument to the app
        sys.argv = ['hailo_detection_test.py', '--input', args.input]
        
        app = GStreamerDetectionApp(detection_callback, user_data)
        app.run()
        
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
    except Exception as e:
        print(f"\nError running detection: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Hailo AI Hat is properly connected")
        print("  2. Check if camera is available: ls /dev/video*")
        print("  3. Verify hailo-apps environment: source setup_env.sh")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Detection Test Summary")
    print("=" * 60)
    print(f"  Total frames processed: {user_data.get_count()}")
    print(f"  Total detections: {user_data.detection_count}")
    print(f"  Average FPS: {user_data.get_fps():.1f}")
    print(f"  Duration: {user_data.get_elapsed_time():.1f} seconds")
    if args.log:
        print(f"  Log saved to: {args.log}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
