#!/usr/bin/env python3
"""
USB Camera Test Script for Raspberry Pi
Tests camera connectivity, captures images, and displays video feed
"""

import cv2
import os
import sys
from datetime import datetime
import time

class CameraTest:
    def __init__(self, camera_index=0):
        """Initialize camera test with specified camera index"""
        self.camera_index = camera_index
        self.cap = None
        
    def list_available_cameras(self, max_cameras=10):
        """Detect and list all available cameras"""
        print("\n" + "="*50)
        print("Scanning for available cameras...")
        print("="*50)
        
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    print(f"✓ Camera {i} detected - Resolution: {width}x{height}, FPS: {fps}")
                cap.release()
        
        if not available_cameras:
            print("✗ No cameras detected!")
        else:
            print(f"\nTotal cameras found: {len(available_cameras)}")
        
        return available_cameras
    
    def connect_camera(self):
        """Connect to the camera"""
        print(f"\nConnecting to camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"✗ Failed to open camera {self.camera_index}")
            return False
        
        print(f"✓ Camera {self.camera_index} connected successfully")
        return True
    
    def display_camera_info(self):
        """Display detailed camera properties"""
        if not self.cap or not self.cap.isOpened():
            print("✗ Camera not connected")
            return
        
        print("\n" + "="*50)
        print("Camera Properties")
        print("="*50)
        
        properties = {
            "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
            "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
            "FPS": cv2.CAP_PROP_FPS,
            "Brightness": cv2.CAP_PROP_BRIGHTNESS,
            "Contrast": cv2.CAP_PROP_CONTRAST,
            "Saturation": cv2.CAP_PROP_SATURATION,
            "Hue": cv2.CAP_PROP_HUE,
            "Gain": cv2.CAP_PROP_GAIN,
            "Exposure": cv2.CAP_PROP_EXPOSURE,
        }
        
        for prop_name, prop_id in properties.items():
            value = self.cap.get(prop_id)
            print(f"{prop_name:20s}: {value}")
    
    def capture_test_image(self, output_dir="test_captures"):
        """Capture and save a test image"""
        if not self.cap or not self.cap.isOpened():
            print("✗ Camera not connected")
            return False
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ret, frame = self.cap.read()
        if not ret:
            print("✗ Failed to capture frame")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"test_image_{timestamp}.jpg")
        
        cv2.imwrite(filename, frame)
        print(f"\n✓ Test image saved: {filename}")
        return True
    
    def live_preview(self, duration=None):
        """Display live camera feed"""
        if not self.cap or not self.cap.isOpened():
            print("✗ Camera not connected")
            return False
        
        print("\n" + "="*50)
        print("Live Camera Preview")
        print("="*50)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save snapshot")
        print("  - Press 'i' to show camera info")
        print("="*50)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ Failed to read frame")
                    break
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Add overlay information
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Camera: {self.camera_index}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('USB Camera Test - Live Feed', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting live preview...")
                    break
                elif key == ord('s'):
                    self.capture_test_image()
                elif key == ord('i'):
                    self.display_camera_info()
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\nDuration limit ({duration}s) reached")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
        
        return True
    
    def run_benchmark(self, duration=10):
        """Run a performance benchmark"""
        if not self.cap or not self.cap.isOpened():
            print("✗ Camera not connected")
            return False
        
        print("\n" + "="*50)
        print(f"Running {duration}-second performance benchmark...")
        print("="*50)
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration:
            ret, frame = self.cap.read()
            if ret:
                frame_count += 1
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time
        
        print(f"\nBenchmark Results:")
        print(f"  Frames captured: {frame_count}")
        print(f"  Duration: {elapsed_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        
        return True
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            print("\n✓ Camera released")
        cv2.destroyAllWindows()

def print_menu():
    """Display test menu"""
    print("\n" + "="*50)
    print("USB Camera Test Menu")
    print("="*50)
    print("1. List available cameras")
    print("2. Display camera information")
    print("3. Capture test image")
    print("4. Live preview")
    print("5. Run performance benchmark")
    print("6. Run all tests")
    print("0. Exit")
    print("="*50)

def main():
    """Main function"""
    print("="*50)
    print("USB Camera Test for Raspberry Pi")
    print("="*50)
    
    # Check if camera index is provided
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}")
            print("Using default camera index: 0")
    
    camera_test = CameraTest(camera_index)
    
    # Interactive mode
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '1':
            camera_test.list_available_cameras()
            
        elif choice == '2':
            if not camera_test.cap or not camera_test.cap.isOpened():
                if not camera_test.connect_camera():
                    continue
            camera_test.display_camera_info()
            
        elif choice == '3':
            if not camera_test.cap or not camera_test.cap.isOpened():
                if not camera_test.connect_camera():
                    continue
            camera_test.capture_test_image()
            
        elif choice == '4':
            if not camera_test.cap or not camera_test.cap.isOpened():
                if not camera_test.connect_camera():
                    continue
            camera_test.live_preview()
            
        elif choice == '5':
            if not camera_test.cap or not camera_test.cap.isOpened():
                if not camera_test.connect_camera():
                    continue
            camera_test.run_benchmark()
            
        elif choice == '6':
            print("\nRunning all tests...")
            camera_test.list_available_cameras()
            if camera_test.connect_camera():
                camera_test.display_camera_info()
                camera_test.capture_test_image()
                camera_test.run_benchmark(duration=5)
                camera_test.live_preview()
            
        elif choice == '0':
            print("\nExiting...")
            break
            
        else:
            print("\n✗ Invalid choice. Please try again.")
    
    camera_test.release()
    print("\nThank you for using USB Camera Test!")

if __name__ == "__main__":
    main()
