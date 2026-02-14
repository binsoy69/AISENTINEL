#!/usr/bin/env python3
"""
YOLO Pose Estimation on Raspberry Pi 5 (Trixie OS) + Hailo AI Accelerator
Using a USB Camera with Behavioral Analysis

Live preview via web browser at http://<pi-ip>:8080

Detects 17 COCO keypoints and calculates behavioral metrics:
- Head tilt angle (ear-to-ear line)
- Head turn direction (nose offset from shoulders)
- Sustained behavior detection with configurable thresholds

Usage:
    # CPU mode (immediate testing, auto-downloads model)
    python3 pose_usb_cam_hailo.py --cpu --port 8080

    # Hailo accelerated mode (requires .hef model)
    python3 pose_usb_cam_hailo.py --model yolov8s_pose.hef --port 8080

    # With behavioral analysis logging
    python3 pose_usb_cam_hailo.py --model yolov8s_pose.hef --log behaviors.log --save-frames

Requirements:
    - Raspberry Pi 5 with Trixie OS
    - Hailo AI Kit / AI HAT+ (for Hailo mode)
    - USB camera
    - Python packages: opencv-python, numpy, flask, ultralytics
    - hailo_platform (system-installed, for Hailo mode)
"""

import argparse
import time
import sys
import os
import threading
import socket
from datetime import datetime

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Try to import Flask for web preview
# ──────────────────────────────────────────────────────────────
try:
    from flask import Flask, Response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[WARN] flask not found. Install with: pip install flask")
    print("       Web preview will be disabled.\n")

# ──────────────────────────────────────────────────────────────
# Try to import hailo_platform
# ──────────────────────────────────────────────────────────────
try:
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[WARN] hailo_platform not found — will fall back to CPU-only mode (ultralytics).")

# ──────────────────────────────────────────────────────────────
# COCO Pose Keypoint Names (17 keypoints)
# ──────────────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO skeleton connections (19 connections)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
    (0, 5), (0, 6),      # nose to shoulders
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # shoulders to hips
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Global for web streaming
_latest_frame = None
_frame_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────
# YOLOv11-pose post-processing helpers
# ──────────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ──────────────────────────────────────────────────────────────
# Hailo Pose Estimator
# ──────────────────────────────────────────────────────────────

class HailoPoseEstimator:
    """Wraps HailoRT Python API for YOLOv8/v11-pose inference on the Hailo NPU."""

    def __init__(self, hef_path, conf_threshold=0.5, kpt_threshold=0.3, iou_threshold=0.45):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.iou_threshold = iou_threshold

        print(f"[INFO] Loading HEF model: {hef_path}")
        self.hef = HEF(hef_path)

        self.vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self.input_shape = self.input_vstream_info[0].shape
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        self.has_nms = any("nms" in info.name.lower() for info in self.output_vstream_info)

        print(f"[INFO] Model input shape : {self.input_shape}")
        for out_info in self.output_vstream_info:
            print(f"[INFO] Model output layer: {out_info.name} -> {out_info.shape}")
        print(f"[INFO] NMS on-chip       : {'YES' if self.has_nms else 'NO (Python NMS)'}")
        print(f"[INFO] Hailo device ready.")

    def detect_pose(self, frame):
        """Run pose estimation on a BGR frame.

        Returns:
            List of pose detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'keypoints': np.array(shape=(17, 3)),  # [x, y, conf]
                },
                ...
            ]
        """
        img_h, img_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        input_dict = {self.input_vstream_info[0].name: input_data}

        with self.network_group.activate(self.network_group_params):
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                results = infer_pipeline.infer(input_dict)

        # Debug: print output shapes on first inference
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            if isinstance(results, dict):
                for name, arr in results.items():
                    print(f"[DEBUG] Output '{name}': shape={np.array(arr).shape}, dtype={np.array(arr).dtype}")
            else:
                print(f"[DEBUG] Output: shape={np.array(results).shape}")

        return self._postprocess_pose(results, img_w, img_h)

    def _postprocess_pose(self, raw_output, img_w, img_h):
        """Parse YOLOv8/v11-pose output format.

        Output: (1, 56, 8400) or (8400, 56) or multiple tensors per scale.
        56 = [x, y, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, ...] (4 + 1 + 17*3)
        """
        # Extract and concatenate output tensors
        if isinstance(raw_output, dict):
            arrays = list(raw_output.values())
            if len(arrays) == 1:
                output = arrays[0]
            else:
                # Multi-scale outputs: concatenate along detection axis
                flat = []
                for a in arrays:
                    a = np.squeeze(a)
                    if a.ndim == 3:
                        a = a.reshape(-1, a.shape[-1])
                    elif a.ndim == 1:
                        continue
                    flat.append(a)
                output = np.concatenate(flat, axis=0)
        else:
            output = raw_output

        # Remove batch dimension
        output = np.squeeze(output)

        # Handle various shapes — target is (N, 56)
        if output.ndim == 2:
            if output.shape[0] == 56 and output.shape[1] > 56:
                output = output.T
            elif output.shape[1] < output.shape[0] and output.shape[1] != 56:
                # Columns might be features, rows might be detections — check
                if output.shape[0] in (56,):
                    output = output.T
        elif output.ndim == 3:
            # (batch, channels, detections) or similar
            output = output.reshape(-1, output.shape[-1])
            if output.shape[0] == 56 and output.shape[1] > 56:
                output = output.T

        # Validate we have 56 feature columns (4 bbox + 1 conf + 51 keypoints)
        # YOLOv8 pose may omit the conf column (56 = 4 + 1 + 51 or 55 = 4 + 51)
        num_cols = output.shape[1]
        has_conf_col = True
        if num_cols == 55:
            # No explicit conf column — derive from class scores
            has_conf_col = False
        elif num_cols != 56:
            # Try transposing as last resort
            if output.shape[0] in (55, 56):
                output = output.T
                num_cols = output.shape[1]
                has_conf_col = num_cols == 56

        # Extract components based on column count
        boxes_xywh = output[:, :4]       # [x, y, w, h]
        if has_conf_col:
            confidences = output[:, 4]       # person confidence
            keypoints_raw = output[:, 5:]    # 17 × 3 = 51 values
        else:
            # No conf column — use max keypoint confidence as proxy
            keypoints_raw = output[:, 4:]    # 17 × 3 = 51 values
            kpt_confs = keypoints_raw[:, 2::3]  # every 3rd value is conf
            confidences = np.mean(kpt_confs, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        keypoints_raw = keypoints_raw[mask]

        # Convert boxes from xywh to xyxy
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        # Scale boxes to original image size
        scale_x = img_w / self.input_w
        scale_y = img_h / self.input_h
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        # Parse keypoints (17 keypoints × 3 values each)
        keypoints_list = []
        for kpt_raw in keypoints_raw:
            kpts = kpt_raw.reshape(17, 3)  # [x, y, conf]
            kpts[:, 0] *= scale_x  # Scale x coordinates
            kpts[:, 1] *= scale_y  # Scale y coordinates
            keypoints_list.append(kpts)

        # Apply NMS
        keep_indices = nms(boxes_xyxy, confidences, self.iou_threshold)

        # Build results
        results = []
        for idx in keep_indices:
            results.append({
                'bbox': boxes_xyxy[idx].astype(int).tolist(),
                'confidence': float(confidences[idx]),
                'keypoints': keypoints_list[idx],  # (17, 3) array
            })

        return results


# ──────────────────────────────────────────────────────────────
# CPU Fallback - Ultralytics Pose
# ──────────────────────────────────────────────────────────────

class UltralyticsPoser:
    """CPU-only pose estimator using ultralytics YOLO."""

    def __init__(self, model_name="yolov11n-pose.pt", conf_threshold=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERROR] pip install ultralytics")
            sys.exit(1)

        self.conf_threshold = conf_threshold
        print(f"[INFO] Loading ultralytics pose model: {model_name} (CPU mode)")
        self.model = YOLO(model_name)

    def detect_pose(self, frame):
        """Run pose detection on CPU."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        poses = []
        for r in results:
            if r.keypoints is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    box = r.boxes[i]
                    kpts = r.keypoints[i]

                    poses.append({
                        'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist(),
                        'confidence': float(box.conf[0]),
                        'keypoints': kpts.data[0].cpu().numpy(),  # (17, 3)
                    })

        return poses


# ──────────────────────────────────────────────────────────────
# Pose Analyzer - Calculate behavioral metrics
# ──────────────────────────────────────────────────────────────

class PoseAnalyzer:
    """Analyzes pose keypoints to calculate behavioral metrics."""

    @staticmethod
    def calculate_head_tilt_angle(keypoints):
        """Calculate head tilt from ear-to-ear line.

        Returns:
            angle (degrees): Positive = right tilt, negative = left tilt
            confidence: Average confidence of ear keypoints
        """
        left_ear = keypoints[3]   # [x, y, conf]
        right_ear = keypoints[4]

        if left_ear[2] < 0.3 or right_ear[2] < 0.3:
            return None, 0.0  # Low confidence

        dx = right_ear[0] - left_ear[0]
        dy = right_ear[1] - left_ear[1]
        angle = np.degrees(np.arctan2(dy, dx))
        confidence = (left_ear[2] + right_ear[2]) / 2
        return angle, confidence

    @staticmethod
    def calculate_head_turn(keypoints):
        """Estimate horizontal head turn based on nose offset from shoulder midpoint.

        Returns:
            offset_ratio: Normalized offset (-1 to 1, 0 = centered)
            direction: 'left', 'right', or 'center'
        """
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        # Check if keypoints are visible
        if nose[2] < 0.3 or left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            return 0.0, 'unknown'

        # Calculate shoulder midpoint
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        if shoulder_width < 10:  # Too narrow, unreliable
            return 0.0, 'unknown'

        # Calculate nose offset
        nose_offset = nose[0] - shoulder_center_x
        offset_ratio = nose_offset / (shoulder_width / 2)

        # Determine direction
        if abs(offset_ratio) < 0.3:
            direction = 'center'
        elif offset_ratio > 0:
            direction = 'right'
        else:
            direction = 'left'

        return offset_ratio, direction


# ──────────────────────────────────────────────────────────────
# Behavior Tracker - Track sustained behaviors over time
# ──────────────────────────────────────────────────────────────

class BehaviorTracker:
    """Track behavioral states over time for each detected person."""

    def __init__(self, head_turn_threshold=0.4, head_tilt_threshold=25,
                 sustained_duration=3.0):
        self.states = {}  # person_id -> state dict
        self.head_turn_threshold = head_turn_threshold
        self.head_tilt_threshold = head_tilt_threshold
        self.sustained_duration = sustained_duration

    def update(self, person_id, metrics, timestamp):
        """Update state for a person."""
        if person_id not in self.states:
            self.states[person_id] = {
                'head_turn_start': None,
                'head_tilt_start': None,
                'alerts': []
            }

        state = self.states[person_id]

        # Check head turn
        head_turn_ratio = metrics.get('head_turn_ratio', 0)
        if abs(head_turn_ratio) > self.head_turn_threshold:
            if state['head_turn_start'] is None:
                state['head_turn_start'] = timestamp
            elif timestamp - state['head_turn_start'] >= self.sustained_duration:
                if 'head_turn' not in state['alerts']:
                    state['alerts'].append('head_turn')
        else:
            state['head_turn_start'] = None
            if 'head_turn' in state['alerts']:
                state['alerts'].remove('head_turn')

        # Check head tilt
        head_tilt_angle = metrics.get('head_tilt_angle')
        if head_tilt_angle is not None:
            tilt_angle = abs(head_tilt_angle)
            if tilt_angle > self.head_tilt_threshold:
                if state['head_tilt_start'] is None:
                    state['head_tilt_start'] = timestamp
                elif timestamp - state['head_tilt_start'] >= self.sustained_duration:
                    if 'head_tilt' not in state['alerts']:
                        state['alerts'].append('head_tilt')
            else:
                state['head_tilt_start'] = None
                if 'head_tilt' in state['alerts']:
                    state['alerts'].remove('head_tilt')

    def get_alerts(self, person_id):
        """Get current alerts for a person."""
        return self.states.get(person_id, {}).get('alerts', [])


# ──────────────────────────────────────────────────────────────
# Pose Visualizer - Draw pose annotations
# ──────────────────────────────────────────────────────────────

class PoseVisualizer:
    """Visualizes pose estimation results on frames."""

    # Color scheme
    COLORS = {
        'keypoint': (0, 255, 0),      # Green
        'skeleton': (255, 100, 0),    # Blue
        'bbox': (0, 200, 255),        # Orange
        'text': (255, 255, 255),      # White
        'warning': (0, 0, 255),       # Red
    }

    @staticmethod
    def draw_skeleton(frame, keypoints, kpt_threshold=0.3):
        """Draw skeleton connections between keypoints."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]

            if start_kpt[2] > kpt_threshold and end_kpt[2] > kpt_threshold:
                pt1 = (int(start_kpt[0]), int(start_kpt[1]))
                pt2 = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(frame, pt1, pt2, PoseVisualizer.COLORS['skeleton'], 2)

    @staticmethod
    def draw_keypoints(frame, keypoints, kpt_threshold=0.3):
        """Draw keypoints as circles with confidence-based sizing."""
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > kpt_threshold:
                radius = int(5 + conf * 3)  # 5-8 pixel radius
                cv2.circle(frame, (int(x), int(y)), radius,
                          PoseVisualizer.COLORS['keypoint'], -1)
                cv2.circle(frame, (int(x), int(y)), radius,
                          (0, 0, 0), 1)  # Black outline

    @staticmethod
    def draw_bbox(frame, bbox, confidence, alerts=None):
        """Draw bounding box with label."""
        x1, y1, x2, y2 = bbox
        color = PoseVisualizer.COLORS['warning'] if alerts else PoseVisualizer.COLORS['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"Person {confidence:.2f}"
        if alerts:
            label += f" [{', '.join(alerts)}]"

        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @staticmethod
    def draw_metrics_overlay(frame, person_id, metrics, alerts, position):
        """Draw calculated metrics on frame."""
        x, y = position
        line_height = 25

        # Draw semi-transparent background
        text_lines = []
        text_lines.append(f"Person {person_id}")

        if metrics.get('head_tilt_angle') is not None:
            text_lines.append(f"Head Tilt: {metrics['head_tilt_angle']:.1f}°")

        if metrics.get('head_turn_direction') != 'unknown':
            text_lines.append(f"Head Turn: {metrics['head_turn_direction']} ({metrics['head_turn_ratio']:.2f})")

        if alerts:
            text_lines.append(f"ALERT: {', '.join(alerts)}")

        # Draw background
        bg_height = len(text_lines) * line_height + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 300, y + bg_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        for i, text in enumerate(text_lines):
            color = PoseVisualizer.COLORS['warning'] if 'ALERT' in text else PoseVisualizer.COLORS['text']
            cv2.putText(frame, text, (x, y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ──────────────────────────────────────────────────────────────
# Performance Monitor
# ──────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """Track FPS and inference time with exponential smoothing."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps_smooth = 0
        self.inference_time_smooth = 0
        self.frame_count = 0
        self.start_time = time.perf_counter()

    def update(self, inference_time):
        """Update with latest inference time."""
        fps_instant = 1.0 / inference_time if inference_time > 0 else 0
        if self.frame_count == 0:
            self.fps_smooth = fps_instant
            self.inference_time_smooth = inference_time
        else:
            self.fps_smooth = self.alpha * fps_instant + (1 - self.alpha) * self.fps_smooth
            self.inference_time_smooth = self.alpha * inference_time + (1 - self.alpha) * self.inference_time_smooth
        self.frame_count += 1

    def get_stats(self):
        """Get current performance stats."""
        elapsed = time.perf_counter() - self.start_time
        return {
            'fps': self.fps_smooth,
            'inference_ms': self.inference_time_smooth * 1000,
            'frame_count': self.frame_count,
            'elapsed_sec': elapsed,
            'avg_fps': self.frame_count / elapsed if elapsed > 0 else 0
        }


# ──────────────────────────────────────────────────────────────
# Flask Web Server
# ──────────────────────────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Pose Estimation - RPi 5 + Hailo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #0ff;
            text-shadow: 0 0 10px rgba(0,255,255,0.5);
            margin-bottom: 10px;
        }
        .info {
            color: #aaa;
            margin-bottom: 20px;
            text-align: center;
        }
        .stream-container {
            border: 2px solid #0ff;
            border-radius: 8px;
            box-shadow: 0 0 20px rgba(0,255,255,0.3);
            overflow: hidden;
            max-width: 90vw;
        }
        .stream-container img {
            display: block;
            width: 100%;
            height: auto;
        }
        .footer {
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>YOLO Pose Estimation - RPi 5 + Hailo</h1>
    <p class="info">Real-time pose detection with behavioral analysis</p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <p class="footer">Stream: MJPEG | Press Ctrl+C in terminal to stop</p>
</body>
</html>
"""


def create_flask_app():
    """Create the Flask app for MJPEG streaming."""
    app = Flask(__name__)

    # Suppress Flask request logs
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/')
    def index():
        return render_template_string(HTML_PAGE)

    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                with _frame_lock:
                    frame = _latest_frame
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    time.sleep(0.05)
                # Cap stream to ~30fps to reduce CPU
                time.sleep(0.03)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def get_local_ip():
    """Get the Pi's local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def start_web_server(port=8080):
    """Start Flask in a background daemon thread."""
    app = create_flask_app()
    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, threaded=True),
        daemon=True,
    )
    thread.start()
    return thread


# ──────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────

def find_usb_camera():
    """Try /dev/video0..9 and return the first one that opens."""
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] USB camera found at /dev/video{idx}")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"[INFO] Camera resolution: {width}x{height} @ {fps}fps")
                return cap
            cap.release()
    return None


# ──────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────

def run_diagnostics():
    """Run system diagnostics and exit."""
    print("="*60)
    print("POSE ESTIMATION DIAGNOSTICS")
    print("="*60)

    # Check Python version
    print(f"\n[Python] {sys.version}")

    # Check Hailo device
    print("\n[Hailo Device]")
    if os.path.exists("/dev/hailo0"):
        print("  ✓ /dev/hailo0 exists")
    else:
        print("  ✗ /dev/hailo0 not found")
        print("    Install: sudo apt install hailo-all")

    # Check hailo_platform
    print("\n[Hailo Platform]")
    if HAILO_AVAILABLE:
        print("  ✓ hailo_platform imported successfully")
    else:
        print("  ✗ hailo_platform not available")
        print("    Install: sudo apt install hailo-all")

    # Check OpenCV
    print("\n[OpenCV]")
    try:
        print(f"  ✓ OpenCV {cv2.__version__}")
    except:
        print("  ✗ OpenCV not available")
        print("    Install: pip install opencv-python")

    # Check ultralytics
    print("\n[Ultralytics]")
    try:
        from ultralytics import YOLO
        print("  ✓ ultralytics available (CPU fallback ready)")
    except ImportError:
        print("  ✗ ultralytics not available")
        print("    Install: pip install ultralytics")

    # Check Flask
    print("\n[Flask]")
    if FLASK_AVAILABLE:
        print("  ✓ Flask available (web streaming ready)")
    else:
        print("  ✗ Flask not available")
        print("    Install: pip install flask")

    # Check camera
    print("\n[Camera]")
    cap = find_usb_camera()
    if cap:
        print("  ✓ USB camera detected")
        cap.release()
    else:
        print("  ✗ No USB camera found")
        print("    Check: ls -l /dev/video*")

    print("\n" + "="*60)
    print("Diagnostics complete.")
    print("="*60)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    global _latest_frame

    parser = argparse.ArgumentParser(
        description="YOLO Pose Estimation - RPi 5 + Hailo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPU mode (immediate testing, auto-downloads model)
  python3 pose_usb_cam_hailo.py --cpu --port 8080

  # Hailo accelerated mode
  python3 pose_usb_cam_hailo.py --model yolo11n-pose.hef --port 8080

  # With behavioral logging
  python3 pose_usb_cam_hailo.py --model yolo11n-pose.hef --log behaviors.log --save-frames
        """
    )

    # Model configuration
    parser.add_argument("--model", "--hef", dest="model",
                       default="yolov8s_pose.hef",
                       help="Path to .hef model file (default: yolov8s_pose.hef)")

    # Camera configuration
    parser.add_argument("--camera", type=int, default=None,
                       help="Camera index (default: auto-detect)")
    parser.add_argument("--width", type=int, default=640,
                       help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Camera capture height (default: 480)")

    # Detection thresholds
    parser.add_argument("--confidence", "--conf", type=float, default=0.5,
                       help="Person detection confidence threshold (default: 0.5)")
    parser.add_argument("--kpt-threshold", type=float, default=0.3,
                       help="Keypoint confidence threshold (default: 0.3)")

    # Behavioral analysis thresholds
    parser.add_argument("--head-turn-threshold", type=float, default=0.4,
                       help="Head turn detection threshold (default: 0.4)")
    parser.add_argument("--head-tilt-threshold", type=float, default=25.0,
                       help="Head tilt angle threshold in degrees (default: 25)")
    parser.add_argument("--sustained-duration", type=float, default=3.0,
                       help="Duration for sustained behavior detection (default: 3s)")

    # Display configuration
    parser.add_argument("--port", type=int, default=8080,
                       help="Web server port (default: 8080)")

    # Performance and logging
    parser.add_argument("--log", type=str,
                       help="Log behavioral alerts to file")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save frames when alerts are triggered")
    parser.add_argument("--save-dir", default="pose_evidence",
                       help="Directory to save alert frames (default: pose_evidence)")

    # System configuration
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU-only mode using ultralytics")
    parser.add_argument("--diagnose", action="store_true",
                       help="Run diagnostic tests and exit")

    args = parser.parse_args()

    # Handle diagnostic mode
    if args.diagnose:
        run_diagnostics()
        return

    print("="*60)
    print("YOLO Pose Estimation Test - RPi 5 + Hailo")
    print("AISENTINEL Project")
    print("="*60)

    # Initialize detector
    if args.cpu or not HAILO_AVAILABLE:
        if not args.cpu:
            print("[INFO] Hailo not available, using CPU fallback")
        detector = UltralyticsPoser("yolov11n-pose.pt", args.confidence)
    else:
        if not os.path.isfile(args.model):
            print(f"[ERROR] HEF model not found: {args.model}")
            print("See POSE_MODEL_SETUP.md for conversion instructions")
            print("Or use --cpu flag for immediate testing")
            sys.exit(1)
        detector = HailoPoseEstimator(args.model, args.confidence, args.kpt_threshold)

    # Open USB camera
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    else:
        cap = find_usb_camera()

    if cap is None or not cap.isOpened():
        print("[ERROR] Failed to open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize components
    analyzer = PoseAnalyzer()
    visualizer = PoseVisualizer()
    perf_monitor = PerformanceMonitor()
    behavior_tracker = BehaviorTracker(
        head_turn_threshold=args.head_turn_threshold,
        head_tilt_threshold=args.head_tilt_threshold,
        sustained_duration=args.sustained_duration
    )

    # Setup logging
    log_file = None
    if args.log:
        log_file = open(args.log, 'w')
        log_file.write(f"Pose Estimation Log - {datetime.now()}\n")
        log_file.write("="*60 + "\n")
        print(f"[INFO] Logging to: {args.log}")

    # Setup save directory
    if args.save_frames:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[INFO] Saving alert frames to: {args.save_dir}")

    # Start web server
    if not FLASK_AVAILABLE:
        print("[ERROR] Flask not available, cannot start web server")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(args.port)
    local_ip = get_local_ip()
    print(f"\n[WEB] Stream at http://{local_ip}:{args.port}\n")
    print("[INFO] Starting pose estimation... Press Ctrl+C to quit\n")

    # Main loop
    try:
        while True:
            t0 = time.perf_counter()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                time.sleep(0.1)
                continue

            # Run pose detection
            pose_results = detector.detect_pose(frame)

            # Analyze each detected person
            current_time = time.perf_counter()
            annotated_frame = frame.copy()

            for i, pose in enumerate(pose_results):
                keypoints = pose['keypoints']
                bbox = pose['bbox']
                confidence = pose['confidence']

                # Calculate metrics
                head_tilt, tilt_conf = analyzer.calculate_head_tilt_angle(keypoints)
                head_turn_ratio, turn_dir = analyzer.calculate_head_turn(keypoints)

                metrics = {
                    'head_tilt_angle': head_tilt,
                    'head_turn_ratio': head_turn_ratio,
                    'head_turn_direction': turn_dir,
                }

                # Update behavior tracker
                person_id = i  # In production, use actual tracking
                behavior_tracker.update(person_id, metrics, current_time)
                alerts = behavior_tracker.get_alerts(person_id)

                # Visualize
                visualizer.draw_skeleton(annotated_frame, keypoints, args.kpt_threshold)
                visualizer.draw_keypoints(annotated_frame, keypoints, args.kpt_threshold)
                visualizer.draw_bbox(annotated_frame, bbox, confidence, alerts)
                visualizer.draw_metrics_overlay(
                    annotated_frame, person_id, metrics, alerts,
                    position=(10, 100 + i * 130)
                )

                # Log alerts
                if alerts:
                    alert_msg = f"[{datetime.now()}] Person {person_id}: {', '.join(alerts)}"
                    if head_tilt is not None:
                        alert_msg += f" (tilt: {head_tilt:.1f}°)"
                    alert_msg += f" (turn: {turn_dir})"

                    print(f"[ALERT] {alert_msg}")
                    if log_file:
                        log_file.write(alert_msg + "\n")
                        log_file.flush()

                    # Save frame if requested
                    if args.save_frames:
                        filename = f"{args.save_dir}/alert_{person_id}_{int(current_time*1000)}_{alerts[0]}.jpg"
                        cv2.imwrite(filename, annotated_frame)

            # Update performance
            inference_time = time.perf_counter() - t0
            perf_monitor.update(inference_time)
            stats = perf_monitor.get_stats()

            # Draw FPS overlay
            cv2.putText(annotated_frame, f"FPS: {stats['fps']:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Inference: {stats['inference_ms']:.1f}ms",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(pose_results)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Update web stream
            with _frame_lock:
                _latest_frame = annotated_frame

            # Print periodic status
            if perf_monitor.frame_count % 30 == 0:
                print(f"[Frame {perf_monitor.frame_count}] "
                      f"FPS: {stats['fps']:.1f}, "
                      f"Detections: {len(pose_results)}")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")

    finally:
        cap.release()
        if log_file:
            log_file.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
