#!/usr/bin/env python3
"""
YOLO Object Detection on Raspberry Pi 5 (Trixie OS) + Hailo AI Accelerator
Using a USB Camera — No hailo-apps required.

Live preview via web browser at http://<pi-ip>:8080

Usage:
    python3 yolo_usb_cam_hailo.py [--model MODEL.hef] [--camera 0] [--threshold 0.5]
    Then open http://<your-pi-ip>:8080 in any browser.

Requirements:
    - Raspberry Pi 5 with Trixie OS
    - Hailo AI Kit / AI HAT+ installed and configured
    - USB camera plugged in
    - Python packages: opencv-python, numpy, flask, hailo_platform (system-installed)
    - pip install flask  (if not already installed)
"""

import argparse
import time
import sys
import os
import threading
import socket

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
# COCO class labels (80 classes used by YOLOv8 COCO models)
# ──────────────────────────────────────────────────────────────
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_LABELS), 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
# YOLOv8 post-processing helpers
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
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep)


def postprocess_yolov8_nms(raw_output, img_w, img_h, input_size=640, conf_threshold=0.5):
    """
    Decode Hailo NMS-postprocessed YOLOv8 output.
    Output is a list of 80 arrays (one per class), each row: [y1, x1, y2, x2, score]
    Coordinates are normalized [0, 1].
    """
    if isinstance(raw_output, dict):
        output = list(raw_output.values())[0]
    else:
        output = raw_output

    # Remove batch dimension
    if isinstance(output, np.ndarray):
        output = np.squeeze(output, axis=0) if output.ndim > 2 else output
    elif isinstance(output, list) and len(output) == 1 and isinstance(output[0], (list, np.ndarray)):
        output = output[0]

    detections = []
    for cls_id, class_dets in enumerate(output):
        if not isinstance(class_dets, np.ndarray):
            try:
                class_dets = np.array(class_dets, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        if class_dets.size == 0:
            continue
        if class_dets.ndim == 1:
            class_dets = class_dets.reshape(1, -1)

        for det in class_dets:
            if len(det) < 5:
                continue
            score = float(det[4])
            if score < conf_threshold:
                continue
            y1, x1, y2, x2 = det[0], det[1], det[2], det[3]
            x1_px = int(np.clip(x1 * img_w, 0, img_w))
            y1_px = int(np.clip(y1 * img_h, 0, img_h))
            x2_px = int(np.clip(x2 * img_w, 0, img_w))
            y2_px = int(np.clip(y2 * img_h, 0, img_h))
            if x2_px - x1_px < 2 or y2_px - y1_px < 2:
                continue
            label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
            detections.append({
                "label": label, "confidence": score,
                "box": [x1_px, y1_px, x2_px, y2_px], "class_id": int(cls_id),
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def postprocess_yolov8_raw(raw_output, img_w, img_h, input_size=640,
                           conf_threshold=0.5, iou_threshold=0.45):
    """Decode raw (non-NMS) YOLOv8 output tensor."""
    if isinstance(raw_output, dict):
        arrays = list(raw_output.values())
        if len(arrays) == 1:
            output = arrays[0]
        else:
            try:
                flat = []
                for a in arrays:
                    a = np.squeeze(a)
                    if a.ndim == 3:
                        a = a.reshape(-1, a.shape[-1])
                    elif a.ndim == 1:
                        continue
                    flat.append(a)
                output = np.concatenate(flat, axis=0)
            except Exception:
                output = max(arrays, key=lambda x: x.size)
    else:
        output = raw_output

    output = np.squeeze(output)
    if output.ndim == 2:
        if output.shape[0] in (84, 85) and output.shape[1] > 85:
            output = output.T
        elif output.shape[1] > output.shape[0]:
            output = output.T
    elif output.ndim == 3:
        output = output.reshape(output.shape[-2], output.shape[-1])
        if output.shape[0] < output.shape[1]:
            output = output.T

    num_classes = output.shape[1] - 4
    if num_classes <= 0:
        return []

    boxes_xywh, class_scores = output[:, :4], output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    mask = confidences > conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh, class_ids, confidences = boxes_xywh[mask], class_ids[mask], confidences[mask]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    boxes_xyxy[:, [0, 2]] *= img_w / input_size
    boxes_xyxy[:, [1, 3]] *= img_h / input_size
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_h)

    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes, cls_scores = boxes_xyxy[cls_mask], confidences[cls_mask]
        for idx in nms(cls_boxes, cls_scores, iou_threshold):
            label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
            detections.append({
                "label": label, "confidence": float(cls_scores[idx]),
                "box": cls_boxes[idx].astype(int).tolist(), "class_id": int(cls_id),
            })
    return detections


def draw_detections(frame, detections, fps=0):
    """Draw bounding boxes, labels, and FPS on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS overlay
    fps_text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (5, 5), (160, 35), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Detection count
    count_text = f"Objects: {len(detections)}"
    cv2.rectangle(frame, (frame.shape[1] - 170, 5), (frame.shape[1] - 5, 35), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (frame.shape[1] - 165, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────
# HAILO inference engine
# ──────────────────────────────────────────────────────────────

class HailoDetector:
    """Wraps HailoRT Python API for YOLOv8 inference on the Hailo NPU."""

    def __init__(self, hef_path, conf_threshold=0.5, iou_threshold=0.45):
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform is not installed.")

        self.conf_threshold = conf_threshold
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

    def detect(self, frame):
        """Run detection on a single BGR frame."""
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

        if self.has_nms:
            return postprocess_yolov8_nms(
                results, img_w, img_h,
                input_size=self.input_w,
                conf_threshold=self.conf_threshold,
            )
        else:
            return postprocess_yolov8_raw(
                results, img_w, img_h,
                input_size=self.input_w,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )


# ──────────────────────────────────────────────────────────────
# CPU fallback
# ──────────────────────────────────────────────────────────────

class UltralyticsDetector:
    """Fallback CPU-only detector using ultralytics YOLO."""

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERROR] pip install ultralytics --break-system-packages")
            sys.exit(1)
        self.conf_threshold = conf_threshold
        print(f"[INFO] Loading ultralytics model: {model_name} (CPU mode)")
        self.model = YOLO(model_name)

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                detections.append({
                    "label": self.model.names.get(cls_id, f"class_{cls_id}"),
                    "confidence": float(box.conf[0]),
                    "box": [x1, y1, x2, y2], "class_id": cls_id,
                })
        return detections


# ──────────────────────────────────────────────────────────────
# Flask MJPEG web preview server
# ──────────────────────────────────────────────────────────────

# Shared state for the web stream
_latest_frame = None
_frame_lock = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Detection - RPi 5</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e; color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
        }
        h1 {
            margin: 20px 0 10px;
            font-size: 1.5em; font-weight: 400;
            color: #0ff;
        }
        .info {
            color: #888; font-size: 0.9em;
            margin-bottom: 15px;
        }
        .stream-container {
            border: 2px solid #0ff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.15);
            max-width: 95vw;
        }
        img {
            display: block;
            width: 100%;
            max-width: 960px;
        }
        .footer {
            margin-top: 15px;
            color: #555; font-size: 0.8em;
        }
    </style>
</head>
<body>
    <h1>YOLO Object Detection - RPi 5 + Hailo</h1>
    <p class="info">Live USB camera feed with real-time detection</p>
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
                return cap
            cap.release()
    return None


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    global _latest_frame

    parser = argparse.ArgumentParser(description="YOLO Detection - RPi 5 + Hailo + USB Cam")
    parser.add_argument("--model", default="yolov8s.hef",
                        help="Path to .hef model file (default: yolov8s.hef)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (default: auto-detect)")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera capture width")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera capture height")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web preview port (default: 8080)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web preview (terminal output only)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode using ultralytics")
    args = parser.parse_args()

    # ── Select detector ──────────────────────────────────────
    if args.cpu or not HAILO_AVAILABLE:
        detector = UltralyticsDetector("yolov8n.pt", conf_threshold=args.threshold)
    else:
        if not os.path.isfile(args.model):
            print(f"[ERROR] HEF model file not found: {args.model}")
            print("        Download with:")
            print("        wget -O yolov8n.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
                  "ModelZoo/Compiled/v2.14.0/hailo8/yolov8n.hef")
            sys.exit(1)
        detector = HailoDetector(args.model, args.threshold, args.iou)

    # ── Open USB camera ──────────────────────────────────────
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera at index {args.camera}")
            sys.exit(1)
    else:
        cap = find_usb_camera()
        if cap is None:
            print("[ERROR] No USB camera found. Try: ls /dev/video* && v4l2-ctl --list-devices")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    # ── Start web preview ────────────────────────────────────
    if not args.no_web and FLASK_AVAILABLE:
        start_web_server(args.port)
        local_ip = get_local_ip()
        print(f"")
        print(f"  +-------------------------------------------------+")
        print(f"  |  LIVE PREVIEW ready at:                          |")
        print(f"  |  -> http://{local_ip}:{args.port:<24}|")
        print(f"  |  -> http://localhost:{args.port:<23}|")
        print(f"  +-------------------------------------------------+")
        print(f"")
    elif not args.no_web:
        print("[WARN] Flask not available - web preview disabled.")
        print("       Install with: pip install flask")

    print(f"[INFO] Running... Press Ctrl+C to stop.\n")

    # ── Main loop ────────────────────────────────────────────
    fps_smooth = 0
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame - retrying...")
                time.sleep(0.1)
                continue

            # Run detection
            detections = detector.detect(frame)

            # Calculate FPS
            elapsed = time.perf_counter() - t0
            fps_instant = 1.0 / elapsed if elapsed > 0 else 0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant
            frame_count += 1

            # Draw detections onto frame
            annotated = draw_detections(frame.copy(), detections, fps_smooth)

            # Update the shared frame for web streaming
            with _frame_lock:
                _latest_frame = annotated

            # Print to terminal periodically
            if frame_count % 30 == 0 or (len(detections) > 0 and frame_count % 5 == 0):
                det_str = ", ".join(
                    f'{d["label"]}({d["confidence"]:.0%})' for d in detections[:8]
                )
                if len(detections) > 8:
                    det_str += f", +{len(detections)-8} more"
                print(f"  [FPS: {fps_smooth:5.1f}]  {det_str or 'no detections'}")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")

    finally:
        cap.release()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
