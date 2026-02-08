#!/usr/bin/env python3
"""
YOLO Object Detection on Raspberry Pi 5 (Trixie OS) + Hailo AI Accelerator
Using a USB Camera — No hailo-apps required.

This script uses the HailoRT Python API (hailo_platform) directly with OpenCV
to capture frames from a USB camera, run YOLOv8s inference on the Hailo NPU,
and display bounding boxes in real time.

Usage:
    python3 yolo_usb_cam_hailo.py [--model MODEL.hef] [--camera 0] [--threshold 0.5]

Requirements:
    - Raspberry Pi 5 with Trixie OS
    - Hailo AI Kit / AI HAT+ installed and configured
    - USB camera plugged in
    - Python packages: opencv-python, numpy, hailo_platform (system-installed)
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Try to import hailo_platform — available as a system package
# after installing hailo-all on Trixie
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

# Distinct colours for each class (randomly seeded for consistency)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_LABELS), 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
# YOLOv8 post-processing helpers
# ──────────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return out


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression (simple Python version)."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def postprocess_yolov8(raw_output, img_w, img_h, input_size=640,
                       conf_threshold=0.5, iou_threshold=0.45):
    """
    Decode raw YOLOv8 output tensor into a list of detections.

    YOLOv8 raw output shape from Hailo is typically one of:
      - (1, 84, 8400)  — transposed format [batch, 4+80, num_preds]
      - (8400, 84)     — already transposed
      - Multiple output tensors for different stride heads

    Each detection row (after transpose): [cx, cy, w, h, cls0, cls1, ..., cls79]

    Returns list of dicts: {label, confidence, box: [x1,y1,x2,y2]}
    """
    # Handle dict output (multiple output layers from Hailo)
    if isinstance(raw_output, dict):
        arrays = list(raw_output.values())
        # If we have multiple tensors, try to concatenate them
        # or use the largest one
        if len(arrays) == 1:
            output = arrays[0]
        else:
            # YOLOv8 may output separate tensors per stride head
            # Try concatenating along prediction dimension
            try:
                # Flatten any batch dimensions and concatenate
                flat = []
                for a in arrays:
                    a = np.squeeze(a)
                    if a.ndim == 3:
                        # (batch, features, preds) or (batch, preds, features)
                        a = a.reshape(-1, a.shape[-1])
                    elif a.ndim == 1:
                        continue
                    flat.append(a)
                output = np.concatenate(flat, axis=0)
            except Exception:
                # Fall back to largest tensor
                output = max(arrays, key=lambda x: x.size)
    else:
        output = raw_output

    output = np.squeeze(output)

    # Determine shape and transpose if needed
    # Target shape: (num_predictions, 4 + num_classes)
    if output.ndim == 2:
        if output.shape[0] == 84 and output.shape[1] > 84:
            # Shape is (84, 8400) → transpose to (8400, 84)
            output = output.T
        elif output.shape[1] == 84:
            pass  # Already (N, 84)
        else:
            # Try the other way
            if output.shape[1] > output.shape[0]:
                output = output.T
    elif output.ndim == 3:
        # (1, 84, 8400) → squeeze and transpose
        output = output.reshape(output.shape[-2], output.shape[-1])
        if output.shape[0] < output.shape[1]:
            output = output.T

    num_classes = output.shape[1] - 4
    if num_classes <= 0:
        return []

    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]

    # Get best class per prediction
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert to xyxy
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # Scale boxes from model input size to original image size
    scale_x = img_w / input_size
    scale_y = img_h / input_size
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    # Clip to image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_h)

    # Per-class NMS
    detections = []
    unique_classes = np.unique(class_ids)
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
            detections.append({
                "label": label,
                "confidence": float(cls_scores[idx]),
                "box": cls_boxes[idx].astype(int).tolist(),
                "class_id": int(cls_id),
            })

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])
        label = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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

        # Create a virtual device (auto-discovers Hailo-8/8L on PCIe)
        self.vdevice = VDevice()

        # Configure the device with the model
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        # Get input/output stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        # Configure vstream params — request UINT8 input (matches OpenCV frames)
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        # Output as FLOAT32 for easier post-processing
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Determine model input dimensions
        self.input_shape = self.input_vstream_info[0].shape  # e.g. (640, 640, 3)
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        print(f"[INFO] Model input shape : {self.input_shape}")
        for out_info in self.output_vstream_info:
            print(f"[INFO] Model output layer: {out_info.name} → {out_info.shape}")
        print(f"[INFO] Hailo device ready.")

    def detect(self, frame):
        """Run detection on a single BGR frame. Returns list of detections."""
        img_h, img_w = frame.shape[:2]

        # Preprocess: resize to model input, keep as uint8 RGB
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Add batch dimension: (1, H, W, 3)
        input_data = np.expand_dims(rgb, axis=0)

        # Build the input dict keyed by input layer name
        input_dict = {self.input_vstream_info[0].name: input_data}

        # Run inference
        with self.network_group.activate(self.network_group_params):
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                results = infer_pipeline.infer(input_dict)

        # Post-process
        detections = postprocess_yolov8(
            results, img_w, img_h,
            input_size=self.input_w,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        return detections


# ──────────────────────────────────────────────────────────────
# CPU fallback using ultralytics (no Hailo required)
# ──────────────────────────────────────────────────────────────

class UltralyticsDetector:
    """Fallback CPU-only detector using ultralytics YOLO."""

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERROR] ultralytics not installed. Install with:")
            print("        pip install ultralytics --break-system-packages")
            sys.exit(1)

        self.conf_threshold = conf_threshold
        print(f"[INFO] Loading ultralytics model: {model_name} (CPU mode)")
        self.model = YOLO(model_name)
        print("[INFO] Model loaded — running on CPU (expect ~2-5 FPS on Pi 5).")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "class_id": cls_id,
                })
        return detections


# ──────────────────────────────────────────────────────────────
# Main loop
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


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection — RPi 5 + Hailo + USB Cam")
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
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless — print detections to terminal only")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode using ultralytics (no Hailo)")
    args = parser.parse_args()

    # ── Select detector ──────────────────────────────────────
    if args.cpu or not HAILO_AVAILABLE:
        detector = UltralyticsDetector(
            model_name="yolov8n.pt",
            conf_threshold=args.threshold,
        )
    else:
        if not os.path.isfile(args.model):
            print(f"[ERROR] HEF model file not found: {args.model}")
            print("        Download one with:")
            print("        wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
                  "ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s.hef")
            print("        (Use hailo8l or hailo8 path depending on your hardware.)")
            sys.exit(1)
        detector = HailoDetector(
            args.model,
            conf_threshold=args.threshold,
            iou_threshold=args.iou,
        )

    # ── Open USB camera ──────────────────────────────────────
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera at index {args.camera}")
            sys.exit(1)
    else:
        cap = find_usb_camera()
        if cap is None:
            print("[ERROR] No USB camera found. Check connection and try:")
            print("        ls /dev/video*")
            print("        v4l2-ctl --list-devices")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")
    print(f"[INFO] Press 'q' to quit.\n")

    # ── Main loop ────────────────────────────────────────────
    fps_smooth = 0
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame — retrying...")
                time.sleep(0.1)
                continue

            # Run detection
            detections = detector.detect(frame)

            # Calculate FPS
            elapsed = time.perf_counter() - t0
            fps_instant = 1.0 / elapsed if elapsed > 0 else 0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant
            frame_count += 1

            # Print detections to terminal periodically
            if frame_count % 30 == 0 or len(detections) > 0:
                det_str = ", ".join(
                    f'{d["label"]}({d["confidence"]:.0%})' for d in detections
                )
                print(f"[FPS: {fps_smooth:5.1f}]  Detections: {det_str or 'none'}")

            # Display
            if not args.no_display:
                frame = draw_detections(frame, detections)
                cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("YOLO Detection — RPi 5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — shutting down.")

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
