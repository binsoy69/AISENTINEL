import cv2
import argparse
import time
import sys
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Back Node Test for AISENTINEL (YOLOv11 on RPi 5 CPU)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to model file (.pt or .ncnn directory)")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0 for USB webcam)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--export-ncnn", action="store_true", help="Export the .pt model to NCNN format before running")
    args = parser.parse_args()

    # Load Model
    print(f"[INFO] Loading model: {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Export to NCNN if requested
    if args.export_ncnn:
        if not args.model.endswith('.pt'):
            print("[ERROR] Export requires a .pt model input.")
            sys.exit(1)
        print("[INFO] Exporting model to NCNN format...")
        model.export(format="ncnn")
        # Update model path to the exported directory
        args.model = args.model.replace(".pt", "_ncnn_model")
        print(f"[INFO] Reloading exported model: {args.model}...")
        model = YOLO(args.model, task="detect")

    # Initialize Camera
    print(f"[INFO] Opening camera source: {args.source}...")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.source}.")
        sys.exit(1)

    # Set camera properties for better performance (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Starting inference loop. Press 'q' to exit.")
    
    prev_time = 0
    fps_avg = 0
    alpha = 0.1  # Smoothing factor for FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # Inference
            start_time = time.time()
            results = model(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            end_time = time.time()

            # FPS Calculation
            inference_time = end_time - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            # Simple exponential moving average for stable FPS display
            fps_avg = (alpha * fps) + ((1 - alpha) * fps_avg)

            # Visualization
            annotated_frame = results[0].plot()

            # Overlay FPS
            cv2.putText(annotated_frame, f"FPS: {fps_avg:.1f} ({inference_time*1000:.1f}ms)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show Frame
            cv2.imshow("AISENTINEL Back Node Test", annotated_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup done.")

if __name__ == "__main__":
    main()
