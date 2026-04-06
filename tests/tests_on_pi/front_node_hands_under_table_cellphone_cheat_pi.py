#!/usr/bin/env python3
"""
Hands Under Table + Phone / Cheat Sheet Detection - Raspberry Pi + Hailo AI HAT
===============================================================================
Combined Pi test that reuses the same calibration flow as
front_node_hands_under_table_pi.py while also running the object-cheating model
from front_node_cellphone_cheat_pi.py.

Models used:
  - yolov8s_pose.hef for person detection / tracking
  - sentinel-yolo11n-min.hef for hand detection
  - sentinel-yolov11n_new.hef for phone / cheat_sheet detection

Workflow:
  1. File dialog opens to select a video
  2. ROI calibration: draw a polygon boundary (limits tracking area)
  3. First frame: click detected persons to assign student numbers
  4. Desk ROI calibration: draw polygon ROIs for each desk
  5. Table-edge calibration: draw one 2-point line per desk near the student
  6. Web stream starts at http://<pi-ip>:8080 with live annotations
  7. Console alerts + evidence screenshots saved to ./evidence_combined/

This script preserves the hands-under-table logic:
  - desk polygons
  - student-side edge lines
  - temporal smoothing
  - warnings for 1 visible hand, alerts for 0 visible hands

It adds phone / cheat_sheet alerts associated to assigned students using the
object model from front_node_cellphone_cheat_pi.py.
"""

import sys
import os
import time
import threading
import socket
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import front_node_hands_under_table_pi as hands_mod
import front_node_cellphone_cheat_pi as obj_mod

# ── Paths ────────────────────────────────────────────────────
REPO_ROOT = SCRIPT_DIR.parent.parent

POSE_MODEL_PATH = hands_mod.POSE_MODEL_PATH
HAND_MODEL_PATH = hands_mod.HAND_MODEL_PATH
OBJECT_MODEL_PATH = obj_mod.OBJ_MODEL_PATH

EVIDENCE_DIR = SCRIPT_DIR / "evidence_combined"
HANDS_EVIDENCE_DIR = EVIDENCE_DIR / "hands"
OBJECT_EVIDENCE_DIR = EVIDENCE_DIR / "objects"

# ── Shared globals for Flask streaming ───────────────────────
_latest_frame = None
_frame_lock = threading.Lock()

try:
    from flask import Flask, Response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


def save_hand_evidence(annotated_frame, raw_frame, video_name, desk_idx, student_id,
                       ts_sec):
    """Save annotated + raw evidence for hands-under-table alerts."""
    os.makedirs(HANDS_EVIDENCE_DIR, exist_ok=True)
    ts_str = hands_mod.fmt_ts(ts_sec).replace(":", "").replace(".", "_")

    fname_ann = (
        f"{video_name}_desk{desk_idx + 1}_sid{student_id}_{ts_str}_annotated.jpg"
    )
    fname_raw = f"{video_name}_desk{desk_idx + 1}_sid{student_id}_{ts_str}_raw.jpg"

    cv2.imwrite(str(HANDS_EVIDENCE_DIR / fname_ann), annotated_frame)
    cv2.imwrite(str(HANDS_EVIDENCE_DIR / fname_raw), raw_frame)
    hands_mod.log_info(f"Hands evidence saved: {fname_ann} + raw")


def save_object_evidence(annotated_frame, raw_frame, student_num, label, conf, ts_sec):
    """Save annotated + raw evidence for phone / cheat_sheet alerts."""
    os.makedirs(OBJECT_EVIDENCE_DIR, exist_ok=True)
    ts_str = hands_mod.fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    safe_label = label.replace(" ", "_")

    fname_ann = (
        f"student{student_num}_{safe_label}_{conf:.0f}pct_{ts_str}_annotated.jpg"
    )
    fname_raw = f"student{student_num}_{safe_label}_{conf:.0f}pct_{ts_str}_raw.jpg"

    cv2.imwrite(str(OBJECT_EVIDENCE_DIR / fname_ann), annotated_frame)
    cv2.imwrite(str(OBJECT_EVIDENCE_DIR / fname_raw), raw_frame)
    hands_mod.log_info(f"Object evidence saved: {fname_ann} + raw")


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Combined Pi Detection</title>
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
    <h1>AISENTINEL - Combined Detection</h1>
    <p class="info">
        Raspberry Pi 5 + Hailo AI HAT | Hands Under Table + Phone / Cheat Sheet
    </p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <p class="footer">Stream: MJPEG | Press Ctrl+C in terminal to stop</p>
</body>
</html>
"""


def create_flask_app():
    app = Flask(__name__)
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                with _frame_lock:
                    frame = _latest_frame

                if frame is not None:
                    _, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
                else:
                    time.sleep(0.05)

                time.sleep(0.03)

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def get_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "localhost"


def start_web_server(port=8080):
    app = create_flask_app()
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
        daemon=True,
    )
    thread.start()
    return thread


def describe_first_frame_context(first_hand_dets, first_obj_dets):
    """Log a lightweight preview of first-frame detections."""
    preview_bits = [f"hands={len(first_hand_dets)}"]

    if first_obj_dets:
        obj_text = ", ".join(
            f"{det['class_name']}({det['confidence']:.0%})" for det in first_obj_dets
        )
        preview_bits.append(f"objects={obj_text}")
    else:
        preview_bits.append("objects=none")

    hands_mod.log_info("First-frame preview: " + " | ".join(preview_bits))


def run_detection(cap, person_detector, hand_detector, object_detector, tracker,
                  student_map, desk_polygons, desk_edge_lines, video_path, port,
                  roi_polygon=None):
    """Run the combined hands + object detection loop."""
    global _latest_frame

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)
    duration = total_frames / fps if fps > 0 else 0

    assigned_tids = set(student_map.keys())
    desk_states = [hands_mod.DeskState(i) for i in range(len(desk_polygons))]
    object_cooldowns = defaultdict(lambda: -999.0)
    object_stats = defaultdict(int)

    print()
    print("=" * 72)
    local_ip = get_local_ip()
    print("  AISENTINEL - Combined Pi Detection")
    print(f"  Video        : {Path(video_path).name}")
    print(f"  Resolution   : {w}x{h} | FPS: {fps:.1f} | Duration: {hands_mod.fmt_ts(duration)}")
    print(f"  Total frames : {total_frames}")
    print(f"  Students     : {len(student_map)} assigned")
    print(f"  Desk ROIs    : {len(desk_polygons)}")
    print(
        f"  Desk lines   : "
        f"{sum(1 for line in desk_edge_lines if line is not None)}/{len(desk_edge_lines)}"
    )
    roi_text = (
        f"Yes ({len(roi_polygon)} vertices)"
        if roi_polygon is not None else "No (full frame)"
    )
    print(f"  ROI          : {roi_text}")
    print("  Detecting    : hands-under-table | phone | cheat_sheet")
    print(
        f"  Hands logic  : {hands_mod.HANDS_MISSING_SUSTAIN_SEC:.1f}s sustain, "
        f"{hands_mod.SMOOTH_WINDOW_FRAMES}f smoothing"
    )
    print(f"  Obj cooldown : {obj_mod.EVENT_COOLDOWN_SEC:.1f}s per student/class")
    print(f"  Evidence     : {EVIDENCE_DIR}")
    print(f"  Web stream   : http://{local_ip}:{port}")
    print("=" * 72)
    print()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    hand_alert_total = 0
    hand_warning_total = 0
    object_alert_total = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                hands_mod.log_info("End of video reached.")
                break

            frame_idx += 1
            ts_sec = frame_idx / fps
            raw_frame = frame.copy()

            t0 = time.perf_counter()

            # 1. Person detection (pose)
            person_dets = person_detector.detect_persons(frame)
            person_dets = hands_mod.filter_detections_by_roi(person_dets, roi_polygon)

            # 2. Hand detection (sentinel hand model)
            hand_raw = hand_detector.detect(frame)
            hand_dets = [
                det for det in hand_raw if det["class_name"] == hands_mod.CLASS_HAND
            ]

            # 3. Phone / cheat_sheet detection (sentinel object model)
            object_raw = object_detector.detect(frame)
            object_raw = hands_mod.filter_detections_by_roi(object_raw, roi_polygon)

            object_dets = []
            for det in object_raw:
                cls_name = det["class_name"]
                if cls_name not in obj_mod.OBJECT_CLASSES:
                    continue
                min_conf = obj_mod.CONFIDENCE_THRESHOLDS.get(cls_name, 0.25)
                if det["confidence"] < min_conf:
                    continue
                object_dets.append(det)

            inference_ms = (time.perf_counter() - t0) * 1000

            annotated = frame.copy()

            if roi_polygon is not None:
                cv2.polylines(
                    annotated, [roi_polygon], True, (0, 255, 255), 1, cv2.LINE_AA
                )

            # 4. Person tracking
            track_ids = tracker.update(person_dets)
            student_tracks = {}

            for i, det in enumerate(person_dets):
                tid = track_ids[i]
                x1, y1, x2, y2 = det["bbox"]

                if tid == -1 or tid not in assigned_tids:
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), hands_mod.COL_UNASSIGNED, 1
                    )
                    continue

                student_tracks[tid] = det["bbox"]
                student_num = student_map[tid]
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), hands_mod.COL_STUDENT, 2
                )
                hands_mod.draw_label(
                    annotated,
                    f"#{student_num} {det['confidence']:.0%}",
                    x1,
                    y1 - 2,
                    hands_mod.COL_STUDENT,
                )

            hand_boxes = []
            for det in hand_dets:
                hand_boxes.append(det["bbox"])
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), hands_mod.COL_HAND, 2)
                hands_mod.draw_label(
                    annotated, f"hand {det['confidence']:.0%}", x1, y1 - 2,
                    hands_mod.COL_HAND
                )

            # 5. Assign students to desks
            student_to_best_desk = {}
            for track_id, student_bbox in student_tracks.items():
                desk_idx, area = hands_mod.find_desk_for_student(
                    student_bbox, desk_polygons, img_shape
                )
                if desk_idx is not None and area > 0:
                    student_to_best_desk[track_id] = (desk_idx, area)

            desk_to_candidates = defaultdict(list)
            for track_id, (desk_idx, area) in student_to_best_desk.items():
                desk_to_candidates[desk_idx].append((track_id, area))

            desk_best_student = {}
            for desk_idx, candidates in desk_to_candidates.items():
                candidates.sort(key=lambda item: item[1], reverse=True)
                desk_best_student[desk_idx] = candidates[0][0]

            for desk_idx, state in enumerate(desk_states):
                best_sid = desk_best_student.get(desk_idx)

                if best_sid is not None:
                    if state.assigned_student_id is None:
                        state.assigned_student_id = best_sid
                        state.last_student_seen_at = ts_sec
                    elif state.assigned_student_id == best_sid:
                        state.last_student_seen_at = ts_sec
                    else:
                        state.assigned_student_id = best_sid
                        state.last_student_seen_at = ts_sec
                        state.reset()
                else:
                    if state.assigned_student_id is not None:
                        if state.assigned_student_id in student_tracks:
                            state.reset_assignment()
                        else:
                            elapsed_absent = ts_sec - state.last_student_seen_at
                            if (
                                state.last_student_seen_at > 0 and
                                elapsed_absent > hands_mod.STUDENT_ABSENT_RESET_SEC
                            ):
                                state.reset_assignment()

            # 6. Associate hands to students
            student_hands = defaultdict(list)

            for hand_bbox in hand_boxes:
                hx, hy = hands_mod.bbox_center(hand_bbox)
                best_sid = None
                best_dist = float("inf")

                for track_id, student_bbox in student_tracks.items():
                    dist = hands_mod.hand_distance_to_bbox((hx, hy), student_bbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_sid = track_id

                if best_sid is not None and best_dist <= hands_mod.HAND_ASSOC_MARGIN_PX:
                    student_hands[best_sid].append(hand_bbox)

                    sx1, sy1, sx2, sy2 = student_tracks[best_sid]
                    student_cx = int((sx1 + sx2) / 2)
                    student_cy = int((sy1 + sy2) / 2)
                    cv2.line(
                        annotated,
                        (int(hx), int(hy)),
                        (student_cx, student_cy),
                        hands_mod.COL_ASSOC_LINE,
                        1,
                        cv2.LINE_AA,
                    )

            # 7. Desk-level hands-under-table logic
            frame_hand_alerts = []
            frame_hand_warnings = []

            for desk_idx, state in enumerate(desk_states):
                sid = state.assigned_student_id

                if sid is None or sid not in student_tracks:
                    state.push_observation(True)
                    state.hands_missing_start = -1.0
                    state.edge_disappear_start = -1.0
                    state.last_hands_in_desk = 0
                    continue

                hands_for_student = student_hands.get(sid, [])
                desk_poly = desk_polygons[desk_idx]
                edge_line = (
                    desk_edge_lines[desk_idx]
                    if desk_idx < len(desk_edge_lines) else None
                )

                hands_in_desk = 0
                nearest_edge_point = None
                nearest_edge_dist = float("inf")

                for hand_bbox in hands_for_student:
                    hx, hy = hands_mod.bbox_center(hand_bbox)
                    if not hands_mod.point_in_polygon(hx, hy, desk_poly):
                        continue

                    hands_in_desk += 1

                    if edge_line is None:
                        continue

                    edge_dist, edge_point = hands_mod.point_to_segment_distance(
                        (hx, hy), edge_line
                    )
                    if (
                        edge_dist <= hands_mod.TABLE_EDGE_NEAR_PX and
                        edge_dist < nearest_edge_dist
                    ):
                        nearest_edge_dist = edge_dist
                        nearest_edge_point = edge_point

                hands_present = hands_in_desk >= hands_mod.MIN_HANDS_IN_DESK
                state.last_hands_in_desk = hands_in_desk

                if hands_present:
                    state.note_visible_hands(
                        ts_sec, nearest_edge_point=nearest_edge_point
                    )
                elif edge_line is not None:
                    state.maybe_arm_edge_disappearance(ts_sec)
                else:
                    state.edge_disappear_start = -1.0

                state.push_observation(hands_present)
                smoothed_missing = state.majority_says_missing()
                edge_gate_active = (
                    edge_line is None or state.edge_disappear_start >= 0
                )
                suspicious_missing = smoothed_missing and edge_gate_active

                if suspicious_missing:
                    if state.hands_missing_start < 0:
                        state.hands_missing_start = ts_sec

                    elapsed = ts_sec - state.hands_missing_start
                    if (
                        elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC and
                        state.can_flag(ts_sec)
                    ):
                        state.last_flagged_at = ts_sec
                        student_num = student_map.get(sid, sid)
                        mode_detail = (
                            "edge-disappear" if edge_line is not None else "roi-missing"
                        )

                        if hands_in_desk == 1:
                            state.total_warnings += 1
                            hand_warning_total += 1
                            hands_mod.log_warning(
                                student_num,
                                desk_idx,
                                ts_sec,
                                f"{mode_detail}, only 1 hand visible for {elapsed:.1f}s",
                            )
                            frame_hand_warnings.append(desk_idx)
                        else:
                            state.total_alerts += 1
                            hand_alert_total += 1
                            hands_mod.log_alert(
                                student_num,
                                desk_idx,
                                ts_sec,
                                f"{mode_detail}, 0 hands visible for {elapsed:.1f}s "
                                f"({sum(1 for v in state.history if not v)}/"
                                f"{len(state.history)} frames)",
                            )
                            frame_hand_alerts.append(desk_idx)
                else:
                    state.hands_missing_start = -1.0

            # 8. Associate phone / cheat_sheet objects to students
            object_associations = obj_mod.associate_objects_to_students(
                object_dets, student_tracks
            )
            frame_object_alerts = []

            for det, assoc_tid in object_associations:
                cls_name = det["class_name"]
                conf = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                color = obj_mod.CLASS_COLORS.get(cls_name, (255, 255, 255))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if assoc_tid != -1 and assoc_tid in student_map:
                    student_num = student_map[assoc_tid]
                    hands_mod.draw_label(
                        annotated,
                        f"{cls_name} {conf:.0%} [S#{student_num}]",
                        x1,
                        y1 - 2,
                        color,
                    )
                else:
                    hands_mod.draw_label(
                        annotated,
                        f"{cls_name} {conf:.0%}",
                        x1,
                        y1 - 2,
                        color,
                    )

                object_stats[cls_name] += 1

                if assoc_tid == -1 or assoc_tid not in student_map:
                    continue

                student_num = student_map[assoc_tid]
                cooldown_key = (assoc_tid, cls_name)
                if (
                    ts_sec - object_cooldowns[cooldown_key]
                    >= obj_mod.EVENT_COOLDOWN_SEC
                ):
                    object_alert_total += 1
                    obj_mod.log_alert(cls_name, student_num, conf, ts_sec)
                    object_cooldowns[cooldown_key] = ts_sec
                    frame_object_alerts.append(
                        {
                            "class_name": cls_name,
                            "student_num": student_num,
                            "confidence": conf,
                        }
                    )

            # 9. Desk overlays + status text
            hands_mod.draw_desk_rois(annotated, desk_polygons, desk_states)
            hands_mod.draw_table_edge_lines(annotated, desk_edge_lines, desk_states)

            for desk_idx, state in enumerate(desk_states):
                if state.hands_missing_start <= 0 or state.assigned_student_id is None:
                    continue

                elapsed = ts_sec - state.hands_missing_start
                poly = desk_polygons[desk_idx]
                cx = int(poly[:, 0].mean())
                cy = int(poly[:, 1].mean()) + 20

                if (
                    elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC and
                    state.last_hands_in_desk == 0
                ):
                    txt = f"ALERT! 0 hands ({elapsed:.1f}s)"
                    color = hands_mod.COL_ALERT
                elif (
                    elapsed >= hands_mod.HANDS_MISSING_SUSTAIN_SEC and
                    state.last_hands_in_desk == 1
                ):
                    txt = f"WARNING! 1 hand ({elapsed:.1f}s)"
                    color = hands_mod.COL_WARNING
                else:
                    txt = f"Watching edge ({elapsed:.1f}s)"
                    color = hands_mod.COL_WARNING

                cv2.putText(
                    annotated,
                    txt,
                    (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    txt,
                    (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # 10. HUD + banners
            ts_text = hands_mod.fmt_ts(ts_sec)
            elapsed_wall = time.perf_counter() - t_start
            actual_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0

            hud_lines = [
                f"Frame: {frame_idx}/{total_frames} | Time: {ts_text}",
                f"Video FPS: {fps:.1f} | Processing FPS: {actual_fps:.1f}",
                (
                    f"Tracked: {len(student_tracks)}/{len(student_map)} | "
                    f"Hands: {len(hand_boxes)} | Obj: {len(object_dets)} | "
                    f"Hand A/W: {hand_alert_total}/{hand_warning_total} | "
                    f"Obj A: {object_alert_total} | Inf: {inference_ms:.0f}ms"
                ),
            ]

            hud_color = hands_mod.COL_HUD
            if hand_alert_total > 0 or object_alert_total > 0:
                hud_color = hands_mod.COL_ALERT
            elif hand_warning_total > 0:
                hud_color = hands_mod.COL_WARNING

            for i, line in enumerate(hud_lines):
                y_pos = 25 + i * 28
                cv2.putText(
                    annotated,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    hud_color,
                    2,
                    cv2.LINE_AA,
                )

            fps_badge = f"FPS {actual_fps:.1f}"
            (badge_w, badge_h), _ = cv2.getTextSize(
                fps_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            badge_x1 = w - badge_w - 28
            badge_y1 = 12
            badge_x2 = w - 10
            badge_y2 = badge_y1 + badge_h + 16
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                hud_color,
                2,
            )
            cv2.putText(
                annotated,
                fps_badge,
                (badge_x1 + 8, badge_y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                hud_color,
                2,
                cv2.LINE_AA,
            )

            banner_y = h - 30
            for event in frame_object_alerts:
                txt = (
                    f"ALERT: Student #{event['student_num']} - "
                    f"{event['class_name'].upper()}"
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, hands_mod.COL_ALERT, 2, cv2.LINE_AA
                )
                banner_y -= 35

            for desk_idx in frame_hand_alerts:
                tid = desk_states[desk_idx].assigned_student_id or 0
                student_num = student_map.get(tid, tid)
                txt = (
                    f"ALERT: Student #{student_num} hands missing from "
                    f"Desk #{desk_idx + 1}"
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, hands_mod.COL_ALERT, 2, cv2.LINE_AA
                )
                banner_y -= 35

            for desk_idx in frame_hand_warnings:
                tid = desk_states[desk_idx].assigned_student_id or 0
                student_num = student_map.get(tid, tid)
                txt = (
                    f"WARNING: Student #{student_num} long hands-missing event "
                    f"at Desk #{desk_idx + 1}"
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, txt, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, hands_mod.COL_WARNING, 2, cv2.LINE_AA
                )
                banner_y -= 35

            (tw, _), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(
                annotated, ts_text, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                annotated, ts_text, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )

            # 11. Save evidence after full annotation is ready
            for desk_idx in frame_hand_alerts:
                tid = desk_states[desk_idx].assigned_student_id or 0
                student_num = student_map.get(tid, tid)
                save_hand_evidence(
                    annotated, raw_frame, video_name, desk_idx, student_num, ts_sec
                )

            for event in frame_object_alerts:
                save_object_evidence(
                    annotated,
                    raw_frame,
                    event["student_num"],
                    event["class_name"],
                    event["confidence"],
                    ts_sec,
                )

            with _frame_lock:
                _latest_frame = annotated

            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                hands_mod.log_info(
                    f"Progress: {pct:.1f}% ({frame_idx}/{total_frames}) | "
                    f"FPS: {actual_fps:.1f}"
                )

    except KeyboardInterrupt:
        hands_mod.log_info("Stopped by user.")

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 72)
    print(f"  Summary: {Path(video_path).name}")
    print("-" * 72)
    print(f"  Frames processed : {frame_idx}")
    if elapsed > 0:
        print(f"  Average FPS      : {frame_idx / elapsed:.1f}")
    print(f"  Desk ROIs        : {len(desk_polygons)}")
    print(
        f"  Desk lines       : "
        f"{sum(1 for line in desk_edge_lines if line is not None)}/{len(desk_edge_lines)}"
    )
    print(f"  Hands alerts     : {hand_alert_total}")
    print(f"  Hands warnings   : {hand_warning_total}")
    print(f"  Object alerts    : {object_alert_total}")
    for cls_name, count in sorted(object_stats.items()):
        print(f"    {cls_name:20s}: {count} detections")
    for desk_idx, state in enumerate(desk_states):
        if state.total_alerts <= 0 and state.total_warnings <= 0:
            continue
        tid = state.assigned_student_id or 0
        student_num = student_map.get(tid, "?")
        print(
            f"    Desk #{desk_idx + 1:2d} (Student #{student_num})"
            f" : {state.total_alerts} alerts, {state.total_warnings} warnings"
        )
    if hand_alert_total > 0 or object_alert_total > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    elif hand_warning_total == 0:
        print("  No combined alerts triggered.")
    print("=" * 72)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AISENTINEL - Hands Under Table + Phone / Cheat Sheet Detection "
            "(Pi + Hailo)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_hands_under_table_cellphone_cheat_pi.py
  python3 front_node_hands_under_table_cellphone_cheat_pi.py --port 9090
  python3 front_node_hands_under_table_cellphone_cheat_pi.py --object-confidence 0.4
  python3 front_node_hands_under_table_cellphone_cheat_pi.py --object-model /path/to/model.hef
        """,
    )
    parser.add_argument(
        "--pose-model",
        default=str(POSE_MODEL_PATH),
        help=f"Path to pose HEF model for person detection (default: {POSE_MODEL_PATH})",
    )
    parser.add_argument(
        "--hand-model",
        default=str(HAND_MODEL_PATH),
        help=f"Path to hand HEF model (default: {HAND_MODEL_PATH})",
    )
    parser.add_argument(
        "--object-model", "--model",
        dest="object_model",
        default=str(OBJECT_MODEL_PATH),
        help=f"Path to phone / cheat_sheet HEF model (default: {OBJECT_MODEL_PATH})",
    )
    parser.add_argument(
        "--object-confidence", "--confidence",
        dest="object_confidence",
        type=float,
        default=0.25,
        help="Base confidence threshold for the object model (default: 0.25)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Flask web server port (default: 8080)",
    )
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  AISENTINEL - Combined Pi Detection")
    print("  Person detection : pose model (IoU tracked)")
    print("  Hand detection   : sentinel-yolo11n-min.hef (hand class)")
    print("  Object detection : sentinel-yolov11n_new.hef (phone + cheat_sheet)")
    print("  Calibration flow : ROI -> assignment -> desk polygons -> table-edge lines")
    print("=" * 72)
    print()

    if not hands_mod.HAILO_AVAILABLE or not obj_mod.HAILO_AVAILABLE:
        print(f"{hands_mod.TC.RED}[ERROR] hailo_platform is required.{hands_mod.TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    pose_path = Path(args.pose_model)
    if not pose_path.exists():
        print(f"{hands_mod.TC.RED}[ERROR] Pose HEF model not found: {pose_path}{hands_mod.TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    hand_path = Path(args.hand_model)
    if not hand_path.exists():
        print(f"{hands_mod.TC.RED}[ERROR] Hand HEF model not found: {hand_path}{hands_mod.TC.RESET}")
        sys.exit(1)

    object_path = Path(args.object_model)
    if not object_path.exists():
        print(f"{hands_mod.TC.RED}[ERROR] Object HEF model not found: {object_path}{hands_mod.TC.RESET}")
        sys.exit(1)

    hands_mod.log_info("Opening file dialog...")
    video_path = hands_mod.select_video_dialog()
    if not video_path:
        hands_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{hands_mod.TC.RED}[ERROR] File not found: {video_path}{hands_mod.TC.RESET}")
        sys.exit(1)
    hands_mod.log_info(f"Selected: {video_path}")

    shared_vdevice = hands_mod.VDevice()
    hands_mod.log_info("Hailo VDevice created (shared across all models).")

    person_detector = hands_mod.HailoPoseEstimator(
        str(pose_path),
        conf_threshold=hands_mod.PERSON_CONFIDENCE,
        vdevice=shared_vdevice,
    )
    hand_detector = hands_mod.HailoObjectDetector(
        str(hand_path),
        class_names=hands_mod.HAND_MODEL_CLASS_NAMES,
        conf_threshold=hands_mod.HAND_CONFIDENCE,
        vdevice=shared_vdevice,
    )
    object_detector = obj_mod.HailoObjectDetector(
        str(object_path),
        conf_threshold=args.object_confidence,
        vdevice=shared_vdevice,
    )

    print(f"\n{hands_mod.TC.BOLD}Object model classes:{hands_mod.TC.RESET}")
    for idx, name in obj_mod.CLASS_NAMES.items():
        role = "  << ALERT" if name in obj_mod.ALERT_CLASSES else "  << IGNORED"
        thresh = obj_mod.CONFIDENCE_THRESHOLDS.get(name, "-")
        print(f"  [{idx}] {name} (thresh={thresh}){role}")

    print(f"\n{hands_mod.TC.BOLD}Hand model classes:{hands_mod.TC.RESET}")
    for idx, name in hands_mod.HAND_MODEL_CLASS_NAMES.items():
        role = "  << USED" if name == hands_mod.CLASS_HAND else "  << IGNORED"
        print(f"  [{idx}] {name}{role}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{hands_mod.TC.RED}[ERROR] Cannot open video: {video_path}{hands_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{hands_mod.TC.RED}[ERROR] Cannot read first frame.{hands_mod.TC.RESET}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0
    hands_mod.log_info(f"Video resolution: {w}x{h}")

    hands_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = hands_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        hands_mod.log_info("Cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    hands_mod.log_info("Running pose detection on first frame for student assignment...")
    first_person_dets = person_detector.detect_persons(first_frame)
    first_person_dets = hands_mod.filter_detections_by_roi(
        first_person_dets, roi_polygon
    )

    tracker = obj_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_person_dets)

    roi_label = " (within ROI)" if roi_polygon is not None else ""
    hands_mod.log_info(
        f"Detected {len(first_person_dets)} persons on first frame{roi_label}."
    )

    first_hand_dets = [
        det for det in hand_detector.detect(first_frame)
        if det["class_name"] == hands_mod.CLASS_HAND
    ]
    first_obj_dets = []
    for det in object_detector.detect(first_frame):
        if det["class_name"] not in obj_mod.OBJECT_CLASSES:
            continue
        min_conf = obj_mod.CONFIDENCE_THRESHOLDS.get(det["class_name"], 0.25)
        if det["confidence"] < min_conf:
            continue
        first_obj_dets.append(det)
    describe_first_frame_context(first_hand_dets, first_obj_dets)

    print()
    print(f"  {hands_mod.TC.BOLD}Instructions:{hands_mod.TC.RESET}")
    print("    1. Click on a person to select them (cyan highlight)")
    print("    2. Type the student number (digits)")
    print("    3. Press ENTER to assign")
    print("    4. Repeat for each student you want to monitor")
    print("    5. Press S to start")
    print()

    student_map = hands_mod.run_assignment_phase(
        first_frame, first_person_dets, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        hands_mod.log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        hands_mod.log_info("No students assigned. Exiting.")
        sys.exit(0)

    tracker.keep_only(set(student_map.keys()))
    hands_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    hands_mod.log_info("Now draw polygon ROIs for each desk on the first frame.")
    desk_polygons = hands_mod.calibrate_desk_rois(first_frame)
    if desk_polygons is None or len(desk_polygons) == 0:
        cap.release()
        hands_mod.log_info("No desk ROIs defined. Exiting.")
        sys.exit(0)

    hands_mod.log_info(
        "Now draw one student-side table-edge line for each desk "
        "(or press S to skip a desk)."
    )
    desk_edge_lines = hands_mod.calibrate_table_edge_lines(first_frame, desk_polygons)
    if desk_edge_lines is None:
        cap.release()
        hands_mod.log_info("Table-edge calibration cancelled. Exiting.")
        sys.exit(0)

    if not FLASK_AVAILABLE:
        print(f"{hands_mod.TC.RED}[ERROR] Flask is required for web streaming.{hands_mod.TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(args.port)
    local_ip = get_local_ip()
    hands_mod.log_info(f"Web stream at http://{local_ip}:{args.port}")

    hands_mod.log_info("Starting combined detection...")
    run_detection(
        cap,
        person_detector,
        hand_detector,
        object_detector,
        tracker,
        student_map,
        desk_polygons,
        desk_edge_lines,
        video_path,
        args.port,
        roi_polygon=roi_polygon,
    )
    cap.release()
    hands_mod.log_info("Done!")


if __name__ == "__main__":
    main()
