#!/usr/bin/env python3
"""
Head Behavior + Passing Papers Detection - Raspberry Pi + Hailo AI HAT
======================================================================
Combined Pi test that reuses the existing head-behavior and
passing-papers pose pipelines while running only one pose inference pass
per frame.

Detections:
  - Head tilt
  - Shoulder turn
  - Passing papers between neighboring students

Workflow:
  1. File dialog opens to select a video
  2. ROI calibration: draw a polygon boundary (optional)
  3. First frame shown with detected persons - click to assign student numbers
  4. Web stream starts at http://<pi-ip>:8080 with live annotations
  5. Console alerts + evidence screenshots saved to ./evidence_combined/

Notes:
  - ROI calibration comes from the passing-papers workflow.
  - Baseline yaw capture for head-tilt compensation comes from the
    head-behavior workflow.
  - The overlay includes the live processing FPS.
"""

import os
import sys
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

import front_node_head_behavior_pi as head_mod
import front_node_passing_papers_pi as pass_mod

# ── Paths ────────────────────────────────────────────────────
POSE_MODEL_PATH = head_mod.POSE_MODEL_PATH
EVIDENCE_DIR = SCRIPT_DIR / "evidence_combined"
HEAD_EVIDENCE_DIR = EVIDENCE_DIR / "head_behavior"
PASSING_EVIDENCE_DIR = EVIDENCE_DIR / "passing_papers"

# ── Shared globals for Flask streaming ───────────────────────
_latest_frame = None
_frame_lock = threading.Lock()

try:
    from flask import Flask, Response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AISENTINEL - Head + Passing Detection</title>
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
        Raspberry Pi 5 + Hailo AI HAT | Head Behavior + Passing Papers
    </p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <p class="footer">Stream: MJPEG | Press Ctrl+C in terminal to stop</p>
</body>
</html>
"""


COLOR_PRIORITY = {
    head_mod.COL_NORMAL: 0,
    head_mod.COL_HEAD_TILT: 1,
    head_mod.COL_SHOULDER_TURN: 1,
    pass_mod.COL_WARNING: 1,
    head_mod.COL_FLAGGED: 2,
}


def elevate_color(current, new_color):
    """Keep the highest-severity color already assigned to a student box."""
    if COLOR_PRIORITY.get(new_color, 0) >= COLOR_PRIORITY.get(current, 0):
        return new_color
    return current


def save_head_evidence(frame, student_num, behavior, ts_sec):
    os.makedirs(HEAD_EVIDENCE_DIR, exist_ok=True)
    ts_str = head_mod.fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    fname = f"student{student_num}_{behavior}_{ts_str}.jpg"
    cv2.imwrite(str(HEAD_EVIDENCE_DIR / fname), frame)
    head_mod.log_info(f"Head evidence saved: {fname}")


def save_passing_evidence(frame, student_nums, ts_sec):
    os.makedirs(PASSING_EVIDENCE_DIR, exist_ok=True)
    ts_str = head_mod.fmt_ts(ts_sec).replace(":", "").replace(".", "_")
    nums_str = "_".join(str(n) for n in student_nums)
    fname = f"passing_s{nums_str}_{ts_str}.jpg"
    cv2.imwrite(str(PASSING_EVIDENCE_DIR / fname), frame)
    head_mod.log_info(f"Passing evidence saved: {fname}")


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


def draw_fps_badge(frame, fps_value, color):
    badge_text = f"FPS {fps_value:.1f}"
    (badge_w, badge_h), _ = cv2.getTextSize(
        badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    x2 = frame.shape[1] - 10
    y1 = 12
    x1 = x2 - badge_w - 18
    y2 = y1 + badge_h + 16

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        badge_text,
        (x1 + 8, y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def run_detection(cap, estimator, tracker, student_map, baseline_yaw_map,
                  video_path, port, roi_polygon=None):
    """Run one pose pass per frame, then apply both behavior detectors."""
    global _latest_frame

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    assigned_tids = set(student_map.keys())
    head_students = {
        tid: head_mod.StudentState(
            track_id=tid,
            student_num=student_num,
            baseline_yaw=baseline_yaw_map.get(tid, 0.0),
        )
        for tid, student_num in student_map.items()
    }
    passing_students = {
        tid: pass_mod.StudentState(track_id=tid, student_num=student_num)
        for tid, student_num in student_map.items()
    }
    pair_states = {}

    def get_pair_state(tid_a, tid_b):
        key = frozenset((tid_a, tid_b))
        if key not in pair_states:
            pair_states[key] = pass_mod.PairInteractionState(tid_a=tid_a, tid_b=tid_b)
        return pair_states[key]

    print()
    print("=" * 74)
    print("  AISENTINEL - Head Behavior + Passing Papers Detection (Pi + Hailo)")
    print(f"  Video          : {Path(video_path).name}")
    print(
        f"  Resolution     : {width}x{height} | FPS: {fps:.1f} | "
        f"Duration: {head_mod.fmt_ts(duration)}"
    )
    print(f"  Students       : {len(student_map)} assigned")
    roi_text = (
        f"Yes ({len(roi_polygon)} vertices)"
        if roi_polygon is not None else "No (full frame)"
    )
    print(f"  ROI            : {roi_text}")
    print("  Detecting      : head tilt | shoulder turn | passing papers")
    print(
        f"  Head sustain   : {head_mod.SUSTAINED_SEC:.1f}s | "
        f"Passing sustain: {pass_mod.MIN_INTERACTION_SEC:.2f}s"
    )
    print(
        f"  Cooldowns      : head {head_mod.EVENT_COOLDOWN_SEC:.1f}s | "
        f"passing {pass_mod.EVENT_COOLDOWN_SEC:.1f}s"
    )
    print(
        "  Passing rule   : only consecutive student numbers can trigger "
        "pair alerts"
    )
    print(f"  Evidence       : {EVIDENCE_DIR}")
    print(f"  Web stream     : http://{get_local_ip()}:{port}")
    print("=" * 74)
    print()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    head_stats = defaultdict(int)
    passing_alert_total = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                head_mod.log_info("End of video reached.")
                break

            frame_idx += 1
            ts_sec = frame_idx / fps if fps > 0 else 0.0

            t0 = time.perf_counter()
            detections = estimator.detect_pose(frame)
            detections = pass_mod.filter_detections_by_roi(detections, roi_polygon)
            track_ids = tracker.update(detections)
            inference_ms = (time.perf_counter() - t0) * 1000

            annotated = frame.copy()
            if roi_polygon is not None:
                cv2.polylines(
                    annotated, [roi_polygon], True, (0, 255, 255), 1, cv2.LINE_AA
                )

            head_frame_events = []
            passing_frame_events = []
            frame_kp_data = {}
            all_student_bboxes = {}
            per_student_display = {}

            # Pass 1: collect tracked students and evaluate head behaviors.
            for i, det in enumerate(detections):
                tid = track_ids[i]
                bbox = det["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if tid == -1 or tid not in assigned_tids:
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), head_mod.COL_UNASSIGNED, 1
                    )
                    continue

                kp = det["keypoints"]
                kp_xy = kp[:, :2]
                kp_conf = kp[:, 2]

                all_student_bboxes[tid] = tuple(bbox)
                frame_kp_data[tid] = (kp_xy, kp_conf)
                per_student_display[tid] = {
                    "color": head_mod.COL_NORMAL,
                    "labels": [],
                }

                pass_mod.draw_skeleton(annotated, kp_xy, kp_conf)
                for kp_idx in (pass_mod.KP_LEFT_WRIST, pass_mod.KP_RIGHT_WRIST):
                    if kp_idx < len(kp_conf) and kp_conf[kp_idx] > pass_mod.KP_CONF_THRESH:
                        wx, wy = int(kp_xy[kp_idx][0]), int(kp_xy[kp_idx][1])
                        cv2.circle(
                            annotated, (wx, wy), 5, pass_mod.COL_WRIST, -1, cv2.LINE_AA
                        )

                display = per_student_display[tid]
                head_state = head_students[tid]

                is_tilted, tilt_score = head_mod.detect_head_tilt(
                    kp_xy, kp_conf, baseline_yaw=head_state.baseline_yaw
                )
                if is_tilted:
                    if head_state.head_tilt_start < 0:
                        head_state.head_tilt_start = ts_sec
                    elapsed = ts_sec - head_state.head_tilt_start

                    if (
                        elapsed >= head_mod.SUSTAINED_SEC
                        and head_state.can_flag("head_tilt", ts_sec)
                    ):
                        head_state.head_tilt_flagged_at = ts_sec
                        head_stats["head_tilt"] += 1
                        head_mod.log_alert(
                            "HEAD TILT",
                            head_state.student_num,
                            ts_sec,
                            f"score={tilt_score:.2f}, sustained {elapsed:.1f}s",
                            head_mod.TC.YELLOW,
                        )
                        head_frame_events.append(("head_tilt", head_state.student_num))

                    if elapsed >= 1.0:
                        display["labels"].append(
                            f"HEAD TILT {tilt_score:.1f}x ({elapsed:.1f}s)"
                        )
                        display["color"] = elevate_color(
                            display["color"], head_mod.COL_HEAD_TILT
                        )
                        if elapsed >= head_mod.SUSTAINED_SEC:
                            display["color"] = elevate_color(
                                display["color"], head_mod.COL_FLAGGED
                            )
                else:
                    head_state.head_tilt_start = -1.0

                is_turned, shoulder_angle, turn_dir = head_mod.detect_shoulder_turn(
                    kp_xy, kp_conf
                )
                if (
                    kp_conf[head_mod.KP_LEFT_SHOULDER] > head_mod.KP_CONF_THRESH
                    and kp_conf[head_mod.KP_RIGHT_SHOULDER] > head_mod.KP_CONF_THRESH
                ):
                    ls_pt = (
                        int(kp_xy[head_mod.KP_LEFT_SHOULDER][0]),
                        int(kp_xy[head_mod.KP_LEFT_SHOULDER][1]),
                    )
                    rs_pt = (
                        int(kp_xy[head_mod.KP_RIGHT_SHOULDER][0]),
                        int(kp_xy[head_mod.KP_RIGHT_SHOULDER][1]),
                    )
                    shoulder_color = (
                        head_mod.COL_SHOULDER_TURN if is_turned else (100, 200, 100)
                    )
                    cv2.line(annotated, ls_pt, rs_pt, shoulder_color, 3, cv2.LINE_AA)
                    mid_x = (ls_pt[0] + rs_pt[0]) // 2
                    mid_y = (ls_pt[1] + rs_pt[1]) // 2
                    cv2.putText(
                        annotated,
                        f"S:{shoulder_angle:.0f}deg",
                        (mid_x + 5, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        shoulder_color,
                        1,
                        cv2.LINE_AA,
                    )

                if is_turned:
                    if head_state.shoulder_turn_start < 0:
                        head_state.shoulder_turn_start = ts_sec
                    elapsed = ts_sec - head_state.shoulder_turn_start

                    if (
                        elapsed >= head_mod.SUSTAINED_SEC
                        and head_state.can_flag("shoulder_turn", ts_sec)
                    ):
                        head_state.shoulder_turn_flagged_at = ts_sec
                        head_stats["shoulder_turn"] += 1
                        head_mod.log_alert(
                            "SHOULDER TURN",
                            head_state.student_num,
                            ts_sec,
                            (
                                f"direction={turn_dir}, angle={shoulder_angle:.1f}°, "
                                f"sustained {elapsed:.1f}s"
                            ),
                            head_mod.TC.CYAN,
                        )
                        head_frame_events.append(
                            ("shoulder_turn", head_state.student_num)
                        )

                    if elapsed >= 1.0:
                        display["labels"].append(
                            f"SHOULDER {turn_dir} {shoulder_angle:.0f}deg ({elapsed:.1f}s)"
                        )
                        display["color"] = elevate_color(
                            display["color"], head_mod.COL_SHOULDER_TURN
                        )
                        if elapsed >= head_mod.SUSTAINED_SEC:
                            display["color"] = elevate_color(
                                display["color"], head_mod.COL_FLAGGED
                            )
                else:
                    head_state.shoulder_turn_start = -1.0

            # Pass 2: evaluate passing-papers interactions on the same tracks.
            evaluated_pairs = set()
            for tid_a in list(frame_kp_data.keys()):
                neighbors = pass_mod.find_row_neighbors(
                    tid_a, all_student_bboxes, passing_students
                )

                for tid_b, _ in neighbors:
                    pair_key = frozenset((tid_a, tid_b))
                    if pair_key in evaluated_pairs or tid_b not in frame_kp_data:
                        continue
                    evaluated_pairs.add(pair_key)

                    state_a = passing_students[tid_a]
                    state_b = passing_students[tid_b]
                    if abs(state_a.student_num - state_b.student_num) != 1:
                        continue

                    ps = get_pair_state(tid_a, tid_b)
                    kp_a_xy, kp_a_conf = frame_kp_data[tid_a]
                    kp_b_xy, kp_b_conf = frame_kp_data[tid_b]
                    bbox_a = all_student_bboxes[tid_a]
                    bbox_b = all_student_bboxes[tid_b]
                    center_a = (
                        (bbox_a[0] + bbox_a[2]) / 2,
                        (bbox_a[1] + bbox_a[3]) / 2,
                    )
                    center_b = (
                        (bbox_b[0] + bbox_b[2]) / 2,
                        (bbox_b[1] + bbox_b[3]) / 2,
                    )

                    avg_pair_h = (
                        (bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])
                    ) / 2.0
                    scaled_prox_px = (
                        pass_mod.WRIST_PROXIMITY_PX *
                        pass_mod._perspective_scale(avg_pair_h)
                    )

                    ps.frame_proximity = False
                    ps.frame_proximity_dist = 9999.0

                    prox_dist = pass_mod.compute_wrist_proximity(
                        kp_a_xy, kp_a_conf, kp_b_xy, kp_b_conf
                    )
                    ps.frame_proximity_dist = prox_dist
                    ps.frame_proximity = prox_dist < scaled_prox_px

                    if prox_dist < scaled_prox_px * 1.5:
                        best_pts = None
                        best_dist = 9999.0
                        for wrist_a, wrist_b in (
                            (
                                pass_mod._kp_pos(
                                    kp_a_xy, kp_a_conf, pass_mod.KP_RIGHT_WRIST
                                ),
                                pass_mod._kp_pos(
                                    kp_b_xy, kp_b_conf, pass_mod.KP_LEFT_WRIST
                                ),
                            ),
                            (
                                pass_mod._kp_pos(
                                    kp_a_xy, kp_a_conf, pass_mod.KP_LEFT_WRIST
                                ),
                                pass_mod._kp_pos(
                                    kp_b_xy, kp_b_conf, pass_mod.KP_RIGHT_WRIST
                                ),
                            ),
                        ):
                            if wrist_a and wrist_b:
                                dist = pass_mod._dist(wrist_a, wrist_b)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pts = (wrist_a, wrist_b)

                        if best_pts:
                            pt_a, pt_b = best_pts
                            line_color = (
                                head_mod.COL_FLAGGED
                                if ps.frame_proximity else pass_mod.COL_WARNING
                            )
                            cv2.line(
                                annotated,
                                (int(pt_a[0]), int(pt_a[1])),
                                (int(pt_b[0]), int(pt_b[1])),
                                line_color,
                                2,
                                cv2.LINE_AA,
                            )
                            mid_x = int((pt_a[0] + pt_b[0]) / 2)
                            mid_y = int((pt_a[1] + pt_b[1]) / 2)
                            head_mod.draw_label(
                                annotated, f"{prox_dist:.0f}px", mid_x, mid_y, line_color
                            )

                    if ps.frame_proximity:
                        if ps.interaction_start < 0:
                            ps.interaction_start = ts_sec
                        ps.last_proximity_time = ts_sec
                        ps.peak_proximity_dist = min(ps.peak_proximity_dist, prox_dist)

                        interaction_dur = ts_sec - ps.interaction_start
                        alert_dir = "RIGHT" if center_b[0] > center_a[0] else "LEFT"

                        for tid_src, neighbor_num in (
                            (tid_a, state_b.student_num),
                            (tid_b, state_a.student_num),
                        ):
                            display = per_student_display.get(tid_src)
                            if display is None:
                                continue
                            display["color"] = elevate_color(
                                display["color"], pass_mod.COL_WARNING
                            )
                            display["labels"].append(
                                f"PROX S#{neighbor_num} {prox_dist:.0f}px {interaction_dur:.1f}s"
                            )

                        cv2.line(
                            annotated,
                            (int(center_a[0]), int(center_a[1])),
                            (int(center_b[0]), int(center_b[1])),
                            pass_mod.COL_NEIGHBOR_LINE,
                            2,
                            cv2.LINE_AA,
                        )

                        if (
                            interaction_dur >= pass_mod.MIN_INTERACTION_SEC
                            and ps.can_flag(ts_sec)
                        ):
                            ps.last_flagged_at = ts_sec
                            state_a.total_alerts += 1
                            state_b.total_alerts += 1
                            passing_alert_total += 1

                            pass_mod.log_alert(
                                "PASSING PAPERS",
                                [state_a.student_num, state_b.student_num],
                                ts_sec,
                                (
                                    f"{alert_dir}, dur={interaction_dur:.1f}s, "
                                    f"closest={ps.peak_proximity_dist:.0f}px"
                                ),
                                pass_mod.TC.RED,
                            )
                            passing_frame_events.append(
                                (state_a.student_num, state_b.student_num, alert_dir)
                            )

                            for tid_src in (tid_a, tid_b):
                                display = per_student_display.get(tid_src)
                                if display is not None:
                                    display["color"] = elevate_color(
                                        display["color"], head_mod.COL_FLAGGED
                                    )

                            ps.reset_interaction()
                    elif ps.interaction_start > 0:
                        ps.reset_interaction()

            # Pass 3: draw student boxes and labels after all states are known.
            for i, det in enumerate(detections):
                tid = track_ids[i]
                if tid == -1 or tid not in assigned_tids:
                    continue

                bbox = det["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                display = per_student_display.get(
                    tid, {"color": head_mod.COL_NORMAL, "labels": []}
                )
                student_num = student_map[tid]

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), display["color"], 2
                )
                head_mod.draw_label(
                    annotated, f"Student #{student_num}", x1, y1 - 2, display["color"]
                )

                label_y = y1 + 18
                for label in display["labels"]:
                    head_mod.draw_label(
                        annotated, label, x1, label_y, display["color"]
                    )
                    label_y += 18

            elapsed_wall = time.perf_counter() - t_start
            actual_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0.0
            head_alert_total = sum(head_stats.values())
            tracked_count = len(frame_kp_data)
            has_warning = any(
                display["color"] in (
                    head_mod.COL_HEAD_TILT,
                    head_mod.COL_SHOULDER_TURN,
                    pass_mod.COL_WARNING,
                )
                for display in per_student_display.values()
            )
            hud_color = pass_mod.COL_HUD
            if head_alert_total > 0 or passing_alert_total > 0:
                hud_color = head_mod.COL_FLAGGED
            elif has_warning:
                hud_color = pass_mod.COL_WARNING

            hud_lines = [
                f"Frame: {frame_idx}/{total_frames} | Time: {head_mod.fmt_ts(ts_sec)}",
                f"Video FPS: {fps:.1f} | Processing FPS: {actual_fps:.1f}",
                (
                    f"Tracked: {tracked_count}/{len(student_map)} | "
                    f"Head A: {head_alert_total} | Passing A: {passing_alert_total} | "
                    f"Inf: {inference_ms:.0f}ms"
                ),
            ]
            for idx, line in enumerate(hud_lines):
                y_pos = 25 + idx * 28
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

            draw_fps_badge(annotated, actual_fps, hud_color)

            banner_y = height - 30
            for behavior, student_num in head_frame_events:
                text = f"ALERT: Student #{student_num} - {behavior.replace('_', ' ').upper()}"
                cv2.putText(
                    annotated, text, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, text, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, head_mod.COL_FLAGGED, 2, cv2.LINE_AA
                )
                banner_y -= 35

            for src_num, nbr_num, direction in passing_frame_events:
                text = (
                    f"ALERT: S#{src_num} & S#{nbr_num} PASSING PAPERS ({direction})"
                )
                cv2.putText(
                    annotated, text, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, text, (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, head_mod.COL_FLAGGED, 2, cv2.LINE_AA
                )
                banner_y -= 35

            ts_text = head_mod.fmt_ts(ts_sec)
            (ts_width, _), _ = cv2.getTextSize(
                ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.putText(
                annotated,
                ts_text,
                (width - ts_width - 10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                ts_text,
                (width - ts_width - 10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for behavior, student_num in head_frame_events:
                save_head_evidence(annotated, student_num, behavior, ts_sec)

            for src_num, nbr_num, _ in passing_frame_events:
                save_passing_evidence(annotated, [src_num, nbr_num], ts_sec)

            with _frame_lock:
                _latest_frame = annotated

            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                head_mod.log_info(
                    f"Progress: {pct:.1f}% ({frame_idx}/{total_frames}) | "
                    f"FPS: {actual_fps:.1f}"
                )

    except KeyboardInterrupt:
        head_mod.log_info("Interrupted by user.")

    elapsed = time.perf_counter() - t_start
    head_alert_total = sum(head_stats.values())

    print()
    print("=" * 74)
    print(f"  Summary: {Path(video_path).name}")
    print("-" * 74)
    print(f"  Frames processed : {frame_idx}")
    if elapsed > 0:
        print(f"  Average FPS      : {frame_idx / elapsed:.1f}")
    print(f"  Students tracked : {len(student_map)}")
    print(f"  Head alerts      : {head_alert_total}")
    for behavior, count in sorted(head_stats.items()):
        print(f"    {behavior:20s}: {count}")
    print(f"  Passing alerts   : {passing_alert_total}")
    for _, state in sorted(
        passing_students.items(), key=lambda item: item[1].student_num
    ):
        if state.total_alerts > 0:
            print(
                f"    Student #{state.student_num:2d} : "
                f"{state.total_alerts} passing events"
            )
    if head_alert_total > 0 or passing_alert_total > 0:
        print(f"  Evidence saved to: {EVIDENCE_DIR}")
    else:
        print("  No combined alerts triggered.")
    print("=" * 74)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AISENTINEL - Head Behavior + Passing Papers Detection "
            "(Pi + Hailo)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_head_behavior_passing_papers_pi.py
  python3 front_node_head_behavior_passing_papers_pi.py --port 9090
  python3 front_node_head_behavior_passing_papers_pi.py --model /path/to/yolo_pose_model.hef
        """,
    )
    parser.add_argument(
        "--model",
        default=str(POSE_MODEL_PATH),
        help=f"Path to pose HEF model (default: {POSE_MODEL_PATH})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Flask web server port (default: 8080)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Person detection confidence (default: 0.5)",
    )
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  AISENTINEL - Combined Pi Detection")
    print("  Pose model      : shared YOLOv8 pose HEF")
    print("  Detects         : head tilt | shoulder turn | passing papers")
    print("  Calibration flow: ROI -> assignment")
    print("  Overlay         : includes processing FPS")
    print("=" * 72)
    print()

    if not head_mod.HAILO_AVAILABLE or not pass_mod.HAILO_AVAILABLE:
        print(f"{head_mod.TC.RED}[ERROR] hailo_platform is required.{head_mod.TC.RESET}")
        print("Install: sudo apt install hailo-all")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"{head_mod.TC.RED}[ERROR] HEF model not found: {model_path}{head_mod.TC.RESET}")
        print("See POSE_MODEL_SETUP.md for download instructions.")
        sys.exit(1)

    head_mod.log_info("Opening file dialog...")
    video_path = pass_mod.select_video_dialog()
    if not video_path:
        head_mod.log_info("No video selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(video_path):
        print(f"{head_mod.TC.RED}[ERROR] File not found: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)
    head_mod.log_info(f"Selected: {video_path}")

    estimator = head_mod.HailoPoseEstimator(
        str(model_path),
        conf_threshold=args.confidence,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{head_mod.TC.RED}[ERROR] Cannot open video: {video_path}{head_mod.TC.RESET}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print(f"{head_mod.TC.RED}[ERROR] Cannot read first frame.{head_mod.TC.RESET}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_scale = min(1.0, 1280 / width) if width > 1280 else 1.0
    head_mod.log_info(f"Video resolution: {width}x{height}")

    head_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = pass_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        head_mod.log_info("Cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    head_mod.log_info("Running pose detection on first frame for student assignment...")
    first_detections = estimator.detect_pose(first_frame)
    first_detections = pass_mod.filter_detections_by_roi(first_detections, roi_polygon)

    tracker = pass_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = tracker.update(first_detections)

    roi_label = " (within ROI)" if roi_polygon is not None else ""
    head_mod.log_info(f"Detected {len(first_detections)} persons on first frame{roi_label}.")
    print()
    print(f"  {head_mod.TC.BOLD}Instructions:{head_mod.TC.RESET}")
    print("    1. Click on a person to select them (cyan highlight)")
    print("    2. Type the student number (digits)")
    print("    3. Press ENTER to assign")
    print("    4. Repeat for each student you want to monitor")
    print("    5. Passing-papers alerts need at least 2 consecutive student numbers")
    print("    6. Press S to start")
    print()

    student_map, baseline_yaw_map = head_mod.run_assignment_phase(
        first_frame, first_detections, first_track_ids, disp_scale
    )
    if student_map is None:
        cap.release()
        head_mod.log_info("Assignment cancelled. Exiting.")
        sys.exit(0)
    if len(student_map) == 0:
        cap.release()
        head_mod.log_info("No students assigned. Exiting.")
        sys.exit(0)
    if len(student_map) < 2:
        head_mod.log_info(
            "Only one student assigned. Head alerts will run, but passing-papers "
            "alerts need at least 2 students."
        )

    tracker.keep_only(set(student_map.keys()))
    head_mod.log_info(f"Tracker locked to {len(student_map)} assigned student(s).")

    if not FLASK_AVAILABLE:
        print(f"{head_mod.TC.RED}[ERROR] Flask is required for web streaming.{head_mod.TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    start_web_server(args.port)
    head_mod.log_info(f"Web stream at http://{get_local_ip()}:{args.port}")

    head_mod.log_info("Starting combined detection...")
    run_detection(
        cap,
        estimator,
        tracker,
        student_map,
        baseline_yaw_map,
        video_path,
        args.port,
        roi_polygon=roi_polygon,
    )
    cap.release()
    head_mod.log_info("Done!")


if __name__ == "__main__":
    main()
