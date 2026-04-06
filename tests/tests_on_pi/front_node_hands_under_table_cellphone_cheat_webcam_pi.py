#!/usr/bin/env python3
"""
Hands Under Table + Phone / Cheat Sheet Detection - Raspberry Pi + Hailo AI HAT
===============================================================================
Webcam variant of front_node_hands_under_table_cellphone_cheat_pi.py.

This script keeps the same calibration flow as the video-based test:
  1. Open a USB webcam
  2. Grab a calibration frame
  3. ROI calibration: draw a polygon boundary (limits tracking area)
  4. Click detected persons to assign student numbers
  5. Desk ROI calibration: draw polygon ROIs for each desk
  6. Table-edge calibration: draw one 2-point line per desk near the student
  7. Re-lock student IDs from a fresh live frame
  8. Web stream starts at http://<pi-ip>:8080 with live annotations
  9. Console alerts + evidence screenshots saved to ./evidence_combined/

It reuses the same models and logic:
  - yolov8s_pose.hef for person detection / tracking
  - sentinel-yolo11n-min.hef for hand detection
  - sentinel-yolov11n_new.hef for phone / cheat_sheet detection
"""

import sys
import os
import time
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import front_node_hands_under_table_pi as hands_mod
import front_node_cellphone_cheat_pi as obj_mod
import front_node_hands_under_table_cellphone_cheat_pi as combined_mod


POSE_MODEL_PATH = combined_mod.POSE_MODEL_PATH
HAND_MODEL_PATH = combined_mod.HAND_MODEL_PATH
OBJECT_MODEL_PATH = combined_mod.OBJECT_MODEL_PATH

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_WARMUP_FRAMES = 12
DEFAULT_MAX_CAMERAS = 10
MAX_CAMERA_READ_FAILURES = 20
OPEN_CAMERA_READ_ATTEMPTS = 1
CAMERA_OPEN_TIMEOUT_MS = 1000
CAMERA_READ_TIMEOUT_MS = 1000


def configure_camera(cap, width, height, use_mjpg=True):
    """Apply live-capture settings, optionally requesting MJPG."""
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAMERA_OPEN_TIMEOUT_MS)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAMERA_READ_TIMEOUT_MS)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def get_video_device_path(camera_index):
    """Return /dev/videoN as a Path."""
    return Path(f"/dev/video{camera_index}")


def is_capture_device(camera_index):
    """Return True only for existing video nodes that advertise capture capability."""
    device_path = get_video_device_path(camera_index)
    if not device_path.exists():
        return False

    v4l2_ctl = shutil.which("v4l2-ctl")
    if not v4l2_ctl:
        return True

    try:
        proc = subprocess.run(
            [v4l2_ctl, "-D", "-d", str(device_path)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return True

    caps_text = f"{proc.stdout}\n{proc.stderr}".lower()
    return (
        "video capture" in caps_text or
        "video capture multiplanar" in caps_text
    )


def list_capture_camera_indices(max_cameras):
    """List valid capture-device indices under /dev/video*."""
    return [
        camera_index
        for camera_index in range(max_cameras)
        if is_capture_device(camera_index)
    ]


def find_usb_camera(width, height, max_cameras):
    """Return the first camera index that opens and yields a frame."""
    capture_indices = list_capture_camera_indices(max_cameras)
    if not capture_indices:
        return None, None, None

    if len(capture_indices) == 1:
        camera_index = capture_indices[0]
        cap, backend_name, config_name = open_camera_with_fallbacks(
            camera_index, width, height, require_frame=False
        )
        if cap is not None:
            hands_mod.log_info(
                f"Auto-selected /dev/video{camera_index} via {backend_name} "
                f"({config_name})."
            )
            return cap, camera_index, backend_name

    for camera_index in capture_indices:
        cap, backend_name, config_name = open_camera_with_fallbacks(
            camera_index, width, height, require_frame=True
        )
        if cap is None:
            continue

        hands_mod.log_info(
            f"Auto-selected /dev/video{camera_index} via {backend_name} "
            f"({config_name})."
        )
        return cap, camera_index, backend_name

    fallback_index = capture_indices[0]
    cap, backend_name, config_name = open_camera_with_fallbacks(
        fallback_index, width, height, require_frame=False
    )
    if cap is not None:
        hands_mod.log_info(
            f"Falling back to /dev/video{fallback_index} via {backend_name} "
            f"({config_name}) without startup frame validation."
        )
        return cap, fallback_index, backend_name

    return None, None, None


def open_webcam(camera_index, width, height, max_cameras):
    """Open the requested camera or auto-discover one."""
    if camera_index is None:
        return find_usb_camera(width, height, max_cameras)

    cap, backend_name, _config_name = open_camera_with_fallbacks(
        camera_index, width, height, require_frame=False
    )
    if cap is None:
        return None, None, None

    return cap, camera_index, backend_name


def read_latest_frame(cap, attempts=30, pause_sec=0.04):
    """Read a recent frame from the camera, allowing brief warm-up/retry."""
    frame = None
    good_frames = 0

    for _ in range(max(1, attempts)):
        ret, current = cap.read()
        if ret and current is not None:
            frame = current
            good_frames += 1
            if good_frames >= 2:
                break
        else:
            good_frames = 0
        time.sleep(pause_sec)

    return frame


def open_single_camera(camera_index, backend_name, backend_id, width, height,
                       use_mjpg, require_frame):
    """Open one camera/backend/config combination and wait for a valid frame."""
    if backend_id is None:
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index, backend_id)

    if not cap.isOpened():
        cap.release()
        return None

    configure_camera(cap, width, height, use_mjpg=use_mjpg)
    if require_frame:
        frame = read_latest_frame(
            cap,
            attempts=OPEN_CAMERA_READ_ATTEMPTS,
            pause_sec=0.05,
        )
        if frame is None:
            cap.release()
            return None

    return cap


def open_camera_with_fallbacks(camera_index, width, height, require_frame):
    """Open a camera with multiple backend/format fallbacks."""
    device_path = get_video_device_path(camera_index)
    if not device_path.exists():
        return None, None, None

    if not is_capture_device(camera_index):
        hands_mod.log_info(f"Skipping {device_path} because it is not a capture device.")
        return None, None, None

    attempts = [
        ("V4L2", cv2.CAP_V4L2, True, "MJPG"),
        ("V4L2", cv2.CAP_V4L2, False, "native"),
        ("default", None, True, "MJPG"),
        ("default", None, False, "native"),
    ]

    for backend_name, backend_id, use_mjpg, config_name in attempts:
        cap = open_single_camera(
            camera_index,
            backend_name,
            backend_id,
            width,
            height,
            use_mjpg=use_mjpg,
            require_frame=require_frame,
        )
        if cap is not None:
            return cap, backend_name, config_name

    return None, None, None


def build_desk_student_numbers(first_person_dets, first_track_ids, student_map,
                               desk_polygons, img_shape):
    """Map each desk ROI to the assigned student number from the calibration frame."""
    desk_candidates = defaultdict(list)

    for det, track_id in zip(first_person_dets, first_track_ids):
        if track_id not in student_map:
            continue

        desk_idx, area = hands_mod.find_desk_for_student(
            det["bbox"], desk_polygons, img_shape
        )
        if desk_idx is None or area <= 0:
            hands_mod.log_info(
                f"Assigned Student #{student_map[track_id]} does not overlap a desk ROI."
            )
            continue

        desk_candidates[desk_idx].append((area, student_map[track_id]))

    desk_student_numbers = {}
    for desk_idx, candidates in desk_candidates.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        desk_student_numbers[desk_idx] = candidates[0][1]

        if len(candidates) > 1:
            kept_student = candidates[0][1]
            hands_mod.log_info(
                f"Desk #{desk_idx + 1} had multiple assigned students in calibration; "
                f"keeping Student #{kept_student}."
            )

    return desk_student_numbers


def refresh_live_student_map(cap, person_detector, desk_polygons,
                             desk_student_numbers, roi_polygon,
                             max_attempts=20):
    """Create a fresh tracker/student map from the live webcam feed."""
    expected_students = set(desk_student_numbers.values())

    for attempt in range(1, max_attempts + 1):
        frame = read_latest_frame(cap, attempts=1, pause_sec=0.02)
        if frame is None:
            continue

        current_person_dets = person_detector.detect_persons(frame)
        current_person_dets = hands_mod.filter_detections_by_roi(
            current_person_dets, roi_polygon
        )

        if not current_person_dets:
            if attempt == 1 or attempt == max_attempts or attempt % 5 == 0:
                hands_mod.log_info(
                    f"Live refresh attempt {attempt}/{max_attempts}: no persons detected yet."
                )
            continue

        tracker = obj_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
        current_track_ids = tracker.update(current_person_dets)

        desk_matches = {}
        for det, track_id in zip(current_person_dets, current_track_ids):
            desk_idx, area = hands_mod.find_desk_for_student(
                det["bbox"], desk_polygons, frame.shape[:2]
            )
            if desk_idx is None or area <= 0:
                continue
            if desk_idx not in desk_student_numbers:
                continue

            best = desk_matches.get(desk_idx)
            if best is None or area > best[1]:
                desk_matches[desk_idx] = (track_id, area)

        refreshed_student_map = {
            track_id: desk_student_numbers[desk_idx]
            for desk_idx, (track_id, _area) in desk_matches.items()
        }

        if not refreshed_student_map:
            if attempt == 1 or attempt == max_attempts or attempt % 5 == 0:
                hands_mod.log_info(
                    f"Live refresh attempt {attempt}/{max_attempts}: "
                    "students detected, but none matched the desk ROIs."
                )
            continue

        matched_students = set(refreshed_student_map.values())
        missing_students = sorted(expected_students - matched_students)

        hands_mod.log_info(
            f"Live refresh locked {len(refreshed_student_map)} student(s) from the webcam."
        )
        for track_id, student_num in sorted(refreshed_student_map.items(), key=lambda x: x[1]):
            hands_mod.log_info(f"  Student #{student_num} -> Track ID {track_id}")

        if missing_students:
            hands_mod.log_info(
                "Students not visible during live refresh: "
                + ", ".join(f"#{student_num}" for student_num in missing_students)
            )

        return frame, tracker, refreshed_student_map

    return None, None, {}


def update_web_stream(frame):
    """Push the latest annotated frame into the shared Flask stream state."""
    with combined_mod._frame_lock:
        combined_mod._latest_frame = frame


def run_detection_webcam(cap, person_detector, hand_detector, object_detector,
                         tracker, student_map, desk_polygons, desk_edge_lines,
                         camera_index, port, roi_polygon=None):
    """Run the combined detection loop on a live webcam feed."""
    source_name = f"webcam{camera_index}"
    camera_label = f"/dev/video{camera_index}"
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if camera_fps <= 0 or camera_fps > 120:
        camera_fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)

    assigned_tids = set(student_map.keys())
    desk_states = [hands_mod.DeskState(i) for i in range(len(desk_polygons))]
    object_cooldowns = defaultdict(lambda: -999.0)
    object_stats = defaultdict(int)

    print()
    print("=" * 72)
    local_ip = combined_mod.get_local_ip()
    print("  AISENTINEL - Combined Pi Detection (Webcam)")
    print(f"  Camera       : {camera_label}")
    print(f"  Resolution   : {w}x{h} | Camera FPS: {camera_fps:.1f}")
    print(f"  Students     : {len(student_map)} live-locked")
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
    print(f"  Evidence     : {combined_mod.EVIDENCE_DIR}")
    print(f"  Web stream   : http://{local_ip}:{port}")
    print("=" * 72)
    print()

    frame_idx = 0
    hand_alert_total = 0
    hand_warning_total = 0
    object_alert_total = 0
    loop_started_at = time.perf_counter()
    consecutive_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 5 == 0:
                    hands_mod.log_info(
                        f"Camera read failed ({consecutive_failures}/{MAX_CAMERA_READ_FAILURES})."
                    )
                if consecutive_failures >= MAX_CAMERA_READ_FAILURES:
                    hands_mod.log_info("Too many camera read failures. Stopping.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            frame_idx += 1
            ts_sec = time.perf_counter() - loop_started_at
            raw_frame = frame.copy()

            t0 = time.perf_counter()

            person_dets = person_detector.detect_persons(frame)
            person_dets = hands_mod.filter_detections_by_roi(person_dets, roi_polygon)

            hand_raw = hand_detector.detect(frame)
            hand_dets = [
                det for det in hand_raw if det["class_name"] == hands_mod.CLASS_HAND
            ]

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
                    annotated,
                    f"hand {det['confidence']:.0%}",
                    x1,
                    y1 - 2,
                    hands_mod.COL_HAND,
                )

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

            elapsed_wall = time.perf_counter() - loop_started_at
            actual_fps = frame_idx / elapsed_wall if elapsed_wall > 0 else 0.0
            ts_text = hands_mod.fmt_ts(ts_sec)

            hud_lines = [
                f"Camera: {camera_label} | Frame: {frame_idx} | Time: {ts_text}",
                f"Camera FPS: {camera_fps:.1f} | Processing FPS: {actual_fps:.1f}",
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
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hands_mod.COL_ALERT,
                    2,
                    cv2.LINE_AA,
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
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hands_mod.COL_ALERT,
                    2,
                    cv2.LINE_AA,
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
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    txt,
                    (10, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hands_mod.COL_WARNING,
                    2,
                    cv2.LINE_AA,
                )
                banner_y -= 35

            (tw, _), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(
                annotated,
                ts_text,
                (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                ts_text,
                (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for desk_idx in frame_hand_alerts:
                tid = desk_states[desk_idx].assigned_student_id or 0
                student_num = student_map.get(tid, tid)
                combined_mod.save_hand_evidence(
                    annotated, raw_frame, source_name, desk_idx, student_num, ts_sec
                )

            for event in frame_object_alerts:
                combined_mod.save_object_evidence(
                    annotated,
                    raw_frame,
                    event["student_num"],
                    event["class_name"],
                    event["confidence"],
                    ts_sec,
                )

            update_web_stream(annotated)

            if frame_idx % 300 == 0:
                hands_mod.log_info(
                    f"Live progress: {frame_idx} frames | "
                    f"Runtime: {hands_mod.fmt_ts(ts_sec)} | FPS: {actual_fps:.1f}"
                )

    except KeyboardInterrupt:
        hands_mod.log_info("Stopped by user.")

    elapsed = time.perf_counter() - loop_started_at
    print()
    print("=" * 72)
    print(f"  Summary: {source_name}")
    print("-" * 72)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Runtime          : {hands_mod.fmt_ts(elapsed)}")
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
        print(f"  Evidence saved to: {combined_mod.EVIDENCE_DIR}")
    elif hand_warning_total == 0:
        print("  No combined alerts triggered.")
    print("=" * 72)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AISENTINEL - Hands Under Table + Phone / Cheat Sheet Detection "
            "(Pi + Hailo, Webcam)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 front_node_hands_under_table_cellphone_cheat_webcam_pi.py
  python3 front_node_hands_under_table_cellphone_cheat_webcam_pi.py --camera 0
  python3 front_node_hands_under_table_cellphone_cheat_webcam_pi.py --camera 1 --port 9090
  python3 front_node_hands_under_table_cellphone_cheat_webcam_pi.py --object-confidence 0.4
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
        "--camera",
        type=int,
        default=None,
        help="USB camera index. Default: auto-detect the first working webcam.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Requested webcam width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Requested webcam height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=f"Frames to discard before grabbing the calibration frame (default: {DEFAULT_WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=DEFAULT_MAX_CAMERAS,
        help=f"How many camera indexes to scan during auto-detect (default: {DEFAULT_MAX_CAMERAS})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Flask web server port (default: 8080)",
    )
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  AISENTINEL - Combined Pi Detection (Webcam)")
    print("  Person detection : pose model (IoU tracked)")
    print("  Hand detection   : sentinel-yolo11n-min.hef (hand class)")
    print("  Object detection : sentinel-yolov11n_new.hef (phone + cheat_sheet)")
    print("  Calibration flow : ROI -> assignment -> desk polygons -> table-edge lines")
    print("  Source           : live webcam")
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

    hands_mod.log_info("Opening webcam...")
    cap, camera_index, backend_name = open_webcam(
        args.camera,
        args.width,
        args.height,
        args.max_cameras,
    )
    if cap is None or camera_index is None:
        print(f"{hands_mod.TC.RED}[ERROR] Cannot open a webcam.{hands_mod.TC.RESET}")
        print("Try: ls /dev/video* && v4l2-ctl --list-devices")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands_mod.log_info(
        f"Connected to /dev/video{camera_index} using {backend_name} backend "
        f"at {actual_w}x{actual_h}."
    )

    hands_mod.log_info("Warming up the webcam...")
    first_frame = read_latest_frame(cap, attempts=args.warmup_frames, pause_sec=0.04)
    if first_frame is None:
        cap.release()
        print(f"{hands_mod.TC.RED}[ERROR] Cannot read a calibration frame from the webcam.{hands_mod.TC.RESET}")
        sys.exit(1)

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

    w = first_frame.shape[1]
    disp_scale = min(1.0, 1280 / w) if w > 1280 else 1.0
    hands_mod.log_info(f"Calibration frame resolution: {first_frame.shape[1]}x{first_frame.shape[0]}")

    hands_mod.log_info("Draw ROI boundary to limit tracking area (or press S to skip).")
    roi_result = hands_mod.calibrate_roi(first_frame, disp_scale)
    if isinstance(roi_result, str) and roi_result == "CANCEL":
        cap.release()
        hands_mod.log_info("Cancelled. Exiting.")
        sys.exit(0)
    roi_polygon = roi_result if isinstance(roi_result, np.ndarray) else None

    hands_mod.log_info("Running pose detection on the calibration frame for student assignment...")
    first_person_dets = person_detector.detect_persons(first_frame)
    first_person_dets = hands_mod.filter_detections_by_roi(
        first_person_dets, roi_polygon
    )

    assignment_tracker = obj_mod.IoUTracker(iou_threshold=0.3, max_lost=60)
    first_track_ids = assignment_tracker.update(first_person_dets)

    roi_label = " (within ROI)" if roi_polygon is not None else ""
    hands_mod.log_info(
        f"Detected {len(first_person_dets)} persons on the calibration frame{roi_label}."
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
    combined_mod.describe_first_frame_context(first_hand_dets, first_obj_dets)

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

    hands_mod.log_info("Now draw polygon ROIs for each desk on the calibration frame.")
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

    desk_student_numbers = build_desk_student_numbers(
        first_person_dets,
        first_track_ids,
        student_map,
        desk_polygons,
        first_frame.shape[:2],
    )
    if not desk_student_numbers:
        cap.release()
        hands_mod.log_info(
            "Assigned students did not map onto the desk ROIs. Exiting."
        )
        sys.exit(0)

    hands_mod.log_info("Desk-to-student mapping from calibration:")
    for desk_idx, student_num in sorted(desk_student_numbers.items()):
        hands_mod.log_info(f"  Desk #{desk_idx + 1} -> Student #{student_num}")

    hands_mod.log_info("Refreshing student locks from a live webcam frame...")
    live_frame, tracker, live_student_map = refresh_live_student_map(
        cap,
        person_detector,
        desk_polygons,
        desk_student_numbers,
        roi_polygon,
    )
    if tracker is None or not live_student_map:
        cap.release()
        hands_mod.log_info(
            "Could not re-lock students from the live webcam feed. "
            "Make sure the assigned students are visible and seated."
        )
        sys.exit(0)

    tracker.keep_only(set(live_student_map.keys()))
    hands_mod.log_info(f"Tracker locked to {len(live_student_map)} live student(s).")

    update_web_stream(live_frame)

    if not combined_mod.FLASK_AVAILABLE:
        cap.release()
        print(f"{hands_mod.TC.RED}[ERROR] Flask is required for web streaming.{hands_mod.TC.RESET}")
        print("Install: pip install flask")
        sys.exit(1)

    combined_mod.start_web_server(args.port)
    local_ip = combined_mod.get_local_ip()
    hands_mod.log_info(f"Web stream at http://{local_ip}:{args.port}")

    hands_mod.log_info("Starting combined live detection...")
    run_detection_webcam(
        cap,
        person_detector,
        hand_detector,
        object_detector,
        tracker,
        live_student_map,
        desk_polygons,
        desk_edge_lines,
        camera_index,
        args.port,
        roi_polygon=roi_polygon,
    )

    cap.release()
    cv2.destroyAllWindows()
    hands_mod.log_info("Done!")


if __name__ == "__main__":
    main()
