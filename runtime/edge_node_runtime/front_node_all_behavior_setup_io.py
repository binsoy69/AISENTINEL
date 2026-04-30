#!/usr/bin/env python3
"""Saved setup profile helpers for front_node_all_behavior_pi.py."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SETUP_PROFILE_DIR = SCRIPT_DIR / "data" / "setup_profiles"
SETUP_PROFILE_SUFFIX = "_all_behavior_setup.json"
SETUP_PROFILE_VERSION = 1


def default_setup_profile_path(video_path):
    """Return the default JSON profile path for a selected video."""
    video_stem = Path(video_path).stem or "session"
    return SETUP_PROFILE_DIR / f"{video_stem}{SETUP_PROFILE_SUFFIX}"


def _serialize_bbox(bbox):
    return [int(round(v)) for v in bbox]


def _deserialize_bbox(bbox):
    return tuple(int(v) for v in bbox)


def _serialize_line(line):
    if line is None:
        return None
    arr = np.asarray(line, dtype=np.int32).reshape(2, 2)
    return [[int(arr[0][0]), int(arr[0][1])], [int(arr[1][0]), int(arr[1][1])]]


def _deserialize_line(line):
    if line is None:
        return None
    return np.asarray(line, dtype=np.int32).reshape(2, 2)


def _serialize_roi_polygon(roi_polygon):
    if roi_polygon is None:
        return None
    pts = np.asarray(roi_polygon, dtype=np.int32).reshape(-1, 2)
    return [[int(x), int(y)] for x, y in pts]


def profile_to_roi_polygon(profile):
    """Convert the stored ROI payload back to a numpy polygon or None."""
    roi_polygon = profile.get("roi_polygon")
    if not roi_polygon:
        return None
    return np.asarray(roi_polygon, dtype=np.int32).reshape(-1, 2)


def save_setup_profile(path, video_path, frame_shape, roi_polygon, assigned_students,
                       baseline_yaw_map, student_lines):
    """Persist a calibration profile to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    students_payload = []
    for idx, student in enumerate(assigned_students):
        track_id = int(student["track_id"])
        line = student_lines[idx] if idx < len(student_lines) else None
        students_payload.append(
            {
                "student_num": int(student["student_num"]),
                "track_id": track_id,
                "bbox": _serialize_bbox(student["bbox"]),
                "baseline_yaw": float(baseline_yaw_map.get(track_id, 0.0)),
                "desk_line": _serialize_line(line),
            }
        )

    data = {
        "profile_version": SETUP_PROFILE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": Path(video_path).name,
        "video_path": str(video_path),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "roi_polygon": _serialize_roi_polygon(roi_polygon),
        "students": students_payload,
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_setup_profile(path):
    """Load a calibration profile JSON payload."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("profile_version", 0))
    if version != SETUP_PROFILE_VERSION:
        raise ValueError(
            f"Unsupported setup profile version {version} in {path.name}"
        )
    return data


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def _bbox_area(bbox):
    w, h = _bbox_size(bbox)
    return w * h


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _match_score(saved_bbox, current_bbox):
    iou = _bbox_iou(saved_bbox, current_bbox)

    saved_cx, saved_cy = _bbox_center(saved_bbox)
    current_cx, current_cy = _bbox_center(current_bbox)
    center_dist = math.hypot(current_cx - saved_cx, current_cy - saved_cy)

    saved_w, saved_h = _bbox_size(saved_bbox)
    saved_diag = max(1.0, math.hypot(saved_w, saved_h))
    center_norm = center_dist / saved_diag

    saved_area = _bbox_area(saved_bbox)
    current_area = _bbox_area(current_bbox)
    area_ratio = current_area / max(saved_area, 1.0)

    if area_ratio < 0.35 or area_ratio > 2.85:
        return None
    if iou < 0.05 and center_norm > 0.80:
        return None

    return iou * 5.0 + max(0.0, 1.0 - center_norm)


def build_runtime_setup_from_profile(profile, detections, track_ids):
    """Map saved calibration data onto fresh first-frame detections."""
    saved_students = sorted(
        profile.get("students", []),
        key=lambda item: int(item["student_num"]),
    )

    current_students = []
    for idx, det in enumerate(detections):
        tid = int(track_ids[idx])
        if tid == -1:
            continue
        current_students.append(
            {
                "track_id": tid,
                "bbox": _deserialize_bbox(det["bbox"]),
            }
        )

    candidate_pairs = []
    for saved_idx, saved_student in enumerate(saved_students):
        saved_bbox = _deserialize_bbox(saved_student["bbox"])
        for current_idx, current_student in enumerate(current_students):
            score = _match_score(saved_bbox, current_student["bbox"])
            if score is not None:
                candidate_pairs.append((score, saved_idx, current_idx))

    candidate_pairs.sort(reverse=True)
    matched_saved = set()
    matched_current = set()
    current_by_saved = {}

    for score, saved_idx, current_idx in candidate_pairs:
        if saved_idx in matched_saved or current_idx in matched_current:
            continue
        matched_saved.add(saved_idx)
        matched_current.add(current_idx)
        current_by_saved[saved_idx] = current_students[current_idx]

    student_map = {}
    baseline_yaw_map = {}
    assigned_students = []
    student_lines = []
    matched_student_nums = []
    unmatched_student_nums = []

    for saved_idx, saved_student in enumerate(saved_students):
        student_num = int(saved_student["student_num"])
        current_student = current_by_saved.get(saved_idx)
        if current_student is None:
            unmatched_student_nums.append(student_num)
            continue

        track_id = int(current_student["track_id"])
        student_map[track_id] = student_num
        baseline_yaw_map[track_id] = float(saved_student.get("baseline_yaw", 0.0))
        assigned_students.append(
            {
                "track_id": track_id,
                "student_num": student_num,
                "bbox": current_student["bbox"],
            }
        )
        student_lines.append(_deserialize_line(saved_student.get("desk_line")))
        matched_student_nums.append(student_num)

    return {
        "student_map": student_map,
        "baseline_yaw_map": baseline_yaw_map,
        "assigned_students": assigned_students,
        "student_lines": student_lines,
        "matched_student_nums": matched_student_nums,
        "unmatched_student_nums": unmatched_student_nums,
        "saved_student_count": len(saved_students),
    }
