"""Tracking factory helpers for edge-node sessions."""

from __future__ import annotations


def create_tracker(runtime_cfg, combined_mod):
    return combined_mod.ReacquiringLockedIoUTracker(
        iou_threshold=runtime_cfg.tracking.iou_threshold,
        max_lost=runtime_cfg.tracking.max_lost,
    )

