"""Compatibility surface for the packaged edge-node runtime session runner."""

from __future__ import annotations

from edge_node_runtime import runtime_config as front_runtime_config
from edge_node_runtime import runtime_support as front_runtime_support
from edge_node_runtime.capture import resolve_calibration_path as _resolve_calibration_path
from edge_node_runtime.evidence import normalize_front_runtime_incident as _normalize_front_runtime_incident
from edge_node_runtime.session_runner import (
    load_front_runtime_context,
    run_front_runtime_session,
)

__all__ = [
    "_normalize_front_runtime_incident",
    "_resolve_calibration_path",
    "front_runtime_config",
    "front_runtime_support",
    "load_front_runtime_context",
    "run_front_runtime_session",
]
