# AISENTINEL Central Dashboard Runtime

This folder contains a standalone multi-node monitoring stack that does not
import or modify the existing `runtime/front_node_pi` runtime.

## Layout

- `central_service/`: laptop-hosted Flask app, SQLite persistence, evidence
  storage, browser UI, and MJPEG feed proxy.
- `node_agent/`: Raspberry Pi agent that serves local feeds, runs local frame
  processing, stores evidence, and syncs incidents/evidence to central.
- `shared/`: DTOs, JSON helpers, and HTTP client helpers shared by both apps.
- `scripts/`: entrypoint scripts for running the central service and node
  agents.
- `data/`: runtime-created databases, evidence, and queue state.
- `tests/`: unit and integration tests for the new stack.

## Default Runtime Model

- The central service owns the shared exam session and browser-facing dashboard.
- Nodes self-register with the central service using per-node API keys.
- Each node keeps detection local and exposes both raw and annotated preview
  streams.
- Evidence is saved locally first, then synced to the central service with
  retry-on-failure queueing.

## Current Detector

The new node runtime is intentionally isolated from the legacy front-node code.
Its built-in detector is a local motion-anomaly pipeline that drives the
session, feed, evidence, and synchronization architecture end to end.

The detector backend is pluggable. A richer model-backed detector can be added
later without changing the central service contracts.
