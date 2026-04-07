"""Authentication helpers for the node agent."""

from __future__ import annotations

from functools import wraps

from flask import current_app, jsonify, request


def agent_api_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        config = current_app.config["NODE_CONFIG"]
        if request.headers.get("X-Node-Id", "").strip() != config.node_id:
            return jsonify({"error": "node_id header mismatch"}), 401
        if request.headers.get("X-Api-Key", "") != config.api_key:
            return jsonify({"error": "api key mismatch"}), 401
        return view_func(*args, **kwargs)

    return wrapper
