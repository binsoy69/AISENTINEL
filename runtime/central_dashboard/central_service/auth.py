"""Authentication helpers for the central dashboard service."""

from __future__ import annotations

from functools import wraps

from flask import current_app, jsonify, redirect, request, session, url_for


def browser_password_matches(expected_password: str, submitted_password: str) -> bool:
    return expected_password == submitted_password


def is_browser_authenticated() -> bool:
    auth_cfg = current_app.config["CENTRAL_CONFIG"].browser_auth
    return (
        session.get("authenticated") is True
        and session.get("username") == auth_cfg.username
    )


def browser_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not is_browser_authenticated():
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapper


def browser_api_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not is_browser_authenticated():
            return jsonify({"error": "authentication required"}), 401
        return view_func(*args, **kwargs)

    return wrapper


def validate_node_headers(node_id: str | None, api_key: str | None) -> bool:
    config = current_app.config["CENTRAL_CONFIG"]
    if not node_id or not api_key:
        return False
    known = config.known_nodes.get(str(node_id).strip())
    if known is None:
        return False
    return known.api_key == str(api_key)


def node_api_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        node_id = request.headers.get("X-Node-Id", "").strip()
        api_key = request.headers.get("X-Api-Key", "")
        if not validate_node_headers(node_id, api_key):
            return jsonify({"error": "node authentication failed"}), 401
        request.node_id = node_id
        return view_func(*args, **kwargs)

    return wrapper
