"""Flask application for the standalone central dashboard service."""

from __future__ import annotations

import atexit
from datetime import timedelta
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from central_dashboard.central_service.auth import (
    browser_api_login_required,
    browser_login_required,
    browser_password_matches,
    is_browser_authenticated,
    node_api_required,
)
from central_dashboard.central_service.config import CentralServiceConfig
from central_dashboard.central_service.db import connect_db, init_db
from central_dashboard.central_service.proxy import relay_stream_chunks
from central_dashboard.central_service.repositories import CentralRepository
from central_dashboard.central_service.services import CentralServiceManager
from central_dashboard.shared.dto import NodeDescriptor, NodeHeartbeat


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _safe_next_path(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if value.startswith("/") and not value.startswith("//"):
        return value
    return ""


def create_app(config: CentralServiceConfig, *, http_client=None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["CENTRAL_CONFIG"] = config
    app.secret_key = config.browser_auth.secret_key
    app.permanent_session_lifetime = timedelta(minutes=config.browser_auth.session_ttl_minutes)

    connection = connect_db(config.db_path)
    init_db(connection)
    repository = CentralRepository(connection)
    manager = CentralServiceManager(config, repository, http_client=http_client)
    app.extensions["central_repository"] = repository
    app.extensions["central_manager"] = manager
    app.extensions["central_connection"] = connection
    app.extensions["central_shutdown_done"] = False

    manager.reset_runtime_sessions_on_startup()

    def _shutdown_active_session() -> None:
        if app.extensions.get("central_shutdown_done"):
            return
        app.extensions["central_shutdown_done"] = True
        try:
            manager.shutdown_active_session()
        except Exception:
            pass

    app.extensions["central_shutdown_handler"] = _shutdown_active_session
    atexit.register(_shutdown_active_session)

    @app.route("/")
    def index():
        if is_browser_authenticated():
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error_message = ""
        next_path = _safe_next_path(request.args.get("next", ""))
        if is_browser_authenticated():
            return redirect(next_path or url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            next_path = _safe_next_path(request.form.get("next", ""))
            auth_cfg = config.browser_auth
            if username == auth_cfg.username and browser_password_matches(auth_cfg.password, password):
                session.clear()
                session.permanent = True
                session["authenticated"] = True
                session["username"] = username
                return redirect(next_path or url_for("dashboard"))
            error_message = "Invalid dashboard credentials."

        return render_template("login.html", error_message=error_message, next_path=next_path)

    @app.route("/logout")
    @browser_login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @browser_login_required
    def dashboard():
        return render_template("dashboard.html", bootstrap=manager.dashboard_snapshot())

    @app.route("/api/v1/dashboard")
    @browser_api_login_required
    def dashboard_api():
        return jsonify(manager.dashboard_snapshot())

    @app.route("/api/v1/sessions", methods=["POST"])
    @browser_api_login_required
    def create_session_api():
        payload = request.get_json(silent=True) or {}
        result = manager.create_session(payload)
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/sessions/current/clear", methods=["POST"])
    @browser_api_login_required
    def clear_current_session_api():
        result = manager.clear_current_session()
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/sessions/<session_id>", methods=["DELETE"])
    @browser_api_login_required
    def delete_session_api(session_id: str):
        result = manager.delete_session(session_id)
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/sessions/subjects/delete", methods=["POST"])
    @browser_api_login_required
    def delete_subject_sessions_api():
        payload = request.get_json(silent=True) or {}
        result = manager.delete_subject_sessions(payload.get("subject_code"))
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/sessions/<session_id>/<action>", methods=["POST"])
    @browser_api_login_required
    def session_action_api(session_id: str, action: str):
        if action not in {"start", "stop", "restart"}:
            return jsonify({"error": "Unsupported session action."}), 404
        result = manager.dispatch_session_command(session_id, action)
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/incidents/<incident_id>/review", methods=["POST"])
    @browser_api_login_required
    def review_incident_api(incident_id: str):
        payload = request.get_json(silent=True) or {}
        result = manager.update_review_status(incident_id, payload.get("review_status"))
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/incidents/clear", methods=["POST"])
    @browser_api_login_required
    def clear_incidents_api():
        payload = request.get_json(silent=True) or {}
        result = manager.clear_incidents(payload.get("session_id"))
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/nodes/register", methods=["POST"])
    @node_api_required
    def register_node_api():
        payload = request.get_json(silent=True) or {}
        descriptor = NodeDescriptor.from_dict(payload)
        if descriptor.node_id != request.node_id:
            return jsonify({"error": "node_id does not match authenticated header"}), 400
        return jsonify(manager.register_node(descriptor))

    @app.route("/api/v1/nodes/heartbeat", methods=["POST"])
    @node_api_required
    def node_heartbeat_api():
        payload = request.get_json(silent=True) or {}
        heartbeat = NodeHeartbeat.from_dict(payload)
        if heartbeat.node_id != request.node_id:
            return jsonify({"error": "node_id does not match authenticated header"}), 400
        return jsonify(manager.record_heartbeat(heartbeat))

    @app.route("/api/v1/incidents", methods=["POST"])
    @node_api_required
    def incidents_api():
        payload = request.get_json(silent=True) or {}
        if str(payload.get("node_id", "")).strip() != request.node_id:
            return jsonify({"error": "node_id does not match authenticated header"}), 400
        result = manager.upsert_incident(payload)
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/evidence/upload", methods=["POST"])
    @node_api_required
    def upload_evidence_api():
        payload = request.get_json(silent=True) or {}
        if str(payload.get("node_id", "")).strip() != request.node_id:
            return jsonify({"error": "node_id does not match authenticated header"}), 400
        result = manager.store_asset(payload)
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/evidence/<path:relative_path>")
    @browser_login_required
    def evidence_file(relative_path: str):
        try:
            file_path = manager.safe_evidence_path(relative_path)
        except ValueError:
            abort(404)
        if not file_path.exists() or not file_path.is_file():
            abort(404)
        return send_file(file_path)

    @app.route("/api/v1/streams/<node_id>/<mode>")
    @browser_login_required
    def proxy_stream(node_id: str, mode: str):
        if mode not in {"raw", "annotated"}:
            abort(404)
        try:
            upstream = manager.open_node_stream(node_id, mode)
        except FileNotFoundError:
            abort(404)
        except Exception as exc:
            return Response(str(exc), status=503, mimetype="text/plain")
        return Response(
            relay_stream_chunks(upstream),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app
