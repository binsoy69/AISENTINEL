"""Flask application for the editable AISENTINEL demo dashboard."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from central_dashboard.central_service.auth import (
    browser_api_login_required,
    browser_login_required,
    browser_password_matches,
    is_browser_authenticated,
)
from central_dashboard.central_service.db import connect_db, init_db
from central_dashboard.central_service.repositories import CentralRepository
from editable_dashboard.editable_service.services import EditableDashboardManager


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
CENTRAL_TEMPLATE_DIR = RUNTIME_ROOT / "central_dashboard" / "central_service" / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _safe_next_path(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if value.startswith("/") and not value.startswith("//"):
        return value
    return ""


def _records_payload_from_request() -> list[dict]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        records = payload.get("records", payload)
    else:
        raw_records = request.form.get("records", "[]")
        records = json.loads(raw_records)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("Editable records payload must be a list.")
    return [dict(record) for record in records if isinstance(record, dict)]


def create_app(config) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(CENTRAL_TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["CENTRAL_CONFIG"] = config
    app.secret_key = config.browser_auth.secret_key
    app.permanent_session_lifetime = timedelta(minutes=config.browser_auth.session_ttl_minutes)

    connection = connect_db(config.db_path)
    init_db(connection)
    connection.close()
    repository = CentralRepository(connection_factory=lambda: connect_db(config.db_path))
    manager = EditableDashboardManager(config, repository)
    app.extensions["central_repository"] = repository
    app.extensions["central_manager"] = manager
    app.extensions["central_connection"] = repository

    @app.teardown_appcontext
    def _close_thread_connection(_exc=None):
        repository.close_thread_connection()

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
        result = manager.create_session(request.get_json(silent=True) or {})
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

    @app.route("/api/v1/sessions/<session_id>/incidents")
    @browser_api_login_required
    def session_incidents_api(session_id: str):
        result = manager.session_incidents(session_id)
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

    @app.route("/api/v1/editable/incidents", methods=["POST"])
    @browser_api_login_required
    def save_editable_incidents_api():
        try:
            records = _records_payload_from_request()
            result = manager.save_editable_incidents(records, request.files)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            result = {"ok": False, "error": str(exc), "status_code": 400}
        status_code = int(result.pop("status_code", 200))
        return jsonify(result), status_code

    @app.route("/api/v1/editable/incidents/<incident_id>", methods=["DELETE"])
    @browser_api_login_required
    def delete_editable_incident_api(incident_id: str):
        result = manager.delete_editable_incident(incident_id)
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
        return Response("Editable demo dashboard does not open camera streams.", status=404, mimetype="text/plain")

    return app
