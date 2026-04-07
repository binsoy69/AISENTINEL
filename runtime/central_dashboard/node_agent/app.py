"""Flask application for the standalone node agent."""

from __future__ import annotations

from flask import Flask, Response, jsonify, request

from central_dashboard.node_agent.auth import agent_api_required
from central_dashboard.node_agent.config import NodeAgentConfig
from central_dashboard.node_agent.state import NodeRuntime
from central_dashboard.shared.dto import CommandAck, SessionCommand


def create_app(
    config: NodeAgentConfig,
    *,
    runtime: NodeRuntime | None = None,
    http_client=None,
    start_background: bool = True,
) -> Flask:
    app = Flask(__name__)
    app.config["NODE_CONFIG"] = config
    runtime = runtime or NodeRuntime(config, http_client=http_client)
    app.extensions["node_runtime"] = runtime

    if start_background:
        runtime.start_background()

    @app.route("/agent/v1/status")
    @agent_api_required
    def status():
        return jsonify(runtime.status_payload())

    @app.route("/agent/v1/session/start", methods=["POST"])
    @agent_api_required
    def start_session():
        payload = request.get_json(silent=True) or {}
        command = SessionCommand.from_dict(payload)
        ack = runtime.start_session(command.session.to_dict())
        return jsonify(ack.to_dict())

    @app.route("/agent/v1/session/stop", methods=["POST"])
    @agent_api_required
    def stop_session():
        ack = runtime.stop_session()
        return jsonify(ack.to_dict())

    @app.route("/agent/v1/session/restart", methods=["POST"])
    @agent_api_required
    def restart_session():
        payload = request.get_json(silent=True) or {}
        command = SessionCommand.from_dict(payload)
        ack = runtime.restart_session(command.session.to_dict())
        return jsonify(ack.to_dict())

    @app.route("/agent/v1/stream/raw")
    @agent_api_required
    def raw_stream():
        return Response(
            runtime.stream_generator("raw"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/agent/v1/stream/annotated")
    @agent_api_required
    def annotated_stream():
        return Response(
            runtime.stream_generator("annotated"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app
