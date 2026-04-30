"""Test helpers for the standalone central dashboard stack."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from urllib.parse import urlparse

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.shared.http import HttpResult


class InProcessHttpClient:
    """Routes HTTP-like calls into Flask test clients."""

    def __init__(self) -> None:
        self.clients = {}
        self.stream_payloads = {}

    def register_app(self, base_url: str, app) -> None:
        self.clients[self._base(base_url)] = app.test_client()

    def set_stream_payload(self, url: str, payload: bytes) -> None:
        self.stream_payloads[url] = payload

    def post_json(self, url: str, payload: dict, *, headers=None, timeout=5.0) -> HttpResult:
        client = self._client_for(url)
        parsed = urlparse(url)
        response = client.post(parsed.path, json=payload, headers=headers or {})
        text = response.get_data(as_text=True)
        return HttpResult(response.status_code, response.get_json(silent=True), text)

    def post_file(
        self,
        url: str,
        fields: dict,
        *,
        file_field: str,
        file_path: str | Path,
        filename: str,
        headers=None,
        timeout=5.0,
    ) -> HttpResult:
        client = self._client_for(url)
        parsed = urlparse(url)
        with Path(file_path).open("rb") as stream:
            data = {
                **fields,
                file_field: (BytesIO(stream.read()), filename),
            }
        response = client.post(
            parsed.path,
            data=data,
            headers=headers or {},
            content_type="multipart/form-data",
        )
        text = response.get_data(as_text=True)
        return HttpResult(response.status_code, response.get_json(silent=True), text)

    def get_json(self, url: str, *, headers=None, timeout=5.0) -> HttpResult:
        client = self._client_for(url)
        parsed = urlparse(url)
        response = client.get(parsed.path, headers=headers or {})
        text = response.get_data(as_text=True)
        return HttpResult(response.status_code, response.get_json(silent=True), text)

    def open_stream(self, url: str, *, headers=None, timeout=10.0):
        if url in self.stream_payloads:
            return BytesIO(self.stream_payloads[url])
        raise FileNotFoundError(url)

    def _client_for(self, url: str):
        base = self._base(url)
        if base not in self.clients:
            raise KeyError(f"No in-process client registered for {base}")
        return self.clients[base]

    @staticmethod
    def _base(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
