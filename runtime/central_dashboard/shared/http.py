"""Minimal stdlib HTTP helpers used by central and node runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class HttpResult:
    status_code: int
    json_data: Any
    text: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class StdlibHttpClient:
    """Simple HTTP client with JSON helpers and stream access."""

    def post_json(
        self,
        url: str,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> HttpResult:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                **(headers or {}),
            },
        )
        return self._execute(request, timeout=timeout)

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> HttpResult:
        request = Request(url, method="GET", headers=headers or {})
        return self._execute(request, timeout=timeout)

    def post_file(
        self,
        url: str,
        fields: dict,
        *,
        file_field: str,
        file_path: str | Path,
        filename: str,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> HttpResult:
        body, content_type = _multipart_body(
            fields,
            file_field=file_field,
            file_path=Path(file_path),
            filename=filename,
        )
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                **(headers or {}),
            },
        )
        return self._execute(request, timeout=timeout)

    def open_stream(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        request = Request(url, method="GET", headers=headers or {})
        return urlopen(request, timeout=timeout)

    def _execute(self, request: Request, *, timeout: float) -> HttpResult:
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                text = payload.decode("utf-8", errors="replace")
                return HttpResult(
                    status_code=int(response.status),
                    json_data=self._parse_json(text),
                    text=text,
                )
        except HTTPError as exc:
            payload = exc.read()
            text = payload.decode("utf-8", errors="replace")
            return HttpResult(
                status_code=int(exc.code),
                json_data=self._parse_json(text),
                text=text,
            )
        except URLError as exc:
            return HttpResult(status_code=0, json_data=None, text=str(exc.reason))

    @staticmethod
    def _parse_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None


def _multipart_body(
    fields: dict,
    *,
    file_field: str,
    file_path: Path,
    filename: str,
) -> tuple[bytes, str]:
    boundary = f"----aisentinel-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
