"""Stream relay helpers for the central dashboard service."""

from __future__ import annotations


def relay_stream_chunks(stream_response, chunk_size: int = 4096):
    """Yield proxied MJPEG stream bytes."""
    try:
        while True:
            chunk = stream_response.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            stream_response.close()
        except Exception:
            pass
