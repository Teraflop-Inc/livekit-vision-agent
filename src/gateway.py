"""One realtime session on the VLM Run gateway.

The socket is ``WS /v1/ws``. The first message opens the session. Every later
message is one JPEG, PNG, WebP, or GIF, sent as raw bytes. The server answers
``result`` for that frame, or ``frame.dropped`` when the frame is too soon or
the session is already busy.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from websockets.asyncio.client import connect as ws_connect

_TEXT_LIMIT = 180


def sample_interval(vision_fps: float, max_fps: float) -> float:
    """Seconds between frames. The slower of the local cap and the session cap wins."""
    rate = vision_fps if max_fps <= 0 else min(vision_fps, max_fps)
    if rate <= 0:
        return 0.5
    return 1.0 / rate


def message_content(response: object) -> object:
    """The assistant text inside a chat-completion ``result.response``."""
    if not isinstance(response, dict):
        return None
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message") or {}
    return message.get("content")


def _trim(value: object) -> object:
    """Keep the data channel small. Masks and other pixel blobs stay on the gateway."""
    if isinstance(value, str):
        return value if len(value) <= _TEXT_LIMIT else f"<{len(value)} chars>"
    if isinstance(value, list):
        head = [_trim(item) for item in value[:24]]
        if len(value) > 24:
            head.append(f"<{len(value) - 24} more>")
        return head
    if isinstance(value, dict):
        return {str(key): _trim(item) for key, item in value.items()}
    return value


def observation(event: dict[str, Any], *, model: str, latency_ms: int) -> dict[str, Any]:
    """The model's JSON for one frame, plus the timing the demo prints."""
    content = message_content(event.get("response"))
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {"text": content}
    if not isinstance(content, dict):
        content = {"value": content}
    body = _trim(content)
    if not isinstance(body, dict):
        body = {"value": body}
    body["_vlmrun"] = {
        "model": model,
        "latency_ms": latency_ms,
        "gateway_ms": event.get("latency_ms"),
        "frame": event.get("frame"),
    }
    return body


class GatewaySession:
    """A single open ``WS /v1/ws`` session."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        method: str = "",
        prompt: str = "",
        method_params: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model
        self.method = method
        self.prompt = prompt
        self.method_params = method_params or {}
        self.max_fps = 0.0
        self._ws: Any = None

    async def open(self) -> None:
        self._ws = await ws_connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=8 * 1024 * 1024,
            open_timeout=30,
        )
        opening: dict[str, object] = {"type": "session.create", "model": self.model, "max_inflight": 1}
        if self.method:
            opening["method"] = self.method
        if self.prompt:
            opening["prompt"] = self.prompt
        if self.method_params:
            opening["method_params"] = self.method_params
        await self._ws.send(json.dumps(opening))
        created = json.loads(await self._ws.recv())
        if created.get("type") == "error":
            message = created.get("message") or created.get("error_type") or "gateway refused the session"
            raise RuntimeError(str(message))
        if created.get("type") != "session.created":
            raise RuntimeError(f"expected session.created, got {created.get('type')}")
        self.max_fps = float(created.get("max_fps") or 0)

    async def send_jpeg(self, jpeg: bytes, timeout: float = 5.0) -> dict[str, Any] | None:
        """Send one frame. ``None`` means the server dropped it.

        Raises ``TimeoutError`` when no reply arrives within ``timeout`` seconds; the
        session should then be closed, since a late reply would answer the wrong frame.
        """
        if self._ws is None:
            raise RuntimeError("session is not open")
        await self._ws.send(jpeg)
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"no reply for the frame within {timeout:g}s") from None
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "frame.dropped":
            return None
        if kind == "error":
            message = event.get("message") or event.get("error_type") or "frame failed"
            raise RuntimeError(str(message))
        if kind != "result":
            raise RuntimeError(f"expected result, got {kind}")
        return event

    async def aclose(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "session.close"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
