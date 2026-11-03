"""A local stand-in for ``wss://gateway.vlm.run/v1/ws``.

It follows the protocol as observed on the live socket: ``session.create`` first, then
``session.created`` with ``max_fps``; each binary message is one frame, numbered from 1,
answered by ``result``, ``frame.dropped`` or ``error``; ``session.close`` ends it with
``session.stats``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

Reply = Callable[[ServerConnection, int, bytes], Awaitable[None]]


async def answer(ws: ServerConnection, frame: int, data: bytes) -> None:
    content = json.dumps({"object": "img.pose.kpts", "items": [{"label": "person"}]})
    await ws.send(
        json.dumps(
            {
                "type": "result",
                "frame": frame,
                "latency_ms": 45.0,
                "response": {"choices": [{"message": {"content": content}}]},
            }
        )
    )


class FakeGateway:
    def __init__(self) -> None:
        self.reply: Reply = answer
        self.refuse: dict[str, Any] | None = None
        self.max_fps = 20.0
        self.headers: list[Headers] = []
        self.creates: list[dict[str, Any]] = []
        self.frames: list[bytes] = []
        self.closes = 0
        self.url = ""

    async def handler(self, ws: ServerConnection) -> None:
        self.headers.append(ws.request.headers if ws.request else Headers())
        self.creates.append(json.loads(await ws.recv()))
        if self.refuse is not None:
            await ws.send(json.dumps({"type": "error", "frame": None, **self.refuse}))
            await ws.close()
            return
        await ws.send(
            json.dumps({"type": "session.created", "session_id": "s1", "max_fps": self.max_fps})
        )
        async for message in ws:
            if isinstance(message, bytes):
                self.frames.append(message)
                await self.reply(ws, len(self.frames), message)
            elif json.loads(message).get("type") == "session.close":
                self.closes += 1
                try:
                    await ws.send(json.dumps({"type": "session.stats", "reads": len(self.frames)}))
                except ConnectionClosed:
                    return  # the client may hang up right after session.close
                await ws.close()


@pytest.fixture
async def gateway() -> AsyncIterator[FakeGateway]:
    fake = FakeGateway()
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        fake.url = f"ws://127.0.0.1:{port}/v1/ws"
        yield fake
