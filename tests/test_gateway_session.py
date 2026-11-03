from __future__ import annotations

import json

import pytest

from gateway import GatewaySession

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


async def test_open_sends_session_create_with_bearer_and_reads_max_fps(gateway):
    gateway.max_fps = 12.0
    session = GatewaySession(gateway.url, "key-1", "usyd-community/vitpose-plus-large", "pose")
    await session.open()
    await session.aclose()

    assert gateway.headers[0]["Authorization"] == "Bearer key-1"
    assert gateway.creates[0] == {
        "type": "session.create",
        "model": "usyd-community/vitpose-plus-large",
        "max_inflight": 1,
        "method": "pose",
    }
    assert session.max_fps == 12.0


async def test_open_sends_method_and_prompt_only_when_set(gateway):
    bare = GatewaySession(gateway.url, "k", "m")
    await bare.open()
    await bare.aclose()
    full = GatewaySession(gateway.url, "k", "m", method="pose", prompt="person")
    await full.open()
    await full.aclose()

    assert "method" not in gateway.creates[0] and "prompt" not in gateway.creates[0]
    assert gateway.creates[1]["method"] == "pose" and gateway.creates[1]["prompt"] == "person"


async def test_refused_session_raises_the_gateway_message(gateway):
    gateway.refuse = {
        "error_type": "invalid_request_error",
        "message": "Model 'qwen/qwen3.8-27b' is generative; a socket streams frame models.",
    }
    session = GatewaySession(gateway.url, "k", "qwen/qwen3.8-27b", "chat")
    with pytest.raises(RuntimeError, match="generative"):
        await session.open()


async def test_frame_goes_as_raw_bytes_and_result_comes_back(gateway):
    session = GatewaySession(gateway.url, "k", "m", "pose")
    await session.open()
    event = await session.send_jpeg(JPEG)
    await session.aclose()

    assert gateway.frames == [JPEG], "one binary message per frame, not base64"
    assert event is not None and event["type"] == "result" and event["frame"] == 1
    content = json.loads(event["response"]["choices"][0]["message"]["content"])
    assert content["items"] == [{"label": "person"}]


async def test_dropped_frame_returns_none(gateway):
    async def drop(ws, frame, data):
        await ws.send(
            json.dumps({"type": "frame.dropped", "frame": frame, "inflight": 1, "reason": "rate"})
        )

    gateway.reply = drop
    session = GatewaySession(gateway.url, "k", "m", "pose")
    await session.open()
    assert await session.send_jpeg(JPEG) is None
    await session.aclose()


async def test_frame_error_raises_the_gateway_message(gateway):
    async def reject(ws, frame, data):
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "error_type": "invalid_request_error",
                    "message": "a frame is raw JPEG, PNG, WebP or GIF bytes",
                    "frame": frame,
                }
            )
        )

    gateway.reply = reject
    session = GatewaySession(gateway.url, "k", "m", "pose")
    await session.open()
    with pytest.raises(RuntimeError, match="raw JPEG"):
        await session.send_jpeg(b"not an image")
    await session.aclose()


async def test_aclose_sends_session_close_and_is_safe_twice(gateway):
    session = GatewaySession(gateway.url, "k", "m", "pose")
    await session.open()
    await session.aclose()
    await session.aclose()
    assert gateway.closes == 1


async def test_send_before_open_raises():
    session = GatewaySession("ws://unused", "k", "m")
    with pytest.raises(RuntimeError, match="not open"):
        await session.send_jpeg(JPEG)


async def test_a_frame_without_a_reply_times_out(gateway):
    async def silent(ws, frame, data):
        return None

    gateway.reply = silent
    session = GatewaySession(gateway.url, "k", "m", "pose")
    await session.open()
    with pytest.raises(TimeoutError, match="no reply"):
        await session.send_jpeg(JPEG, timeout=0.2)
    await session.aclose()
