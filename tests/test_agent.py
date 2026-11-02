from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from typing import Any

import pytest
from livekit import rtc
from PIL import Image

import agent


def rgb_frame(width: int = 64, height: int = 48) -> rtc.VideoFrame:
    return rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, bytes(width * height * 3))


class FakeParticipant:
    def __init__(self) -> None:
        self.texts: list[tuple[str, str]] = []

    async def send_text(self, text: str, *, topic: str = "", **_: Any) -> None:
        self.texts.append((topic, text))


class FakeRoom:
    def __init__(self) -> None:
        self.remote_participants: dict[str, Any] = {}
        self.local_participant = FakeParticipant()
        self.handlers: dict[str, Callable[..., None]] = {}

    def on(self, event: str, handler: Callable[..., None]) -> None:
        self.handlers[event] = handler


class FakeSession:
    """Stands in for GatewaySession; behavior is set per test through class attributes."""

    instances: list[FakeSession] = []
    open_failures = 0
    send_failures = 0
    drop_all = False

    def __init__(self, url: str, api_key: str, model: str, method: str = "", prompt: str = ""):
        self.model = model
        self.max_fps = 20.0
        self.sent: list[bytes] = []
        self.closed = False
        FakeSession.instances.append(self)

    async def open(self) -> None:
        if FakeSession.open_failures:
            FakeSession.open_failures -= 1
            raise RuntimeError("gateway unavailable")

    async def send_jpeg(self, jpeg: bytes) -> dict[str, Any] | None:
        self.sent.append(jpeg)
        if FakeSession.send_failures:
            FakeSession.send_failures -= 1
            raise RuntimeError("connection lost")
        if FakeSession.drop_all:
            return None
        content = json.dumps({"object": "img.pose.kpts", "items": [{"label": "person"}]})
        return {
            "type": "result",
            "frame": len(self.sent),
            "latency_ms": 45.0,
            "response": {"choices": [{"message": {"content": content}}]},
        }

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_gateway(monkeypatch):
    FakeSession.instances = []
    FakeSession.open_failures = 0
    FakeSession.send_failures = 0
    FakeSession.drop_all = False
    monkeypatch.setattr(agent, "GatewaySession", FakeSession)
    monkeypatch.setenv("VLMRUN_API_KEY", "test-key")


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def run_loop(vision: agent.VisionAgent):
    return asyncio.create_task(vision._analysis_loop())


async def stop(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def sent_count() -> int:
    return sum(len(s.sent) for s in FakeSession.instances)


async def test_result_is_published_on_the_vision_topic():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)

    topic, text = room.local_participant.texts[0]
    assert topic == "vision"
    payload = json.loads(text)
    assert payload["items"] == [{"label": "person"}]
    assert payload["_vlmrun"]["model"] == agent.MODEL
    assert payload["_vlmrun"]["frame"] == 1
    assert payload["_vlmrun"]["gateway_ms"] == 45.0
    assert isinstance(payload["_vlmrun"]["latency_ms"], int)


async def test_frames_are_sent_as_jpeg():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 1)
    await stop(task)

    assert FakeSession.instances[0].sent[0][:3] == b"\xff\xd8\xff", "JPEG magic bytes"


async def test_nothing_is_sent_without_a_frame():
    room = FakeRoom()
    task = await run_loop(agent.VisionAgent(room))  # type: ignore[arg-type]
    await asyncio.sleep(0.3)
    await stop(task)

    assert sent_count() == 0
    assert room.local_participant.texts == []


async def test_dropped_frames_publish_nothing():
    FakeSession.drop_all = True
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 2)
    await stop(task)

    assert room.local_participant.texts == []


async def test_a_failed_frame_closes_the_session_and_reconnects():
    FakeSession.send_failures = 1
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)

    assert len(FakeSession.instances) == 2
    assert FakeSession.instances[0].closed, "the broken session was closed"


async def test_a_failed_open_is_retried():
    FakeSession.open_failures = 1
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)

    assert len(FakeSession.instances) == 2


async def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("VLMRUN_API_KEY")
    with pytest.raises(RuntimeError, match="VLMRUN_API_KEY"):
        await agent.VisionAgent(FakeRoom())._analysis_loop()  # type: ignore[arg-type]


def test_wide_frames_are_scaled_to_960_pixels():
    jpeg = agent._jpeg(rgb_frame(1920, 1080))
    image = Image.open(io.BytesIO(jpeg))
    assert image.format == "JPEG"
    assert image.size == (960, 540)


def test_narrow_frames_keep_their_size():
    image = Image.open(io.BytesIO(agent._jpeg(rgb_frame(640, 480))))
    assert image.size == (640, 480)


# Known gaps. Each test states the behavior the realtime socket's billing asks for
# ("billed for the seconds it stays open"; "open one socket when the participant
# publishes a camera and close it when they leave"). strict=True makes a fix show up as
# XPASS, so the marker gets removed when the gap is closed.


@pytest.mark.xfail(strict=True, reason="opens a billed session before any camera is published")
async def test_no_session_until_a_camera_frame_arrives():
    task = await run_loop(agent.VisionAgent(FakeRoom()))  # type: ignore[arg-type]
    await asyncio.sleep(0.3)
    await stop(task)
    assert FakeSession.instances == []


@pytest.mark.xfail(strict=True, reason="keeps re-sending the last frame after the camera leaves")
async def test_sending_stops_when_the_camera_leaves():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 1)
    unsubscribed = room.handlers.get("track_unsubscribed")
    if unsubscribed is not None:
        unsubscribed(object(), object(), object())
    before = sent_count()
    await asyncio.sleep(0.5)
    await stop(task)
    assert sent_count() == before, f"sent {sent_count() - before} frames after the camera left"


@pytest.mark.xfail(strict=True, reason="re-sends the same frame when no new frame has arrived")
async def test_a_frame_is_sent_once():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    vision.latest = rgb_frame()
    task = await run_loop(vision)
    await asyncio.sleep(0.6)
    await stop(task)
    assert sent_count() == 1, f"one frame was sent {sent_count()} times"
