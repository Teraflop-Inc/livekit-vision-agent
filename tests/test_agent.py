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
        self.files: list[dict[str, Any]] = []
        self.order: list[str] = []

    async def send_text(self, text: str, *, topic: str = "", **_: Any) -> None:
        self.texts.append((topic, text))
        self.order.append(f"text:{topic}")

    async def send_bytes(self, data: bytes, *, topic: str = "", **kwargs: Any) -> None:
        self.files.append({"data": data, "topic": topic, **kwargs})
        self.order.append(f"bytes:{topic}")


class FakeRoom:
    def __init__(self) -> None:
        self.remote_participants: dict[str, Any] = {}
        self.local_participant = FakeParticipant()
        self.handlers: dict[str, Callable[..., None]] = {}
        self.text_handlers: dict[str, Callable[..., None]] = {}

    def on(self, event: str, handler: Callable[..., None]) -> None:
        self.handlers[event] = handler

    def register_text_stream_handler(self, topic: str, handler: Callable[..., None]) -> None:
        self.text_handlers[topic] = handler


class FakeSession:
    """Stands in for GatewaySession; behavior is set per test through class attributes."""

    instances: list[FakeSession] = []
    open_failures = 0
    send_failures = 0
    stalls = 0
    drop_all = False

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        method: str = "",
        prompt: str = "",
        method_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.method = method
        self.prompt = prompt
        self.method_params = method_params
        self.max_fps = 20.0
        self.sent: list[bytes] = []
        self.closed = False
        FakeSession.instances.append(self)

    async def open(self) -> None:
        if FakeSession.open_failures:
            FakeSession.open_failures -= 1
            raise RuntimeError("gateway unavailable")

    async def send_jpeg(self, jpeg: bytes, timeout: float = 5.0) -> dict[str, Any] | None:
        self.sent.append(jpeg)
        if FakeSession.stalls:
            FakeSession.stalls -= 1
            await asyncio.sleep(timeout)
            raise TimeoutError("no reply for the frame")
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
    FakeSession.stalls = 0
    FakeSession.drop_all = False
    monkeypatch.setattr(agent, "GatewaySession", FakeSession)
    monkeypatch.setattr(agent, "RETRY_S", 0.05)
    monkeypatch.setenv("VLMRUN_API_KEY", "test-key")


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def run_loop(vision: agent.VisionAgent):
    return asyncio.create_task(vision._analysis_loop())


def camera(vision: agent.VisionAgent, interval: float = 0.03) -> asyncio.Task:
    """Pushes a new frame every ``interval`` seconds, like a live camera."""

    async def feed() -> None:
        while True:
            vision.push(rgb_frame())
            await asyncio.sleep(interval)

    return asyncio.create_task(feed())


async def stop(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def sent_count() -> int:
    return sum(len(s.sent) for s in FakeSession.instances)


async def test_result_is_published_on_the_vision_topic():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)
    feed.cancel()

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
    feed = camera(vision)
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
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 2)
    await stop(task)

    assert room.local_participant.texts == []


async def test_a_failed_frame_closes_the_session_and_reconnects():
    FakeSession.send_failures = 1
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)

    assert len(FakeSession.instances) == 2
    assert FakeSession.instances[0].closed, "the broken session was closed"


async def test_a_failed_open_is_retried():
    FakeSession.open_failures = 1
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
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


# The gateway bills the seconds a session is open: "open one socket when the participant
# publishes a camera and close it when they leave".


async def test_no_session_until_a_camera_frame_arrives():
    task = await run_loop(agent.VisionAgent(FakeRoom()))  # type: ignore[arg-type]
    await asyncio.sleep(0.3)
    await stop(task)
    assert FakeSession.instances == []


async def test_session_closes_when_the_camera_leaves():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    room.handlers["track_unsubscribed"] = vision._on_track_gone
    vision._track_sid = "cam"
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 2)
    feed.cancel()
    room.handlers["track_unsubscribed"](_Track("cam"), object(), object())
    await wait_for(lambda: FakeSession.instances[-1].closed)
    before = sent_count()
    await asyncio.sleep(0.3)
    await stop(task)
    assert sent_count() == before, "nothing sent after the camera left"


async def test_idle_camera_closes_the_session_and_a_new_frame_reopens_it(monkeypatch):
    monkeypatch.setattr(agent, "IDLE_CLOSE_S", 0.2)
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    task = await run_loop(vision)
    vision.push(rgb_frame())
    await wait_for(lambda: sent_count() == 1)
    await wait_for(lambda: FakeSession.instances[0].closed, timeout=2.0)
    vision.push(rgb_frame())
    await wait_for(lambda: sent_count() == 2)
    await stop(task)
    assert len(FakeSession.instances) == 2


async def test_a_frame_is_sent_once():
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    vision.push(rgb_frame())
    task = await run_loop(vision)
    await asyncio.sleep(0.6)
    await stop(task)
    assert sent_count() == 1, f"one frame was sent {sent_count()} times"


async def test_only_the_newest_frame_is_sent():
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    first, newest = rgb_frame(64, 48), rgb_frame(96, 48)
    vision.push(first)
    vision.push(newest)
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() == 1)
    await asyncio.sleep(0.2)
    await stop(task)
    image = Image.open(io.BytesIO(FakeSession.instances[0].sent[0]))
    assert sent_count() == 1 and image.size == (96, 48)


class _Track:
    def __init__(self, sid: str) -> None:
        self.sid = sid


async def test_failed_opens_back_off(monkeypatch):
    """A model at capacity must not be hammered: waits double up to MAX_RETRY_S."""
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def record(seconds: float) -> None:
        waits.append(seconds)
        await real_sleep(0)

    FakeSession.open_failures = 5
    monkeypatch.setattr(agent, "RETRY_S", 1.0)
    monkeypatch.setattr(agent, "MAX_RETRY_S", 4.0)
    monkeypatch.setattr(agent.asyncio, "sleep", record)
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    for _ in range(8):
        vision.push(rgb_frame())
    task = asyncio.create_task(vision._analysis_loop())
    for _ in range(200):
        if len(FakeSession.instances) >= 6:
            break
        vision.push(rgb_frame())
        await real_sleep(0.005)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert waits[:5] == [1.0, 2.0, 4.0, 4.0, 4.0], waits


async def test_a_stalled_frame_reconnects_and_results_resume(monkeypatch):
    monkeypatch.setattr(agent, "FRAME_TIMEOUT_S", 0.2)
    FakeSession.stalls = 1
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: bool(room.local_participant.texts))
    await stop(task)
    feed.cancel()
    assert len(FakeSession.instances) == 2
    assert FakeSession.instances[0].closed, "the stuck session was closed"


class _Reader:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def read_all(self) -> str:
        return json.dumps(self._body)


async def test_each_result_comes_with_the_frame_it_was_read_on():
    """The page draws a result on the JPEG it belongs to, so the frame goes first, same id."""
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: len(room.local_participant.texts) >= 2)
    await stop(task)
    feed.cancel()

    files = room.local_participant.files
    assert room.local_participant.order[:2] == ["bytes:frame", "text:vision"]
    for file, (_, text) in zip(files, room.local_participant.texts, strict=False):
        payload = json.loads(text)
        assert file["mime_type"] == "image/jpeg" and file["data"][:3] == b"\xff\xd8\xff"
        assert file["name"] == str(payload["_vlmrun"]["seq"])
        assert file["attributes"] == {"frame": str(payload["_vlmrun"]["seq"])}


async def test_control_message_switches_the_model():
    room = FakeRoom()
    vision = agent.VisionAgent(room)  # type: ignore[arg-type]
    room.register_text_stream_handler("control", vision._on_control)
    feed = camera(vision)
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 1)
    await vision._apply_control(_Reader({"model": "geopavlakos/hamer", "method": "pose"}))  # type: ignore[arg-type]
    await wait_for(lambda: FakeSession.instances[-1].model == "geopavlakos/hamer")
    await wait_for(lambda: len(FakeSession.instances[-1].sent) >= 1)
    await stop(task)
    feed.cancel()

    assert FakeSession.instances[0].closed, "the old model's session was closed"
    payload = json.loads(room.local_participant.texts[-1][1])
    assert payload["_vlmrun"]["model"] == "geopavlakos/hamer"


async def test_control_message_without_a_model_is_ignored():
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    before = vision._want
    await vision._apply_control(_Reader({"method": "pose"}))  # type: ignore[arg-type]
    assert vision._want == before


async def test_segment_sends_prompt_and_png_masks():
    vision = agent.VisionAgent(FakeRoom())  # type: ignore[arg-type]
    vision._want = ("some/segment-model", "segment", "")
    vision.push(rgb_frame())
    task = await run_loop(vision)
    await wait_for(lambda: sent_count() >= 1)
    await stop(task)

    session = FakeSession.instances[0]
    assert session.prompt == "face", "segment defaults to a prompt"
    assert session.method_params == {"prompt": "face", "mask_format": "png"}
