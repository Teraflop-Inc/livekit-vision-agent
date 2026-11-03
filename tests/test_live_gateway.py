"""The live VLM Run socket. Skipped without VLMRUN_API_KEY; sessions are a few seconds."""

from __future__ import annotations

import asyncio
import io
import os
import urllib.request

import pytest
from PIL import Image

from gateway import GatewaySession, observation

pytestmark = pytest.mark.skipif(
    not os.environ.get("VLMRUN_API_KEY"), reason="VLMRUN_API_KEY not set"
)

URL = os.environ.get("VLMRUN_GATEWAY_WS", "wss://gateway.vlm.run/v1/ws")
PEOPLE_PHOTO = "https://ultralytics.com/images/bus.jpg"


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def people() -> bytes:
    req = urllib.request.Request(PEOPLE_PHOTO, headers={"User-Agent": "tests"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return jpeg(Image.open(io.BytesIO(resp.read())).resize((640, 480)))


async def test_vitpose_finds_people_and_nobody_in_a_blank_frame():
    session = GatewaySession(URL, os.environ["VLMRUN_API_KEY"], "usyd-community/vitpose-plus-large", "pose")
    await session.open()
    try:
        assert session.max_fps > 0
        counts = []
        for frame in (people(), jpeg(Image.new("RGB", (640, 480), "white"))):
            event = await session.send_jpeg(frame)
            assert event is not None, "frame dropped"
            obs = observation(event, model=session.model, latency_ms=0)
            counts.append(len(obs["items"]))
            await asyncio.sleep(1 / session.max_fps)
    finally:
        await session.aclose()
    assert counts[0] > 0 and counts[1] == 0, counts


async def test_generative_model_is_refused():
    session = GatewaySession(URL, os.environ["VLMRUN_API_KEY"], "qwen/qwen3.8-27b", "chat")
    with pytest.raises(RuntimeError, match="generative"):
        await session.open()
