"""Watch a LiveKit room and read each new frame on the VLM Run gateway.

The page publishes a camera or a screen. This worker keeps the newest frame,
JPEG-encodes it, and sends those bytes on one ``WS /v1/ws`` session. Each
``result`` is published back into the room on the ``vision`` text stream.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from PIL import Image

from gateway import GatewaySession, observation, sample_interval

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger("vlmrun-vision")

GATEWAY_WS = os.environ.get("VLMRUN_GATEWAY_WS", "wss://gateway.vlm.run/v1/ws")
MODEL = os.environ.get("VLMRUN_MODEL", "usyd-community/vitpose-plus-large")
METHOD = os.environ.get("VLMRUN_METHOD", "pose")
PROMPT = os.environ.get("VLMRUN_PROMPT", "")
VISION_FPS = float(os.environ.get("VISION_FPS", "8"))
RESULT_TOPIC = "vision"


def _jpeg(frame: rtc.VideoFrame, max_width: int = 960) -> bytes:
    rgb = frame.convert(rtc.VideoBufferType.RGB24)
    image = Image.frombytes("RGB", (rgb.width, rgb.height), bytes(rgb.data))
    if image.width > max_width:
        image = image.resize((max_width, int(image.height * max_width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


class VisionAgent:
    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        self.latest: rtc.VideoFrame | None = None
        self._reader: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.room.on("track_subscribed", self._on_track)
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
        await self._analysis_loop()

    def _on_track(
        self,
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        logger.info("watching %s", participant.identity)
        self._watch(track)

    def _watch(self, track: rtc.Track) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        self._reader = asyncio.create_task(self._read_frames(track))

    async def _read_frames(self, track: rtc.Track) -> None:
        stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGB24)
        try:
            async for event in stream:
                self.latest = event.frame
        finally:
            await stream.aclose()

    async def _analysis_loop(self) -> None:
        api_key = os.environ.get("VLMRUN_API_KEY", "")
        if not api_key:
            raise RuntimeError("VLMRUN_API_KEY is not set")
        session: GatewaySession | None = None
        interval = sample_interval(VISION_FPS, 0)
        try:
            while True:
                if session is None:
                    session = GatewaySession(GATEWAY_WS, api_key, MODEL, METHOD, PROMPT)
                    try:
                        await session.open()
                    except Exception:
                        logger.exception("gateway session failed")
                        session = None
                        await asyncio.sleep(1)
                        continue
                    interval = sample_interval(VISION_FPS, session.max_fps)
                    logger.info("gateway open model=%s max_fps=%s", MODEL, session.max_fps)
                started = time.perf_counter()
                frame = self.latest
                if frame is not None:
                    try:
                        jpeg = await asyncio.to_thread(_jpeg, frame)
                        event = await session.send_jpeg(jpeg)
                        latency_ms = round((time.perf_counter() - started) * 1000)
                        if event is not None:
                            payload = observation(event, model=MODEL, latency_ms=latency_ms)
                            await self.room.local_participant.send_text(json.dumps(payload), topic=RESULT_TOPIC)
                            logger.info("result %sms", latency_ms)
                    except Exception:
                        logger.exception("frame failed")
                        await session.aclose()
                        session = None
                        await asyncio.sleep(1)
                        continue
                elapsed = time.perf_counter() - started
                await asyncio.sleep(max(0.0, interval - elapsed))
        finally:
            if session is not None:
                await session.aclose()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    logger.info("joined %s", ctx.room.name)
    await VisionAgent(ctx.room).start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
