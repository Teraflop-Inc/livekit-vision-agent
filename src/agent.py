"""Watch a LiveKit room and read each new frame on the VLM Run gateway.

The page publishes a camera or a screen. This worker keeps the newest frame,
JPEG-encodes it, and sends those bytes on one ``WS /v1/ws`` session. Each
``result`` is published back into the room on the ``vision`` text stream.

The gateway bills the seconds a session is open, so the session opens on the
first frame, closes when the video track goes away or no new frame arrives for
``IDLE_CLOSE_S``, and each frame is sent once.
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
# Close the session after this long without a new frame (a paused or muted camera).
IDLE_CLOSE_S = 5.0
# Wait between failed attempts doubles from RETRY_S up to MAX_RETRY_S, so a model at
# capacity ("capacity_error", no retry_after) isn't hammered.
RETRY_S = 1.0
MAX_RETRY_S = 30.0


def _jpeg(frame: rtc.VideoFrame, max_width: int = 960) -> bytes:
    # LiveKit can't convert RGB24 to RGB24 ("rgb24 to Rgb24 is not supported"), and the
    # reader already asks for RGB24, so only convert frames in another format.
    rgb = frame if frame.type == rtc.VideoBufferType.RGB24 else frame.convert(rtc.VideoBufferType.RGB24)
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
        self._new_frame = asyncio.Event()
        self._track_sid: str | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stream: rtc.VideoStream | None = None
        self._closing: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self.room.on("track_subscribed", self._on_track)
        self.room.on("track_unsubscribed", self._on_track_gone)
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
        await self._analysis_loop()

    def push(self, frame: rtc.VideoFrame) -> None:
        """Offer a new frame; it replaces any frame not yet sent."""
        self.latest = frame
        self._new_frame.set()

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

    def _on_track_gone(
        self,
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        _participant: rtc.RemoteParticipant,
    ) -> None:
        if track.sid != self._track_sid:
            return
        logger.info("video track left")
        self._release_reader()
        self._track_sid = None
        self.latest = None
        self._new_frame.set()  # wakes the loop so it closes the session

    def _watch(self, track: rtc.Track) -> None:
        self._release_reader()
        self._track_sid = track.sid
        self._stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGB24)
        self._reader = asyncio.create_task(self._read_frames(self._stream))

    def _release_reader(self) -> None:
        # A separate task closes the stream: a reader cancelled before it ever ran would
        # skip its own cleanup and leak the stream.
        reader, stream = self._reader, self._stream
        self._reader, self._stream = None, None
        if reader is None and stream is None:
            return

        async def close() -> None:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if stream is not None:
                await stream.aclose()

        task = asyncio.create_task(close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _read_frames(self, stream: rtc.VideoStream) -> None:
        async for event in stream:
            if stream is not self._stream:
                return  # released; a late frame must not reopen a session
            self.push(event.frame)

    async def _next_frame(self, timeout: float | None) -> rtc.VideoFrame | None:
        """The newest unsent frame, or None after ``timeout`` without one."""
        try:
            await asyncio.wait_for(self._new_frame.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        self._new_frame.clear()
        frame, self.latest = self.latest, None
        return frame

    async def _analysis_loop(self) -> None:
        api_key = os.environ.get("VLMRUN_API_KEY", "")
        if not api_key:
            raise RuntimeError("VLMRUN_API_KEY is not set")
        session: GatewaySession | None = None
        interval = sample_interval(VISION_FPS, 0)
        failures = 0
        try:
            while True:
                frame = await self._next_frame(IDLE_CLOSE_S if session is not None else None)
                if frame is None:
                    # The camera left or went idle: close the billed session until the next frame.
                    if session is not None:
                        logger.info("closing gateway session (no frames)")
                        await session.aclose()
                        session = None
                    continue

                if session is None:
                    session = GatewaySession(GATEWAY_WS, api_key, MODEL, METHOD, PROMPT)
                    try:
                        await session.open()
                    except Exception as e:
                        failures += 1
                        wait = min(MAX_RETRY_S, RETRY_S * 2 ** (failures - 1))
                        logger.warning("gateway session failed (%s); retrying in %.0fs", e, wait)
                        session = None
                        await asyncio.sleep(wait)
                        continue
                    interval = sample_interval(VISION_FPS, session.max_fps)
                    logger.info("gateway open model=%s max_fps=%s", MODEL, session.max_fps)

                started = time.perf_counter()
                try:
                    jpeg = await asyncio.to_thread(_jpeg, frame)
                    event = await session.send_jpeg(jpeg)
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    failures = 0
                    if event is not None:
                        payload = observation(event, model=MODEL, latency_ms=latency_ms)
                        await self.room.local_participant.send_text(json.dumps(payload), topic=RESULT_TOPIC)
                        logger.info("result %sms", latency_ms)
                except Exception:
                    logger.exception("frame failed")
                    failures += 1
                    await session.aclose()
                    session = None
                    await asyncio.sleep(min(MAX_RETRY_S, RETRY_S * 2 ** (failures - 1)))
                    continue
                # Pace to the slower of VISION_FPS and the session's max_fps.
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
        finally:
            if session is not None:
                await session.aclose()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    logger.info("joined %s", ctx.room.name)
    await VisionAgent(ctx.room).start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
