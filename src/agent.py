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
    # The reader already asks VideoStream for RGB24, so the buffer is packed RGB.
    image = Image.frombytes("RGB", (frame.width, frame.height), bytes(frame.data))
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
        self.model = MODEL
        self.method = METHOD
        self.prompt = PROMPT
        self._want = (self.model, self.method, self.prompt)
        self._seq = 0

    async def start(self) -> None:
        self.room.register_text_stream_handler("control", self._on_control)
        self.room.on("track_subscribed", self._on_track)
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
        await self._analysis_loop()

    def _on_control(self, reader: rtc.TextStreamReader, _participant: str) -> None:
        task = asyncio.create_task(self._apply_control(reader))
        task.add_done_callback(lambda done: done.exception() if done.done() and not done.cancelled() and done.exception() else None)

    async def _apply_control(self, reader: rtc.TextStreamReader) -> None:
        try:
            body = json.loads(await reader.read_all())
        except Exception:
            logger.exception("control message")
            return
        model = str(body.get("model") or "").strip()
        if not model:
            return
        method = str(body.get("method") or "")
        prompt = str(body.get("prompt") or "")
        self._want = (model, method, prompt)
        logger.info("model request %s method=%s", model, method or "(default)")

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
                frame = self.latest
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue
                want = self._want
                if session is None or session.selection != want:
                    if session is not None:
                        await session.aclose()
                        session = None
                    model, method, prompt = want
                    open_prompt = prompt
                    params = None
                    if method == "segment":
                        open_prompt = prompt or "face"
                        params = {"prompt": open_prompt, "mask_format": "png"}
                    session = GatewaySession(GATEWAY_WS, api_key, model, method, open_prompt, params)
                    session.selection = want
                    try:
                        await session.open()
                    except Exception:
                        logger.exception("gateway session failed")
                        session = None
                        await asyncio.sleep(1)
                        continue
                    interval = sample_interval(VISION_FPS, session.max_fps)
                    logger.info("gateway open model=%s method=%s max_fps=%s", model, method or "(default)", session.max_fps)
                started = time.perf_counter()
                try:
                    jpeg = await asyncio.to_thread(_jpeg, frame)
                    event = await session.send_jpeg(jpeg)
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    if event is not None:
                        self._seq += 1
                        payload = observation(event, model=session.model, latency_ms=latency_ms)
                        payload["_vlmrun"]["seq"] = self._seq
                        frame_id = str(self._seq)
                        # The drawing belongs to this JPEG, not to a later camera frame.
                        await self.room.local_participant.send_bytes(
                            jpeg,
                            topic="frame",
                            name=frame_id,
                            mime_type="image/jpeg",
                            attributes={"frame": frame_id},
                            compress=False,
                        )
                        await self.room.local_participant.send_text(json.dumps(payload), topic=RESULT_TOPIC)
                        logger.info("result %sms model=%s", latency_ms, session.model)
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
