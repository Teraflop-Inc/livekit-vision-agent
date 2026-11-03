"""Real-room check: a fake camera publishes into a LiveKit Cloud room, the agent runs
against the live VLM Run socket, and its ``vision`` text stream is read back.

Needs LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET and VLMRUN_API_KEY. Prints
PASS/FAIL per check and exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from livekit import api, rtc
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ["LIVEKIT_URL"]
W, H = 640, 480
PEOPLE_PHOTO = "https://ultralytics.com/images/bus.jpg"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def token(identity: str, room: str) -> str:
    return (
        api.AccessToken()
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def people() -> Image.Image:
    req = urllib.request.Request(PEOPLE_PHOTO, headers={"User-Agent": "e2e"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB").resize((W, H))


async def main() -> int:
    room_name = f"lva-e2e-{uuid.uuid4().hex[:8]}"
    camera = rtc.Room()
    observations: list[tuple[str, dict]] = []
    arrived: list[float] = []
    left_at = 0.0
    phase = "warmup"
    tasks: set[asyncio.Task] = set()

    async def read(reader: rtc.TextStreamReader) -> None:
        observations.append((phase, json.loads(await reader.read_all())))
        arrived.append(time.monotonic())

    def on_text(reader: rtc.TextStreamReader, identity: str) -> None:
        task = asyncio.create_task(read(reader))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    camera.register_text_stream_handler("vision", on_text)
    await camera.connect(URL, token("camera", room_name))
    source = rtc.VideoSource(W, H)
    track = rtc.LocalVideoTrack.create_video_track("camera", source)
    await camera.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    )
    image = people()

    async def publish() -> None:
        while True:
            rgba = image.convert("RGBA")
            source.capture_frame(rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, rgba.tobytes()))
            await asyncio.sleep(0.1)

    publisher = asyncio.create_task(publish())
    log = open("/tmp/agent.log", "wb")  # noqa: SIM115  # closed after the agent exits
    agent = subprocess.Popen(
        [sys.executable, "src/agent.py", "connect", "--room", room_name],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end and not observations:
            await asyncio.sleep(0.5)
        phase = "people"
        await asyncio.sleep(8)
        image = Image.new("RGB", (W, H), "white")
        phase = "switching"
        await asyncio.sleep(2)
        phase = "blank"
        await asyncio.sleep(8)
        # The camera leaves while the agent keeps running: the billed session must close.
        phase = "gone"
        publisher.cancel()
        await camera.local_participant.unpublish_track(track.sid)
        left_at = time.monotonic()
        await asyncio.sleep(8)
    finally:
        agent.terminate()
        try:
            agent.wait(15)
        except subprocess.TimeoutExpired:
            agent.kill()
        log.close()
        publisher.cancel()
        await camera.disconnect()

    with_people = [len(o.get("items", [])) for ph, o in observations if ph == "people"]
    blank = [len(o.get("items", [])) for ph, o in observations if ph == "blank"]
    check("agent publishes observations on the vision topic", bool(observations), f"n={len(observations)}")
    check(
        "people photo: every observation has people",
        len(with_people) >= 3 and all(n > 0 for n in with_people),
        f"n={len(with_people)} counts={sorted(set(with_people))}",
    )
    check(
        "blank frame: every observation is empty",
        len(blank) >= 3 and all(n == 0 for n in blank),
        f"n={len(blank)} counts={sorted(set(blank))}",
    )
    if observations:
        meta = observations[-1][1].get("_vlmrun", {})
        check("observation carries timing", "latency_ms" in meta and "gateway_ms" in meta, json.dumps(meta))

    agent_log = Path("/tmp/agent.log").read_text(errors="replace")
    check(
        "session closes when the camera leaves",
        "closing gateway session" in agent_log,
        "agent log shows the close" if "closing gateway session" in agent_log else "no close in log",
    )
    # Frames already in flight when the camera leaves may still be answered; after that
    # (2 s covers LiveKit's unsubscribe delay) nothing more may arrive.
    late = [t - left_at for t in arrived if left_at and t > left_at + 2.0]
    check(
        "nothing published 2 s after the camera left",
        not late,
        f"in-flight after leave={sum(1 for t in arrived if left_at and t > left_at)} late={len(late)}",
    )
    errors = [line for line in agent_log.splitlines() if "frame failed" in line or "Exception" in line]
    check("agent log has no frame errors", not errors, " | ".join(errors[:3])[:300])
    print("--- agent log tail ---")
    print(agent_log[-3000:])

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
