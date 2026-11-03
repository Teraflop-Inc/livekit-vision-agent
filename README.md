# Live vision agent

A [LiveKit Agents](https://docs.livekit.io/agents/) worker that joins a room, watches the camera or screen that was published there, and posts what the model sees back into the room as JSON.

Inference goes to the VLM Run realtime socket, `wss://gateway.vlm.run/v1/ws`. Each message after the session opens is a single JPEG, sent as raw bytes, and the reply is the model's result for that frame. The gateway bills the seconds a session is open, so the session opens on the first camera frame and closes when the camera leaves or sends no new frame for 5 seconds.

The page shows your camera immediately. The JSON panel updates when each result arrives, with the round-trip on the first line.

## Test it

```bash
uv run pytest                    # unit tests; live socket tests run when VLMRUN_API_KEY is set
uv run python e2e/room_test.py   # a real LiveKit room; needs the LiveKit and VLM Run keys
```

## Run it

You need [uv](https://docs.astral.sh/uv/), a LiveKit project, and a VLM Run API key.

```bash
cp .env.example .env.local
uv sync --extra dev
```

In two terminals:

```bash
uv run src/agent.py dev
uv run frontend/serve.py
```

Open http://localhost:8080 and click **Share camera** or **Share screen**.

## What the agent sends

`session.create` names `VLMRUN_MODEL`. `VLMRUN_METHOD` selects the method (`pose` for ViTPose and HaMeR, `segment` for SAM). `VLMRUN_PROMPT` is the session prompt.

The model must have a `realtime` block. A generative chat model is refused on this socket. The default is `usyd-community/vitpose-plus-large`.

`VISION_FPS` is a local cap. The session also enforces the model's `max_fps`, and a frame that arrives sooner comes back as `frame.dropped`. The agent keeps only the newest frame, so a slow result never queues an old one.

Each observation is a text stream on the topic `vision`. `_vlmrun.latency_ms` is the time from encoding the frame to receiving the result.

## Deploy

```bash
docker build -t vlmrun-vision-agent .
docker run --env-file .env.local vlmrun-vision-agent
```

LiveKit Cloud:

```bash
lk agent create
lk agent deploy
```
