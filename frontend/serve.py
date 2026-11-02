"""Serve the demo page and mint a LiveKit token at ``/token``.

The browser publishes video. The agent, not this page, holds the VLM Run key.
"""

import json
import os
import pathlib
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler

from dotenv import load_dotenv
from livekit import api

ROOT = pathlib.Path(__file__).parent
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

ROOM_NAME = os.environ.get("DEMO_ROOM", "vlmrun-vision")
PORT = int(os.environ.get("PORT", "8080"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] != "/token":
            return super().do_GET()
        token = (
            api.AccessToken()
            .with_identity(f"viewer-{uuid.uuid4().hex[:6]}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=ROOM_NAME,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
        body = json.dumps({"url": os.environ["LIVEKIT_URL"], "token": token}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    missing = [name for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"{', '.join(missing)} is not set. Copy .env.example to .env.local and fill it in.")
    print(f"demo on http://localhost:{PORT}  room {ROOM_NAME}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
