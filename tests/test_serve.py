from __future__ import annotations

import base64
import importlib.util
import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "frontend" / "serve.py"


def load_serve(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "a-secret-that-is-long-enough-for-hs256-signing")
    monkeypatch.setenv("DEMO_ROOM", "test-room")
    spec = importlib.util.spec_from_file_location("serve", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jwt_claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def test_token_endpoint_grants_publish_to_the_demo_room(monkeypatch):
    serve = load_serve(monkeypatch)
    server = HTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/token") as resp:
            body = json.loads(resp.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html") as resp:
            page = resp.read()
    finally:
        server.shutdown()

    assert body["url"] == "wss://example.livekit.cloud"
    video = jwt_claims(body["token"])["video"]
    assert video["room"] == "test-room"
    assert video["roomJoin"] and video["canPublish"] and video["canSubscribe"]
    assert b"<html" in page.lower(), "the demo page is served"
    assert b"VLMRUN_API_KEY" not in page, "the page never holds the VLM Run key"
