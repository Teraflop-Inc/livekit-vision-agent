import json

from gateway import message_content, observation, sample_interval


def test_sample_interval_uses_the_slower_cap():
    assert sample_interval(8, 20) == 1 / 8
    assert sample_interval(30, 4) == 1 / 4
    assert sample_interval(0, 0) == 0.5


def test_observation_parses_a_json_string_and_drops_a_mask():
    body = json.dumps({"items": [{"label": "person"}], "mask": {"data": "x" * 500}})
    event = {
        "type": "result",
        "frame": 3,
        "latency_ms": 12.5,
        "response": {"choices": [{"message": {"content": body}}]},
    }
    parsed = observation(event, model="usyd-community/vitpose-plus-large", latency_ms=40)
    assert parsed["items"] == [{"label": "person"}]
    assert parsed["mask"]["data"].startswith("<")
    assert parsed["_vlmrun"]["latency_ms"] == 40
    assert parsed["_vlmrun"]["frame"] == 3
    assert message_content(event["response"]) == body
