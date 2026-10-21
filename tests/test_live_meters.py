from server import make_server


class MeterBridge:
    """Plays a scripted timeline: each exec of the sampling code advances one row."""

    def __init__(self, timeline, playing=True):
        self.timeline = list(timeline)
        self.playing = playing
        self.calls = []

    def request(self, method, params):
        self.calls.append(params.get("code") or params.get("expr"))
        code = params.get("code", "")
        if method == "eval":
            return self.timeline[0][0] if self.timeline else 0.0
        if "start_playing" in code:
            self.playing = True
            return None
        if "stop_playing" in code:
            self.playing = False
            return None
        if "current_song_time =" in code:
            return None
        song_time, master, tracks = self.timeline.pop(0) if len(self.timeline) > 1 else self.timeline[0]
        return [song_time, self.playing, master, tracks]


def _call(server, arguments):
    response = server.handle({
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "live_meters", "arguments": arguments},
    })
    return response["result"]["structuredContent"]


TIMELINE = [
    (0.0, 0.2, [("Drums", 0.5), ("Pad", 0.4), ("Spare", 0.0)]),
    (4.0, 0.3, [("Drums", 0.6), ("Pad", 0.2), ("Spare", 0.0)]),
    (8.0, 0.9, [("Drums", 0.9), ("Pad", 0.1), ("Spare", 0.0)]),
    (12.0, 0.8, [("Drums", 0.8), ("Pad", 0.1), ("Spare", 0.0)]),
    (16.0, 0.1, [("Drums", 0.0), ("Pad", 0.3), ("Spare", 0.0)]),
]


def test_meters_summarise_tracks_sections_and_silence():
    bridge = MeterBridge(TIMELINE)
    result = _call(make_server(bridge), {
        "play_from": 0, "until_beat": 16, "interval": 0.02,
        "sections": [{"name": "Intro", "start": 0, "end": 8}, {"name": "Drop", "start": 8, "end": 16}],
    })
    assert result["samples"] == 5
    assert result["beats"] == [0.0, 16.0]
    assert result["master"] == {"peak": 0.9, "mean": 0.46}
    assert result["tracks"]["Drums"] == {"peak": 0.9, "mean": 0.56}
    assert result["silent"] == ["Spare"]
    assert result["sections"]["Intro"]["master"] == {"peak": 0.3, "mean": 0.25}
    assert result["sections"]["Drop"]["loudest"][0] == {"track": "Drums", "peak": 0.9}
    assert any("start_playing" in (c or "") for c in bridge.calls)
    assert bridge.playing is False


def test_meters_can_filter_tracks_and_keep_playing():
    bridge = MeterBridge(TIMELINE)
    result = _call(make_server(bridge), {"play_from": 0, "until_beat": 8, "tracks": ["Pad"], "keep_playing": True, "interval": 0.02})
    assert list(result["tracks"]) == ["Pad"]
    assert bridge.playing is True


def test_meters_stop_early_and_hint_when_nothing_plays():
    bridge = MeterBridge([(0.0, 0.0, [("Drums", 0.0)])], playing=False)
    result = _call(make_server(bridge), {"duration": 5, "interval": 0.02})
    assert result["samples"] == 4
    assert "play_from" in result["hint"]
    assert not any("start_playing" in (c or "") for c in bridge.calls)
