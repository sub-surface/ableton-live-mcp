import math
import types

from test_remote_bridge_fake_live import FakeClip, FakeDevice, FakeParameter, FakeTrack, FakeVector, make_bridge


class FakeMixerParam(FakeParameter):
    """Volume-style parameter: 0.85 shows 0 dB, 40*log10 curve, like Live's fader."""

    def str_for_value(self, value):
        return "-inf dB" if value <= 0 else "%.1f dB" % (40.0 * math.log10(value / 0.85))


class ArrangeTrack(FakeTrack):
    def __init__(self, name, midi):
        FakeTrack.__init__(self, name)
        self.has_midi_input = midi
        self.color_index = None
        self.mixer_device = types.SimpleNamespace(
            volume=FakeMixerParam("Volume", 0.85), panning=FakeParameter("Pan", 0.0, -1.0, 1.0),
            sends=[FakeParameter("A", 0.0), FakeParameter("B", 0.0)])

    def create_midi_clip(self, start, length):
        clip = FakeClip("MIDI", start_time=start, end_time=start + length)
        clip._notes = []
        self.arrangement_clips.append(clip)
        return clip

    def create_audio_clip(self, file_path, start):
        if "missing" in file_path:
            raise RuntimeError("The provided path does not appear to point to a valid audio file")
        clip = FakeClip(file_path.rsplit("/", 1)[-1], midi=False, start_time=start, end_time=start + 8.0)
        self.arrangement_clips.append(clip)
        return clip

    def delete_clip(self, clip):
        self.arrangement_clips.remove(clip)

    def insert_device(self, device_name, device_index=-1):
        device = FakeDevice()
        device.name = device.class_name = device_name
        device.parameters = FakeVector([FakeParameter("Frequency", 0.5)])
        self.devices.append(device)


def make_arrange_bridge(monkeypatch):
    bridge, song, _app = make_bridge(monkeypatch)
    song.tracks = FakeVector([ArrangeTrack("Existing", midi=False)])

    def create(midi):
        return lambda index=-1: song.tracks.append(ArrangeTrack("New", midi))

    song.create_midi_track, song.create_audio_track = create(True), create(False)
    return bridge, song


def test_arrange_creates_tracks_places_clips_and_sets_mixer(monkeypatch):
    bridge, song = make_arrange_bridge(monkeypatch)
    result = bridge._rpc_arrange({"tempo": 160, "tracks": [
        {"name": "Break", "color": 14, "volume": "-6 dB", "sends": {"B": 0.3}, "devices": ["Auto Filter"],
         "params": [{"device": "Auto Filter", "param": "Frequency", "value": 0.7}],
         "clips": [{"file": "C:/s/amen.wav", "start": 64, "repeat": 3, "warp": False}]},
        {"name": "Sub", "instrument": "Operator",
         "clips": [{"start": 0, "length": 8, "repeat": 2, "every": 16, "notes": [[33, 0, 4, 100], {"pitch": 29, "start_time": 4, "duration": 4}]}]},
    ]})
    assert result["errors"] == []
    assert song.tempo == 160.0
    names = [t.name for t in song.tracks]
    assert names == ["Existing", "Break", "Sub"]
    brk, sub = song.tracks[1], song.tracks[2]
    assert [(c.start_time, c.warping, c.color_index) for c in brk.arrangement_clips] == [(64.0, False, 14), (72.0, False, 14), (80.0, False, 14)]
    assert abs(40.0 * math.log10(brk.mixer_device.volume.value / 0.85) + 6.0) < 0.05
    assert brk.mixer_device.sends[1].value == 0.3
    assert brk.devices[0].parameters[0].value == 0.7
    assert sub.has_midi_input and sub.devices[0].name == "Operator"
    assert [c.start_time for c in sub.arrangement_clips] == [0.0, 16.0]
    assert [(n.pitch, n.start_time, n.velocity) for n in sub.arrangement_clips[0]._notes] == [(33, 0.0, 100.0), (29, 4.0, 100.0)]


def test_arrange_clears_only_the_requested_range_on_existing_tracks(monkeypatch):
    bridge, song = make_arrange_bridge(monkeypatch)
    existing = song.tracks[0]
    for start in (0.0, 8.0, 16.0):
        existing.create_audio_clip("C:/s/old.wav", start)
    result = bridge._rpc_arrange({"tracks": [{"name": "Existing", "clear": [8, 16], "clips": [{"file": "C:/s/new.wav", "start": 8}]}]})
    assert result["tracks"] == [{"name": "Existing", "created": False, "clips": 1}]
    assert sorted((c.start_time, c.name) for c in existing.arrangement_clips) == [(0.0, "old.wav"), (8.0, "new.wav"), (16.0, "old.wav")]


def test_arrange_collects_errors_without_stopping(monkeypatch):
    bridge, song = make_arrange_bridge(monkeypatch)
    result = bridge._rpc_arrange({"tracks": [
        {"name": "Existing", "type": "midi"},
        {"name": "Ghost", "create": False},
        {"name": "Bad", "clips": [{"file": "C:/s/missing.wav", "start": 0}, {"file": "C:/s/ok.wav", "start": 8}]},
    ]})
    assert [e.split(":")[0] for e in result["errors"]] == ["track 'Existing'", "track 'Ghost'", "clip on 'Bad' at 0"]
    assert result["tracks"] == [{"name": "Bad", "created": True, "clips": 1}]
