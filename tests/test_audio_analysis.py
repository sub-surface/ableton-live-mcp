import math
import struct
import wave

import pytest

np = pytest.importorskip("numpy")

import audio_analysis as aa
from server import make_server

RATE = 44100


def _write_wav(path, signal, rate=RATE, channels=1):
    data = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    if channels == 2:
        data = np.column_stack([data, data])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return str(path)


def _tone(freq, seconds, decay=None):
    t = np.arange(int(seconds * RATE)) / RATE
    env = np.exp(-t / decay) if decay else 1.0
    return np.sin(2 * math.pi * freq * t) * env


def _chord_loop(midi_notes_per_bar, bpm, bars):
    """Piano-ish chords on every beat, so the loop has onsets and a clear key."""
    beat = 60.0 / bpm
    out = np.zeros(int(round(bars * 4 * beat * RATE)))
    for bar in range(bars):
        notes = midi_notes_per_bar[bar % len(midi_notes_per_bar)]
        for b in range(4):
            start = int(round((bar * 4 + b) * beat * RATE))
            hit = sum(_tone(440 * 2 ** ((n - 69) / 12.0), beat, decay=beat / 3) for n in notes) / len(notes)
            out[start:start + len(hit)] += hit[: len(out) - start]
    return 0.5 * out


def test_loop_tempo_comes_from_bar_length_and_key_from_chords(tmp_path):
    # i - VI - III - VII in A minor.
    progression = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]]
    path = _write_wav(tmp_path / "loop.wav", _chord_loop(progression, 172.0, 4), channels=2)
    result = aa.analyze_audio({"file_path": path, "target_key": "C major", "target_tempo": 86})
    assert result["kind"] == "loop"
    assert result["bpm"] == pytest.approx(172.0, abs=0.01) and result["bars"] == 4.0
    assert result["tempo_from"] == "loop_length"
    assert result["key"] in ("A minor", "C major")
    assert result["to_target_key"]["semitones"] == 0
    assert result["to_target_tempo"] == {"ratio": 0.5, "repitch_semitones": -12.0}


def test_one_shot_reports_root_note_and_transpose_to_key(tmp_path):
    kick = _tone(55.0 * 2 ** (0.3 / 12), 0.8, decay=0.25)  # A1, 30 cents sharp
    path = _write_wav(tmp_path / "kick.wav", kick)
    result = aa.analyze_audio({"file_path": path, "target_key": "F minor"})
    assert result["kind"] == "one_shot" and "bpm" not in result
    assert result["root"]["note"] == "A1"
    assert result["root"]["cents"] == pytest.approx(30, abs=5)
    assert result["root"]["to_target_key"] == {"semitones": -4}


def test_loudness_matches_bs1770_reference_tone(tmp_path):
    # BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LUFS, so -20 dBFS in both channels reads -20.0.
    tone = 0.1 * _tone(997.0, 5.0)
    assert aa.analyze_file(_write_wav(tmp_path / "mono.wav", tone))["lufs"] == pytest.approx(-23.01, abs=0.1)
    result = aa.analyze_file(_write_wav(tmp_path / "stereo.wav", tone, channels=2))
    assert result["peak_dbfs"] == pytest.approx(-20.0, abs=0.05)
    assert result["lufs"] == pytest.approx(-20.0, abs=0.1)


def test_float_and_24_bit_wav_are_read_without_soundfile(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "soundfile", None)
    signal = 0.5 * _tone(220.0, 0.2)
    for tag, bits, encode in ((3, 32, lambda x: x.astype("<f4").tobytes()),
                              (1, 24, lambda x: b"".join(struct.pack("<i", int(v * 8388607))[:3] for v in x))):
        body = encode(signal)
        fmt = struct.pack("<HHIIHH", tag, 1, RATE, RATE * bits // 8, bits // 8, bits)
        path = tmp_path / ("x%d.wav" % bits)
        path.write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(body)) + b"WAVE"
                         + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body)
        samples, rate, channels, frames = aa.read_audio(str(path))
        assert (rate, channels, frames) == (RATE, 1, len(signal))
        assert np.max(np.abs(samples[:, 0] - signal)) < 1e-4


@pytest.mark.parametrize("text, expected", [
    ("A minor", (9, "minor")), ("Am", (9, "minor")), ("F#m", (6, "minor")), ("Bb", (10, "major")),
    ("Eb maj", (3, "major")), ("c min", (0, "minor")), ("8A", (9, "minor")), ("8B", (0, "major")), ("12A", (1, "minor")),
])
def test_parse_key_and_camelot_round_trip(text, expected):
    assert aa.parse_key(text) == expected
    assert aa.parse_key(aa.camelot(text)) == expected


def test_transpose_prefers_smallest_shift_and_relative_keys():
    assert aa.transpose_between("C major", "A minor") == {"semitones": 0, "same_mode": False}
    assert aa.transpose_between("A minor", "F minor") == {"semitones": -4, "same_mode": True}
    assert aa.transpose_between("D minor", "B minor")["semitones"] == -3


def test_tool_reports_per_file_errors(tmp_path):
    good = _write_wav(tmp_path / "good.wav", 0.5 * _tone(110.0, 0.5, decay=0.1))
    response = make_server(object()).handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "analyze_audio", "arguments": {"file_paths": [good, str(tmp_path / "missing.wav")]}},
    })
    results = response["result"]["structuredContent"]["results"]
    assert results[0]["root"]["note"] == "A2"
    assert "no such file" in results[1]["error"]
