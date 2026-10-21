"""Offline audio analysis for samples: key, tempo, root note and loudness.

Runs in the MCP server process (not in Live), so it can use numpy. soundfile is used
when installed; otherwise PCM/float WAV files are read with a small RIFF parser.
"""

from __future__ import annotations

import math
import os
import re
import struct
from typing import Any

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_ALIASES = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#", "CB": "B", "FB": "E", "E#": "F", "B#": "C"}
# Krumhansl-Kessler key profiles.
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_KEY_PATTERN = re.compile(r"([A-Ga-g])([#b♯♭]?)\s*(major|minor|maj|min|m)?", re.IGNORECASE)
_LOOP_BEATS = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_FILES = 64


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("analyze_audio needs numpy: pip install 'ableton-live-mcp[analysis]'") from exc
    return numpy


def analyze_audio(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    paths = list(params.get("file_paths") or [])
    if params.get("file_path"):
        paths.insert(0, params["file_path"])
    if not paths:
        raise ValueError("file_path or file_paths is required")
    if len(paths) > MAX_FILES:
        raise ValueError("at most %d files per call" % MAX_FILES)
    options = {
        "max_seconds": float(params.get("max_seconds") or 90.0),
        "bpm_range": tuple(params.get("bpm_range") or (70.0, 180.0)),
        "target_key": params.get("target_key"),
        "target_tempo": params.get("target_tempo"),
    }
    results = []
    for path in paths:
        try:
            results.append(analyze_file(path, **options))
        except Exception as exc:
            results.append({"file": path, "error": str(exc)})
    return results[0] if len(results) == 1 and params.get("file_path") and not params.get("file_paths") else {"results": results}


def analyze_file(path: str, max_seconds: float = 90.0, bpm_range=(70.0, 180.0), target_key=None, target_tempo=None) -> dict[str, Any]:
    np = _numpy()
    samples, rate, channels, total_frames = read_audio(path, max_seconds)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    duration = total_frames / float(rate)
    result: dict[str, Any] = {
        "file": path,
        "duration": round(duration, 4),
        "sample_rate": rate,
        "channels": channels,
        "analyzed_seconds": round(len(mono) / float(rate), 3),
    }
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    result["peak_dbfs"] = _db(peak)
    result["rms_dbfs"] = _db(rms)
    result["lufs"] = integrated_loudness(samples if samples.ndim == 2 else samples[:, None], rate)
    if peak <= 1e-6:
        result["silent"] = True
        return result

    onsets = count_onsets(mono, rate)
    is_loop = onsets >= 3 or (duration >= 2.0 and _fits_loop_length(duration, bpm_range))
    result["kind"] = "loop" if is_loop and duration >= 0.5 else "one_shot"
    if result["kind"] == "loop":
        envelope, hop_seconds = onset_envelope(mono, rate)
        result.update(estimate_tempo(envelope, hop_seconds, duration, bpm_range))
        result.update(estimate_key(mono, rate) or {})
    else:
        root = estimate_root(mono, rate)
        if root:
            result["root"] = root

    if target_key and result.get("key"):
        result["to_target_key"] = transpose_between(result["key"], target_key)
    if target_key and result.get("root"):
        result["root"]["to_target_key"] = transpose_between(NOTE_NAMES[result["root"]["midi"] % 12], target_key, match_tonic=True)
    if target_tempo and result.get("bpm"):
        ratio = float(target_tempo) / result["bpm"]
        result["to_target_tempo"] = {"ratio": round(ratio, 4), "repitch_semitones": round(12 * math.log2(ratio), 2)}
    return result


# --- reading -----------------------------------------------------------------------

def read_audio(path: str, max_seconds: float = 90.0):
    """Return (float32 samples [frames, channels], rate, channels, total_frames)."""
    np = _numpy()
    if not os.path.isfile(path):
        raise FileNotFoundError("no such file: %s" % path)
    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        try:
            info = soundfile.info(path)
            frames = min(info.frames, int(max_seconds * info.samplerate)) if max_seconds else info.frames
            data, rate = soundfile.read(path, frames=frames, dtype="float32", always_2d=True)
            return data, int(rate), info.channels, info.frames
        except RuntimeError:
            if not path.lower().endswith(".wav"):
                raise
    if not path.lower().endswith((".wav", ".wave")):
        raise RuntimeError("only WAV is supported without soundfile: pip install soundfile")
    return _read_wav(path, max_seconds, np)


def _read_wav(path, max_seconds, np):
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
            raise RuntimeError("not a RIFF/WAVE file: %s" % path)
        fmt = None
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                raise RuntimeError("WAV file has no data chunk")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                body = handle.read(size + (size & 1))
                tag, channels, rate, _bps, align, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, channels, rate, align, bits)
            elif cid == b"data":
                if fmt is None:
                    raise RuntimeError("WAV data before fmt chunk")
                tag, channels, rate, align, bits = fmt
                total = size // align if size != 0xFFFFFFFF else (os.path.getsize(path) - handle.tell()) // align
                frames = min(total, int(max_seconds * rate)) if max_seconds else total
                raw = handle.read(frames * align)
                return _decode_pcm(raw, tag, channels, bits, np), rate, channels, total
            else:
                handle.seek(size + (size & 1), 1)


def _decode_pcm(raw, tag, channels, bits, np):
    width = bits // 8
    usable = len(raw) - len(raw) % (width * channels)
    raw = raw[:usable]
    if tag == 3 and bits in (32, 64):
        data = np.frombuffer(raw, dtype="<f%d" % width).astype(np.float32)
    elif tag == 1 and bits == 8:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits in (16, 32):
        data = np.frombuffer(raw, dtype="<i%d" % width).astype(np.float32) / float(2 ** (bits - 1))
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / float(2 ** 23)
    else:
        raise RuntimeError("unsupported WAV encoding (format %d, %d bits); install soundfile" % (tag, bits))
    return data.reshape(-1, channels)


# --- loudness ----------------------------------------------------------------------

def _db(value: float):
    return round(20.0 * math.log10(value), 2) if value > 1e-9 else None


def _biquad_response(b, a, freqs, rate, np):
    z = np.exp(-2j * np.pi * freqs / rate)
    return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)


def _k_weighting(freqs, rate, np):
    # BS.1770 pre-filter (high shelf) and RLB high-pass, recomputed for any sample rate.
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh, vb = 10 ** (gain / 20.0), (10 ** (gain / 20.0)) ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = _biquad_response(
        [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0], freqs, rate, np)
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1.0 + k / q + k * k
    highpass = _biquad_response([1.0, -2.0, 1.0], [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0], freqs, rate, np)
    return np.abs(shelf * highpass)


def integrated_loudness(samples, rate):
    """Gated BS.1770 integrated loudness in LUFS (K-weighting applied in the frequency domain)."""
    np = _numpy()
    frames = samples.shape[0]
    block, step = int(0.4 * rate), int(0.1 * rate)
    if frames < block:
        pad = np.zeros((block - frames, samples.shape[1]), dtype=samples.dtype)
        samples, frames = np.vstack([samples, pad]), block
    size = 1 << (frames - 1).bit_length()
    gain = _k_weighting(np.fft.rfftfreq(size, 1.0 / rate), rate, np)
    weighted = np.fft.irfft(np.fft.rfft(samples.astype(np.float64), n=size, axis=0) * gain[:, None], n=size, axis=0)[:frames]
    weights = [1.0, 1.0, 1.0, 1.41, 1.41][: samples.shape[1]] if samples.shape[1] <= 5 else [1.0] * samples.shape[1]
    power = (weighted ** 2) @ np.asarray(weights, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, frames - block + 1, step)
    energy = (cumulative[starts + block] - cumulative[starts]) / block
    energy = energy[energy > 0]
    loud = energy[-0.691 + 10 * np.log10(energy) > -70.0] if energy.size else energy
    if not loud.size:
        return None
    relative = -0.691 + 10 * math.log10(loud.mean()) - 10.0
    gated = loud[-0.691 + 10 * np.log10(loud) > relative]
    return round(-0.691 + 10 * math.log10((gated if gated.size else loud).mean()), 2)


# --- rhythm ------------------------------------------------------------------------

def _stft_magnitude(mono, n_fft, hop, np):
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    count = 1 + (len(mono) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return np.abs(np.fft.rfft(mono[index] * np.hanning(n_fft)[None, :], axis=1))


def onset_envelope(mono, rate):
    np = _numpy()
    hop = max(1, int(round(rate * 0.01)))
    spectrum = np.log1p(100.0 * _stft_magnitude(mono, 2048 if rate <= 48000 else 4096, hop, np))
    flux = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    width = max(1, min(31, len(flux)))
    flux -= np.convolve(flux, np.ones(width) / width, mode="same")[: len(flux)]
    return np.maximum(flux, 0.0), hop / float(rate)


def count_onsets(mono, rate, rise_db=9.0, min_gap=0.06):
    """Attacks in the 10 ms level envelope: rises of rise_db above the lowest level since the last attack."""
    np = _numpy()
    frame = max(1, int(rate * 0.01))
    count = len(mono) // frame
    if not count:
        return 0
    rms = np.sqrt(np.mean(mono[: count * frame].reshape(count, frame).astype(np.float64) ** 2, axis=1))
    floor = max(float(rms.max()) * 10 ** (-50 / 20.0), 1e-9)
    level = 20 * np.log10(np.concatenate([[floor], np.maximum(rms, floor)]))  # leading silence: first hit counts
    onsets, low, last, gap = 0, level[0], -len(level), int(round(min_gap / 0.01))
    for i, value in enumerate(level):
        low = min(low, value)
        if value - low >= rise_db and i - last >= gap:
            onsets, low, last = onsets + 1, value, i
    return onsets


def _fits_loop_length(duration, bpm_range, tolerance=0.002):
    """True when the length is a whole number of bars at an integer tempo in range."""
    for beats in _LOOP_BEATS[2:]:
        bpm = 60.0 * beats / duration
        if bpm_range[0] <= bpm <= bpm_range[1] and abs(bpm - round(bpm)) / bpm < tolerance:
            return True
    return False


def estimate_tempo(envelope, hop_seconds, duration, bpm_range=(70.0, 180.0)):
    np = _numpy()
    lo, hi = float(bpm_range[0]), float(bpm_range[1])
    centered = envelope - envelope.mean()
    acf = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    if acf[0] <= 0:
        return {}
    acf = acf / acf[0]

    def strength(bpm):
        total = 0.0
        for multiple, weight in ((1, 1.0), (2, 0.5), (4, 0.25), (0.5, 0.25)):
            lag = 60.0 / bpm * multiple / hop_seconds
            if 1 <= lag < len(acf) - 1:
                i = int(lag)
                total += weight * (acf[i] + (acf[i + 1] - acf[i]) * (lag - i))
        return total

    grid = np.arange(lo, hi + 0.05, 0.1)
    scores = np.array([strength(b) for b in grid])
    best = float(grid[int(np.argmax(scores))])
    confidence = float(np.clip((scores.max() - np.median(scores)) / (abs(scores.max()) + 1e-9), 0.0, 1.0))
    out = {"bpm": round(best, 1), "tempo_confidence": round(confidence, 2)}
    # Loops are cut to whole bars, so duration gives an exact tempo when one fits.
    fits = [(60.0 * beats / duration, beats) for beats in _LOOP_BEATS if lo <= 60.0 * beats / duration <= hi]
    if fits:
        bpm, beats = min(fits, key=lambda item: abs(item[0] - best) / best)
        if abs(bpm - best) / best < 0.04:
            out.update({"bpm": round(bpm, 2), "beats": beats, "bars": beats / 4.0, "tempo_from": "loop_length"})
        else:
            out["loop_length_bpm"] = [{"bpm": round(b, 2), "beats": n} for b, n in fits]
    out.setdefault("tempo_from", "onsets")
    return out


# --- pitch -------------------------------------------------------------------------

def _spectral_peaks(spectrum, np):
    """Zero everything except local maxima across frequency (5 bins) above 5% of the frame peak."""
    padded = np.pad(spectrum, ((0, 0), (2, 2)))
    neighbours = np.stack([padded[:, k:k + spectrum.shape[1]] for k in range(5)])
    keep = (spectrum >= neighbours.max(axis=0)) & (spectrum > 0.05 * spectrum.max(axis=1, keepdims=True))
    return np.where(keep, spectrum, 0.0)


def estimate_tuning(spectrum, freqs, np):
    """Cents offset from A440, from parabolically interpolated spectral peaks 150 Hz-3 kHz."""
    band = np.flatnonzero((freqs >= 150) & (freqs <= 3000))
    if len(band) < 3:
        return 0.0
    sub = spectrum[:, band[0] - 1:band[-1] + 2]
    frames, bins = np.nonzero(_spectral_peaks(sub, np)[:, 1:-1])
    if not len(bins):
        return 0.0
    a, b, c = (np.log(sub[frames, bins + k] + 1e-12) for k in (0, 1, 2))
    denom = a - 2 * b + c
    shift = np.clip(0.5 * (a - c) / np.where(denom == 0, -1e-12, denom), -0.5, 0.5)
    hz = (band[0] + bins + shift) * (freqs[1] - freqs[0])
    cents = 1200 * np.log2(hz / 440.0)
    angle = np.angle(np.sum(sub[frames, bins + 1] * np.exp(2j * np.pi * cents / 100.0)))
    return float(angle / (2 * np.pi) * 100.0)


def _pitch_class_profile(mono, rate, np):
    n_fft = 8192 if rate <= 48000 else 16384
    mono = mono[: int(rate * 60)]
    spectrum = _stft_magnitude(mono, n_fft, n_fft // 4, np)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    tuning = estimate_tuning(spectrum, freqs, np)
    band = (freqs >= 80.0) & (freqs <= 4000.0)
    classes = np.mod(np.round(69 + 12 * np.log2(freqs[band] / 440.0) - tuning / 100.0).astype(int), 12)
    peaks = np.sqrt(_spectral_peaks(spectrum, np)[:, band])
    chroma = np.array([peaks[:, classes == pc].sum() for pc in range(12)])
    return chroma, tuning


def estimate_key(mono, rate):
    np = _numpy()
    chroma, tuning = _pitch_class_profile(mono, rate, np)
    if chroma.sum() <= 0:
        return None
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            rotated = np.roll(profile, tonic)
            scores.append((float(np.corrcoef(chroma, rotated)[0, 1]), "%s %s" % (NOTE_NAMES[tonic], mode)))
    scores.sort(reverse=True)
    best, runner = scores[0], scores[1]
    return {
        "key": best[1],
        "key_confidence": round(max(0.0, best[0]), 2),
        "key_alternative": runner[1],
        "camelot": camelot(best[1]),
        "tuning_cents": round(tuning, 1),
    }


def estimate_root(mono, rate):
    """Strongest low fundamental, for tuning kicks, 808s and bass one-shots."""
    np = _numpy()
    start = int(np.argmax(np.abs(mono)))
    segment = mono[start + int(0.02 * rate): start + int(0.02 * rate) + rate]
    if len(segment) < int(0.05 * rate):
        segment = mono[start:]
    if len(segment) < 256:
        return None
    size = 1 << max(16, (len(segment) - 1).bit_length())
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=size))
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    band = (freqs >= 25.0) & (freqs <= 1000.0)
    if not band.any() or spectrum[band].max() <= 0:
        return None
    # Harmonic product over 3 harmonics favours the fundamental over louder overtones.
    product = spectrum.copy()
    for h in (2, 3):
        decimated = spectrum[::h]
        product[: len(decimated)] *= decimated ** 0.5
        product[len(decimated):] = 0
    idx = np.flatnonzero(band)
    peak = int(idx[np.argmax(product[idx])])
    if 0 < peak < len(spectrum) - 1:
        a, b, c = np.log(spectrum[peak - 1: peak + 2] + 1e-12)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        shift = 0.0
    freq = float((peak + shift) * rate / size)
    midi = 69 + 12 * math.log2(freq / 440.0)
    nearest = int(round(midi))
    clarity = float(spectrum[peak] / (spectrum[band].mean() + 1e-12))
    return {
        "frequency": round(freq, 2),
        "note": "%s%d" % (NOTE_NAMES[nearest % 12], nearest // 12 - 1),
        "midi": nearest,
        "cents": round((midi - nearest) * 100.0, 1),
        "clarity": round(min(clarity / 50.0, 1.0), 2),
    }


# --- key helpers -------------------------------------------------------------------

def parse_key(text: str):
    """'A minor', 'Am', 'F#', 'Bb maj', '8A' (Camelot) -> (pitch class, 'major'|'minor')."""
    value = str(text).strip()
    camel = re.fullmatch(r"(1[0-2]|[1-9])([AB])", value.upper())
    if camel:
        number, letter = int(camel.group(1)), camel.group(2)
        major_tonic = (number - 8) * 7 % 12
        return (major_tonic + 9) % 12 if letter == "A" else major_tonic, "minor" if letter == "A" else "major"
    match = _KEY_PATTERN.fullmatch(value)
    if not match:
        raise ValueError("unrecognised key %r; use e.g. 'A minor', 'F#m', 'Bb' or '8A'" % text)
    letter, accidental, mode = match.groups()
    name = letter.upper() + {"": "", "#": "#", "♯": "#", "b": "B", "♭": "B"}[accidental.lower()]
    tonic = NOTE_NAMES.index(_FLAT_ALIASES.get(name, name))
    return tonic, "minor" if mode and mode.lower().startswith("m") and not mode.lower().startswith("maj") else "major"


def camelot(key: str) -> str:
    tonic, mode = parse_key(key)
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    return "%d%s" % ((major_tonic * 7 + 7) % 12 + 1, "B" if mode == "major" else "A")


def transpose_between(source: str, target: str, match_tonic: bool = False) -> dict[str, Any]:
    """Smallest transpose (-6..+5 semitones) from source onto target.

    Keys of different modes are matched through the relative key (C major -> A minor is 0).
    With match_tonic (one-shots), source is a note and is moved onto the target's tonic.
    """
    s_tonic, s_mode = parse_key(source)
    t_tonic, t_mode = parse_key(target)
    if not match_tonic and s_mode != t_mode:
        t_tonic = (t_tonic + 3) % 12 if t_mode == "minor" else (t_tonic - 3) % 12
    shift = (t_tonic - s_tonic) % 12
    if shift > 5:
        shift -= 12
    return {"semitones": shift} if match_tonic else {"semitones": shift, "same_mode": s_mode == t_mode}
