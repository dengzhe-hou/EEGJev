"""Native EEG loading and signal transforms used by the v1.3 build."""

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, iirnotch, resample_poly


FS = 200
NOTCH_Q = 30.0


def read_native(root, record):
    """Read one recording in the channel order and units in its recipe."""
    import mne
    from scipy.io import loadmat

    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Recording paths must be relative to their source root")
    path = root / relative
    kind = record["reader"]
    channels = record["channels"]
    if not channels or len(set(channels)) != len(channels):
        raise ValueError(f"{record['recording']}: invalid channel list")
    selection = record.get("channel_map", {})
    picks = [selection.get(name, name) for name in channels]
    names_needed = list(dict.fromkeys(
        name for pick in picks for name in ([pick] if isinstance(pick, str) else pick)
    ))
    fs = float(record["fs"])
    boundaries = record.get("boundaries_s")
    if kind == "gal_ws":
        ws = loadmat(path, simplify_cells=True)["ws"]
        names = list(ws["names"]["eeg"])
        data = np.asarray(ws["win"][record["lift"] - 1]["eeg"], dtype=np.float64).T
        data = data[[names.index(name) for name in names_needed]]
    elif kind == "bciciv_mat":
        mat = loadmat(path, simplify_cells=True)
        names = list(mat["nfo"]["clab"])
        data = np.asarray(mat["cnt"], dtype=np.float64).T
        data = data[[names.index(name) for name in names_needed]]
    else:
        readers = {
            "eeglab": mne.io.read_raw_eeglab,
            "edf": mne.io.read_raw_edf,
            "brainvision": mne.io.read_raw_brainvision,
        }
        if kind not in readers:
            raise ValueError(f"Unsupported reader {kind!r}")
        with readers[kind](path, preload=False, verbose="error") as raw:
            if not np.isclose(fs, raw.info["sfreq"], rtol=1e-12, atol=0.0):
                raise ValueError(f"{path}: recipe sampling rate {fs:g} != file sampling rate {raw.info['sfreq']:g}")
            indices = [raw.ch_names.index(name) for name in names_needed]
            data = raw.get_data(picks=indices) * record.get("stored_unit_scale", 1e6)
            if boundaries is None:
                boundaries = [
                    float(onset)
                    for onset, label in zip(raw.annotations.onset, raw.annotations.description)
                    if "boundary" in label.lower() or "new segment" in label.lower()
                ]

    if not fs.is_integer() or fs <= 0:
        raise ValueError(f"{path}: expected an integer native sampling rate")
    duration = data.shape[1] / fs
    if "duration_s" in record and abs(duration - record["duration_s"]) > 1.0:
        raise ValueError(f"{path}: recording and recipe durations differ")
    indices = {name: index for index, name in enumerate(names_needed)}
    if picks == names_needed:
        x = data
    else:
        rows = []
        for pick in picks:
            if isinstance(pick, str):
                rows.append(data[indices[pick]])
            else:
                first, second = pick
                rows.append(data[indices[first]] - data[indices[second]])
        x = np.stack(rows)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: recording has non-finite samples")
    return x, fs, boundaries or []


def preprocess(x, fs, channels, reference, frequencies):
    """Apply the native-rate reference and zero-phase notch before resampling."""
    if reference:
        indices = [channels.index(channel) for channel in reference]
        x = x - x[indices].mean(axis=0, keepdims=True)
    for frequency in frequencies:
        if frequency < fs / 2:
            b, a = iirnotch(frequency, NOTCH_Q, fs)
            x = filtfilt(b, a, x, axis=1, padtype="odd", padlen=min(int(3 * fs), x.shape[1] - 1))
    ratio = Fraction(FS, int(fs))
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=1, padtype="line")


def flat_runs(x, fs):
    """Find native intervals of at least 0.1 s with no change in any channel."""
    same = np.all(np.diff(x, axis=1) == 0, axis=0)
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype(np.int8), 0]))
    return [
        (start / fs, (end + 1) / fs)
        for start, end in zip(edges[::2], edges[1::2])
        if (end - start + 1) / fs >= 0.1
    ]


def check_window(x, fs, runs, boundaries, start, length, drop, long_window):
    end = start + length
    if start < drop or end > x.shape[1] / fs - 0.5:
        raise ValueError("Window lies outside the usable recording interval")
    if any(start - 0.1 < time < end + 0.1 for time in boundaries if long_window or time > drop):
        raise ValueError("Window crosses a native discontinuity")
    if any(max(start, left) < min(end, right) for left, right in runs):
        raise ValueError("Window overlaps a native flat stretch")
    samples = x[:, round(start * fs):round(end * fs)]
    if np.all(samples == samples[:, :1], axis=1).any():
        raise ValueError("Window contains a dead channel")


def cut(x200, start, length):
    """Cut at the 200 Hz sample index and standardize each channel."""
    first, n_samples = round(start * FS), round(length * FS)
    if first < 0 or first + n_samples > x200.shape[1]:
        raise ValueError(f"Window [{start}, {start + length}] exceeds resampled data")
    samples = x200[:, first:first + n_samples]
    deviation = samples.std(axis=1, keepdims=True)
    if not np.isfinite(samples).all() or (deviation == 0).any():
        raise ValueError("Invalid samples after resampling")
    return ((samples - samples.mean(axis=1, keepdims=True)) / deviation).astype(np.float32)
