"""Rebuild and read the existing synthetic T1/T2 example from a native recording."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from eeg_jev.benchmark.v13_dataset import BenchmarkV13
from score_linked_demo import prepare_inputs, score, write_jsonl


def write_recording(root):
    raw = root / "native"
    raw.mkdir()
    t = np.arange(22 * 250) / 250
    signal = np.column_stack((
        20 * np.sin(2 * np.pi * 9 * t) + 3 * np.cos(2 * np.pi * 4 * t),
        16 * np.cos(2 * np.pi * 12 * t) + 2 * np.sin(2 * np.pi * 20 * t),
    )).astype("<f4")
    signal.tofile(raw / "demo.eeg")
    (raw / "demo.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\nDataFile=demo.eeg\nMarkerFile=demo.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\n"
        "SamplingInterval=4000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\nCh1=C3,,1,uV\nCh2=C4,,1,uV\n",
        encoding="utf-8",
    )
    (raw / "demo.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\nDataFile=demo.eeg\n[Marker Infos]\n",
        encoding="utf-8",
    )
    config = root / "raw.json"
    config.write_text(json.dumps({"roots": {"demo": "native"}, "recordings": "recordings.jsonl"}), encoding="utf-8")
    write_jsonl(root / "recordings.jsonl", [{
        "source": "demo", "recording": "demo-recording", "reader": "brainvision",
        "path": "demo.vhdr", "fs": 250, "duration_s": 22,
        "channels": ["C3", "C4"], "reference_channels": [], "notch_hz": [],
        "segment_start_samples": {"demo-segment": 601},
    }])
    return config, signal.T.astype(np.float64)


def main():
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        metadata = workspace / "metadata_bundle"
        metadata.mkdir()
        prepare_inputs(metadata)
        config, raw = write_recording(workspace)
        row = {"source": "demo", "split": "val", "recording": "demo-recording", "length_s": 1}
        start = 601 / 250 + 0.114
        write_jsonl(metadata / "frozen_v13/train/demo_windows.jsonl", [
            dict(row, window_id="w1", segment="demo-segment", start_s=0.114, abs_start_s=round(start, 4)),
            dict(row, window_id="w2", start_s=4.0),
        ])
        t2 = metadata / "new_tracks/validated/T2_demo.jsonl"
        events = [json.loads(line) for line in t2.read_text().splitlines()]
        for event, start_s in zip(events, (2.0, 10.0)):
            event.update(recording="demo-recording", start_s=start_s, length_s=8)
        write_jsonl(t2, events)

        corpus = workspace / "corpus"
        subprocess.run([
            sys.executable, "-I", "-B", str(HERE.parent / "tools/rebuild_v13.py"),
            "--metadata-root", str(metadata), "--raw-config", str(config), "--out", str(corpus),
        ], check=True)
        data = BenchmarkV13(corpus)
        data.require_frozen()
        x200 = resample_poly(raw, 4, 5, axis=1, padtype="line")

        def expected(start_s, seconds):
            first = round(200 * start_s)
            window = x200[:, first:first + 200 * seconds]
            return (window - window.mean(axis=1, keepdims=True)) / window.std(axis=1, keepdims=True)

        for key, start_s in (("w1", start), ("w2", 4.0)):
            observed = data.core_eeg("demo", "val", key)
            assert observed.dtype == np.float32
            np.testing.assert_allclose(observed, expected(start_s, 1), atol=1e-6, rtol=1e-6)
        t1 = list(data.new_questions("T1", "val"))
        assert len(t1) == 4 and all(window.shape == (2, 200) for row in t1 for window in data.new_eeg(row))
        long = list(data.new_questions("T2", "val"))
        assert len(long) == 2
        for row, start_s in zip(long, (2.0, 10.0)):
            np.testing.assert_allclose(data.new_eeg(row)[0], expected(start_s, 8), atol=1e-6, rtol=1e-6)
        choices = {f"pair1/swap{swap}/{target}": 0 for swap in (0, 1) for target in (0, 1)}
        score(corpus, "T1", choices, 0.5)
        score(corpus, "T2", {"cue-a": 0, "cue-b": 1}, 1.0)


if __name__ == "__main__":
    main()
