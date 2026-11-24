"""Answer a fixed synthetic EEGJev-Bench fixture with a local v1.3 checkpoint and score it."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "model"))
sys.path.insert(0, str(PACKAGE / "bench"))
from eegjev.inference import FS, LinkedEEGJev
from eeg_jev.benchmark.v13_dataset import BenchmarkV13

SOURCE = "photo"
PAIR_TASK = "photo-state-1s"
EVENT_TASK = "photo/flicker frequency change/8s"
FIXTURE_SEED = 42


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def standardize(x):
    return ((x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)).astype(np.float32)


def make_fixture(root):
    """Write the waveforms and both ground-truth files before loading model weights."""
    montages = json.loads((PACKAGE / "model/eegjev/montages.json").read_text())
    channels = montages[SOURCE]
    rng = np.random.default_rng(FIXTURE_SEED)
    phase = np.linspace(0, .7, len(channels))[:, None]
    gain = np.array([1.5 if name.startswith(("O", "P")) else .5 for name in channels])[:, None]

    def background(t):
        return (.35 * np.sin(2 * np.pi * 9 * t + phase)
                + .2 * rng.normal(size=(len(channels), len(t))))

    t = np.arange(FS) / FS
    resting = background(t)
    photic = background(t) + gain * np.sin(2 * np.pi * 15 * t + phase)
    signal_dir = root / "signals"
    signal_dir.mkdir(parents=True)
    np.save(signal_dir / "pair.npy", np.stack([standardize(photic), standardize(resting)]))
    common = {"source": SOURCE, "split": "val", "subject": "synthetic-01"}
    window_ids = ["synth-window-01", "synth-window-02"]
    window_rows = [
        dict(common, window_id=wid, length_s=1, cache_ref={
            "format": "npy", "file": "signals/pair.npy", "row": i})
        for i, wid in enumerate(window_ids)
    ]
    save_rows(root / "frozen_v13/train/photo_windows.jsonl", window_rows)
    pair = dict(common, pair_id="T1/synthetic-pair", task=PAIR_TASK, attribute="state",
                windows=window_ids, labels=["photic stimulation", "resting state"], length_s=1)
    save_rows(root / "new_tracks/data/t1_photo.jsonl", [pair])

    # Both views contain exactly the same change in one continuous 12-second signal.
    t = np.arange(12 * FS) / FS
    event_second = 6
    cycles = 15 * np.minimum(t, event_second) + 20 * np.maximum(t - event_second, 0)
    stream = background(t) + gain * np.sin(2 * np.pi * cycles + phase)
    event = dict(common, task=EVENT_TASK, group_id="T2/synthetic-event", length_s=8,
                 question="In which half did the flicker frequency change?",
                 options=["first half", "second half"])
    event_views, event_rows = [], []
    for i, start in enumerate([4, 0]):
        event_views.append(standardize(stream[:, start * FS:(start + 8) * FS]))
        offset = event_second - start
        event_rows.append(dict(event, question_id=f"T2/synthetic-event/view{i+1}",
                               event_offset_s=offset, answer_index=int(offset >= 4),
                               cache_ref={"format": "npy", "file": "signals/event.npy", "row": i}))
    np.save(signal_dir / "event.npy", np.stack(event_views))
    save_rows(root / "new_tracks/validated/T2_photo.jsonl", event_rows)
    save_json(root / "metadata/tasks_v13.json", {"tracks": {
        "T1": {"cells": [PAIR_TASK]}, "T2": {"cells": [EVENT_TASK]}}})
    save_json(root / "MANIFEST.json", {"sources": [SOURCE], "synthetic": True, "sample_rate_hz": FS,
              "channels": channels, "signal_rng_seed": FIXTURE_SEED, "split": "val"})
    save_json(root / "FROZEN.json", {"fixture": "synthetic photo montage, v1", "created_before_inference": True})


def run(checkpoint, mpnet, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    make_fixture(fixture)
    data = BenchmarkV13(fixture)
    data.require_frozen()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    model = LinkedEEGJev(checkpoint, mpnet)
    print("Loaded checkpoint and frozen text encoder on CPU", flush=True)
    answers = []
    scores = {}
    for track, expected_count in (("T1", 4), ("T2", 2)):
        rows = list(data.new_questions(track, "val", SOURCE))
        ids = {r["question_id"] for r in rows}
        if len(rows) != expected_count or len(ids) != expected_count:
            raise ValueError(f"Incomplete synthetic {track} group")
        predictions = []
        for row in rows:
            signals = data.new_eeg(row)
            choice, logits = model.answer(track, row["source"], signals, row["question"], row["options"])
            predictions.append({"question_id": row["question_id"], "prediction_index": choice})
            answers.append({"track": track, "id": row["question_id"], "question": row["question"],
                            "options": row["options"], "logits": logits,
                            "predicted": row["options"][choice],
                            "expected": row["options"][row["answer_index"]],
                            "correct": choice == row["answer_index"]})
        prediction_path = output / f"{track}_predictions.jsonl"
        score_path = output / f"{track}_score.json"
        save_rows(prediction_path, predictions)
        subprocess.run([sys.executable, "-I", "-B", str(PACKAGE / "bench/tools/benchmark_v13.py"),
                        "score-new", "--root", str(fixture), "--split", "val", "--track", track,
                        "--predictions", str(prediction_path), "--out", str(score_path)], check=True)
        report = json.loads(score_path.read_text())
        correct = sum(a["correct"] for a in answers if a["track"] == track)
        # For this particular one-source/one-person/one-group fixture the mean is count / views.
        if report["distinct_event_or_pair_groups"] != 1 or report["source_balanced_mean"] != correct / len(rows):
            raise ValueError(f"Bench score disagrees with the saved choices for {track}")
        scores[track] = {"questions": len(rows), "correct": correct,
                         "source_balanced_mean": report["source_balanced_mean"]}
    save_rows(output / "answers.jsonl", answers)
    with (output / "answers.tsv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["track", "id", "question", "predicted", "expected", "correct"]
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(answers)
    receipt = {"scope": "synthetic validation fixture only", "scores": scores, "model": model.receipt,
               "checkpoint": str(checkpoint.resolve()), "frozen_mpnet": str(mpnet.resolve()),
               "package_root": str(PACKAGE), "working_directory": str(Path.cwd()),
               "sample_rate_hz": FS, "torch": str(torch.__version__),
               "numpy": np.__version__, "threads": torch.get_num_threads(), "training": False}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({"scores": scores, "seed": model.receipt["seed"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mpnet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory for the fixture and results")
    args = parser.parse_args()
    run(args.checkpoint, args.mpnet, args.output)
