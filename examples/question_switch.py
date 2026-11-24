"""Ask two questions of one synthetic ASYM-format EEG window with a fixed six-label list."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "model"))
from eegjev.inference import FS, LinkedEEGJev
from eegjev.short_window import answer_short_window

SOURCE = "asym"
SEED = 42
OPTIONS = ("fearful", "right hand", "happy (mouth open)", "angry", "left hand", "happy (mouth closed)")
QUESTIONS = (
    ("emotion", "Which expression was on the face shown?"),
    ("response_hand", "Which hand was the participant instructed to respond with in this block?"),
)
# Membership is used to describe the response after all six candidates have been scored.
LABEL_GROUP = {label: ("response_hand" if label in ("left hand", "right hand") else "emotion") for label in OPTIONS}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_fixture(directory):
    directory.mkdir(parents=True)
    montages = json.loads((PACKAGE / "model/eegjev/montages.json").read_text())
    channels = montages[SOURCE]
    t = np.arange(FS) / FS
    phase = np.linspace(0, 1.1, len(channels))[:, None]
    rng = np.random.default_rng(SEED)
    signal = np.sin(2 * np.pi * 9 * t + phase) + 0.25 * rng.standard_normal((len(channels), FS))
    signal = ((signal - signal.mean(axis=1, keepdims=True)) / signal.std(axis=1, keepdims=True)).astype("float32")
    np.save(directory / "window.npy", signal)
    save_json(directory / "questions.json", {
        "source": SOURCE, "sample_rate_hz": FS, "channel_order": channels, "rng_seed": SEED,
        "window_file": "window.npy", "synthetic": True, "options": OPTIONS,
        "questions": [{"attribute": field, "text": text} for field, text in QUESTIONS],
        "specific_answer_ground_truth": None,
    })


def write_answers(path, answers):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=("attribute", "question", "chosen_label", "chosen_attribute",
                                                 "matches_requested_attribute"), delimiter="\t", lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(answers)


def run(checkpoint, encoder, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    make_fixture(fixture)
    inputs = json.loads((fixture / "questions.json").read_text())
    signal = np.load(fixture / inputs["window_file"], allow_pickle=False)
    options = inputs["options"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")

    model = LinkedEEGJev(checkpoint, encoder)
    answers = []
    print("Offered choices in both questions", " | ".join(options), sep="\n", flush=True)
    for q in inputs["questions"]:
        index, logits = answer_short_window(model, inputs["source"], signal, q["text"], options)
        label = options[index]
        group = LABEL_GROUP[label]
        answers.append({"attribute": q["attribute"], "question": q["text"], "chosen_label": label,
                        "chosen_attribute": group, "matches_requested_attribute": group == q["attribute"],
                        "chosen_index": index, "logits_in_option_order": logits})
        print(f'{q["text"]}  ->  {label}', flush=True)
    expression, hand = [row["chosen_label"] for row in answers]
    comparison = (f"| Facial expression | Instructed response hand |\n"
                  f"| --- | --- |\n| {expression} | {hand} |\n")
    (output / "comparison.md").write_text(comparison, encoding="utf-8")
    write_answers(output / "answers.tsv", answers)
    save_json(output / "answers.json", {"window_file": "fixture/window.npy", "options": options, "answers": answers})
    save_json(output / "receipt.json", {
        "kind": "synthetic question switch", "model": model.receipt,
        "sample_rate_hz": FS, "window_shape": list(signal.shape), "question_count": len(answers),
        "specific_answer_ground_truth": None, "threads": torch.get_num_threads(),
        "checkpoint": str(checkpoint.resolve()), "text_encoder": str(encoder.resolve()),
        "python": sys.version.split()[0], "torch": str(torch.__version__),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mpnet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    run(args.checkpoint, args.mpnet, args.output)
