"""Use frozen EEGJev weights to demonstrate Choice, Noul, Score and Act on synthetic EEG."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "model"))

from eegjev.inference import FS, LinkedEEGJev
from eegjev.typed import Choice, Noul, Score, answer_typed


def synthetic_window(source, seconds=1):
    montages = json.loads((PACKAGE / "model/eegjev/montages.json").read_text())
    phases = np.linspace(0.0, 1.0, len(montages[source]))[:, None]
    time = np.arange(seconds * FS) / FS
    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * 9 * time + phases)
    signal += 0.25 * rng.standard_normal(signal.shape)
    signal -= signal.mean(axis=1, keepdims=True)
    return (signal / signal.std(axis=1, keepdims=True)).astype(np.float32)


def demo_questions():
    question = "Were the participant's eyes open or closed?"
    eyes = ("eyes open", "eyes closed")
    return {
        "eye_state": Choice(question, eyes, with_answerability=True),
        "eyes_closed": Noul(question, eyes, true_option="eyes closed"),
        "nback_load": Score(
            "Which n-back level was the participant doing?",
            ("0-back", "1-back", "2-back"),
            values=(0, 1, 2),
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--refusal", type=Path, required=True)
    parser.add_argument("--mpnet", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    decision = LinkedEEGJev(args.decision, args.mpnet)
    decision.load_refusal(args.refusal)
    d3 = LinkedEEGJev(args.d3, args.mpnet)
    source = "cogbci"
    answers = answer_typed(decision, source, [synthetic_window(source)], demo_questions(), ordered_model=d3)
    print(json.dumps(answers, indent=2, allow_nan=False))
