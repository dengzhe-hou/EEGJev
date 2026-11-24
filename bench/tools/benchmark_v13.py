"""Inspect and score the frozen EEGJev-Bench v1.3 corpus."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_jev.benchmark.v13_dataset import BenchmarkV13, score_linked
from eeg_jev.benchmark.v13_core_score import score_core


CORE = ("D1", "D1h", "D2", "D3", "D4")
LINKED = ("T1", "T2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "score-new", "score-core"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--track", choices=CORE + LINKED)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--source")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--test-model-scoring-authorized", action="store_true")
    args = parser.parse_args()
    if args.command != "inspect" and args.split == "test" and not args.test_model_scoring_authorized:
        parser.error("Model scoring of test data requires explicit authorization")

    data = BenchmarkV13(args.root)
    if args.command == "inspect":
        print(json.dumps({"frozen": data.frozen, "tracks": data.tasks,
                          "test_scope": data.manifest.get("test_authorization")}, ensure_ascii=False))
        return

    data.require_frozen()
    tracks = LINKED if args.command == "score-new" else CORE
    if args.track not in tracks or not args.predictions or not args.out:
        parser.error(f"{args.command} requires --track ({', '.join(tracks)}), --predictions and --out")
    predictions = {}
    with args.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            question_id = row["question_id"]
            if question_id in predictions:
                raise ValueError(f"Duplicate prediction {question_id}")
            predictions[question_id] = row["prediction_index"]

    if args.command == "score-new":
        report = score_linked(data.new_questions(args.track, args.split, args.source), predictions)
    else:
        base_track = "D1" if args.track == "D1h" else args.track
        eligible = data.tasks[base_track]["cells"]
        if args.source:
            eligible = [cell for cell in eligible if cell.split("-", 1)[0] == args.source]
        report = score_core(data.core_questions(args.split, args.source, args.track),
                            predictions, eligible, args.track)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
