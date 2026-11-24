"""Run the v1.3 T1 and T2 scorers on invented validation questions and predictions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


SCORER = Path(__file__).resolve().parents[1] / "tools" / "benchmark_v13.py"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def prepare_inputs(root):
    (root / "metadata").mkdir()
    (root / "MANIFEST.json").write_text(json.dumps({"sources": ["demo"]}))
    (root / "FROZEN.json").write_text(json.dumps({"description": "Synthetic validation fixture only"}))
    tracks = {"T1": {"cells": ["demo-pair"]}, "T2": {"cells": ["demo-event"]}}
    (root / "metadata" / "tasks_v13.json").write_text(json.dumps({"tracks": tracks}))

    common = {"split": "val", "source": "demo", "subject": "example-person"}
    pair = dict(common, task="demo-pair", pair_id="pair1", attribute="state",
                windows=["w1", "w2"], labels=["rest", "move"])
    write_jsonl(root / "new_tracks" / "data" / "t1_demo.jsonl", [pair])

    event = dict(common, task="demo-event", group_id="event1",
                 question="Which half contains the cue?", options=["first half", "second half"])
    presentations = [dict(event, question_id="cue-a", answer_index=0),
                     dict(event, question_id="cue-b", answer_index=1)]
    write_jsonl(root / "new_tracks" / "validated" / "T2_demo.jsonl", presentations)


def score(root, track, predictions, expected):
    path = root / f"{track}_predictions.jsonl"
    write_jsonl(path, [dict(question_id=q, prediction_index=i) for q, i in predictions.items()])
    out = root / f"{track}_score.json"
    subprocess.run([sys.executable, "-I", "-B", str(SCORER), "score-new", "--root", str(root),
                    "--split", "val", "--track", track, "--predictions", str(path), "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    report = json.loads(out.read_text())
    observed = report["source_balanced_mean"]
    if observed != expected or report["distinct_event_or_pair_groups"] != 1:
        raise ValueError(f"{track}: expected {expected} in one group, got {report}")
    print(f"{track}: {observed:.2f} ({len(predictions)} invented validation questions)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare_inputs(root)
        # A fixed window-A response is right on two of the four linked views.
        choices = {f"pair1/swap{swap}/{target}": 0 for swap in (0, 1) for target in (0, 1)}
        score(root, "T1", choices, 0.5)
        score(root, "T2", {"cue-a": 0, "cue-b": 1}, 1.0)


if __name__ == "__main__":
    main()
