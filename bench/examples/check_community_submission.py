"""Check community packaging and ranking with invented validation questions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_linked_demo import prepare_inputs, write_jsonl


CLI = Path(__file__).resolve().parents[1] / "tools" / "community_v13.py"


def synthetic_core(root):
    tasks_path = root / "metadata" / "tasks_v13.json"
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    common = dict(split="val", source="demo", length_s=1, v13_task=True, attribute="state",
                  options=["rest", "move"])
    records = {
        "D1": [dict(common, answer=label) for label in common["options"] for _ in range(10)],
        "D2": [
            dict(common, attribute=attr, pair=["state", "hand"], window_id=f"demo-{i}-{j}-{n}",
                 options=opts, answer=opts[index])
            for i in range(2) for j in range(2) for n in range(10)
            for attr, opts, index in (("state", ["rest", "move"], i), ("hand", ["left", "right"], j))
        ],
        "D3": [dict(common, ordered=True, answer_index=i, answer=label)
               for i, label in enumerate(common["options"]) for _ in range(10)],
        "D4": [dict(common, gold_decision=decision) for decision in ("answer", "refuse")],
    }
    questions, predictions = [], {}
    for track, rows in records.items():
        predictions[track] = {}
        data["tracks"][track] = {"cells": sorted({f"demo-{row['attribute']}-1s" for row in rows})}
        for row in rows:
            predictions[track][f"{track}/demo/questions/{len(questions)}"] = (
                None if row.get("gold_decision") == "refuse" else row["options"].index(row.get("answer", "rest"))
            )
            questions.append(dict(row, track=track))
    tasks_path.write_text(json.dumps(data), encoding="utf-8")
    write_jsonl(root / "frozen_v13" / "train" / "demo_questions.jsonl", questions)
    return predictions


def run(*args, **kwargs):
    return subprocess.run([sys.executable, "-I", "-B", str(CLI), *map(str, args)], **kwargs)


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        prepare_inputs(root)
        predictions = synthetic_core(root)
        predictions["T1"] = {f"pair1/swap{swap}/{target}": 0 for swap in (0, 1) for target in (0, 1)}
        predictions["T2"] = {"cue-a": 0, "cue-b": 1}
        submissions = root / "submissions"
        for track, choices in predictions.items():
            inputs = root / f"{track}_decisions.jsonl"
            write_jsonl(inputs, [dict(question_id=q, prediction_index=i) for q, i in choices.items()])
            submission = submissions / f"{track.lower()}.json"
            run("pack", "--root", root, "--track", track, "--predictions", inputs, "--model", "Synthetic model",
                "--model-url", "https://example.org/model", "--evaluation-url", "https://example.org/fixture",
                "--out", submission, check=True)
            result = json.loads(submission.read_text(encoding="utf-8"))
            if result["score"] != (0.5 if track == "T1" else 1.0) or result["questions"] != len(choices):
                raise ValueError(f"Unexpected {track} score or coverage")
        board = root / "board.md"
        run("build", "--submissions", submissions, "--out", board, check=True)
        rendered = board.read_text(encoding="utf-8")
        if rendered.count("[Synthetic model]") != 6 or "50.00" not in rendered or "100.00" not in rendered:
            raise ValueError("The synthetic submissions did not render correctly")
        print("D1, D2, D3, D4, T1 and T2 packed and rendered from invented validation inputs.")


if __name__ == "__main__":
    main()
