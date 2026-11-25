"""Package v1.3 validation scores and build the community leaderboard."""

import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit


COMMUNITY = Path(__file__).resolve().parents[1] / "community"
METRICS = {
    "D1": "Balanced accuracy",
    "D2": "Balanced joint score",
    "D3": "Balanced accuracy",
    "D4": "Matched refusal score",
    "T1": "Linked-view accuracy",
    "T2": "Linked-view accuracy",
}


def read_predictions(path, track):
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            question_id = row["question_id"]
            prediction = row["prediction_index"]
            if question_id in predictions:
                raise ValueError(f"Duplicate prediction ID {question_id}")
            if type(prediction) is not int and not (track == "D4" and prediction is None):
                raise ValueError(f"{question_id} requires a candidate index or a D4 null refusal")
            predictions[question_id] = prediction
    return predictions


def count_questions(rows, question_ids):
    for row in rows:
        question_id = row["question_id"]
        if question_id in question_ids:
            raise ValueError(f"Duplicate question ID {question_id}")
        question_ids.add(question_id)
        yield row


def score_validation(root, track, source, predictions):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eeg_jev.benchmark.v13_core_score import core_task_id, score_core
    from eeg_jev.benchmark.v13_dataset import BenchmarkV13, score_linked

    benchmark = BenchmarkV13(root)
    benchmark.require_frozen()
    if source and source not in benchmark.manifest["sources"]:
        raise ValueError(f"Unknown source {source}")
    question_ids = set()

    if track in ("T1", "T2"):
        rows = benchmark.new_questions(track, "val", source)
        report = score_linked(count_questions(rows, question_ids), predictions)
        score, tasks = report["source_balanced_mean"], sorted(report["by_task"])
    else:
        eligible = set(benchmark.tasks[track]["cells"])
        rows = benchmark.core_questions("val", source, track)
        rows = (row for row in rows if core_task_id(row) in eligible)
        report = score_core(count_questions(rows, question_ids), predictions, eligible, track)
        if track == "D2":
            joint = report["joint"]
            score = joint["aggregate"]["source_balanced_mean"]
            tasks = sorted(task for task, cell in joint["cells"].items() if cell.get("passes_minimum10"))
        else:
            metric = "group_balanced_refusal" if track == "D4" else "balanced_accuracy"
            score, tasks = report[metric]["source_balanced_mean"], sorted(report[metric]["by_task"])

    if predictions.keys() != question_ids:
        raise ValueError("Predictions must cover exactly the selected validation questions")
    if score is None or not tasks:
        raise ValueError(f"The v1.3 scorer produced no eligible {track} score")
    return len(question_ids), tasks, score


def validate(entry):
    fields = {"version", "split", "track", "model", "model_url", "evaluation_url", "questions", "tasks", "score"}
    if not isinstance(entry, dict) or set(entry) != fields:
        raise ValueError("Use the aggregate-only community format written by pack")
    if entry["version"] != "v1.3" or entry["split"] != "val" or entry["track"] not in METRICS:
        raise ValueError("Community results require a v1.3 validation track")
    model = entry["model"]
    if not isinstance(model, str) or not model.strip() or model != model.strip() or len(model.splitlines()) != 1:
        raise ValueError("Model name must be one non-empty line")
    for field in ("model_url", "evaluation_url"):
        url = entry[field]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or any(c.isspace() or c in '<>|' for c in url):
            raise ValueError(f"{field} must be an HTTPS URL")
    questions = entry["questions"]
    tasks = entry["tasks"]
    score = entry["score"]
    if type(questions) is not int or questions < 1:
        raise ValueError("A result must cover at least one validation question")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("Supply the non-empty list of scored task IDs")
    if tasks != sorted(set(tasks)):
        raise ValueError("Task IDs must be unique and sorted")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The aggregate score must be between 0 and 1")


def pack(args):
    predictions = read_predictions(args.predictions, args.track)
    questions, tasks, score = score_validation(args.root, args.track, args.source, predictions)
    entry = dict(version="v1.3", split="val", track=args.track, model=args.model,
                 model_url=args.model_url, evaluation_url=args.evaluation_url,
                 questions=questions, tasks=tasks, score=score)
    validate(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_text(value):
    return html.escape(value).replace("|", "&#124;").replace("[", "&#91;").replace("]", "&#93;")


def build(args):
    groups = defaultdict(list)
    for path in sorted(args.submissions.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: {error}") from error
        key = entry["track"], entry["evaluation_url"], entry["questions"], tuple(entry["tasks"])
        groups[key].append(entry)

    lines = [
        "# EEGJev-Bench community validation leaderboard",
        "",
        "Scores are comparable within each table. Each table uses one public evaluation recipe",
        "and the same question count and scored tasks. [Submit a result](../../CONTRIBUTING.md#submit-a-validation-result).",
        "",
    ]
    if not groups:
        lines.append("No reviewed community results yet.")
    for (track, evaluation, questions, tasks), entries in sorted(groups.items()):
        lines += [
            f"## {track} · {METRICS[track]}",
            "",
            f"[Evaluation recipe](<{evaluation}>) · {questions:,} validation questions · {len(tasks)} scored tasks",
            "",
            "| Model | Score (%) |",
            "| :--- | ---: |",
        ]
        for entry in sorted(entries, key=lambda row: (-row["score"], row["model"].casefold(), row["model_url"])):
            label = markdown_text(entry["model"])
            lines.append(f"| [{label}](<{entry['model_url']}>) | {entry['score'] * 100:.2f} |")
        lines += ["", "Scored tasks " + ", ".join(map(markdown_text, tasks)), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("pack", help="Score one v1.3 validation track and save the aggregate")
    package.add_argument("--root", type=Path, required=True)
    package.add_argument("--track", choices=METRICS, required=True)
    package.add_argument("--source")
    package.add_argument("--predictions", type=Path, required=True)
    package.add_argument("--model", required=True)
    package.add_argument("--model-url", required=True)
    package.add_argument("--evaluation-url", required=True)
    package.add_argument("--out", type=Path, required=True)
    package.set_defaults(run=pack)
    leaderboard = commands.add_parser("build", help="Render the checked-in submissions")
    leaderboard.add_argument("--submissions", type=Path, default=COMMUNITY / "submissions")
    leaderboard.add_argument("--out", type=Path, default=COMMUNITY / "LEADERBOARD.md")
    leaderboard.set_defaults(run=build)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
