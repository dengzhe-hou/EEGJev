"""Compute the v1.3 core metrics from final candidate and refusal decisions."""

from collections import Counter, defaultdict

import numpy as np

from .v13_joint_d2 import score_bundle


def core_task_id(row):
    return f"{row['source']}-{row['attribute']}-{row['length_s']}s"


def aggregate_cells(values, cell_sources, eligible):
    by_source = defaultdict(list)
    for cell, value in values.items():
        by_source[cell_sources[cell]].append(value)
    source_means = {source: float(np.mean(scores)) for source, scores in sorted(by_source.items())}
    return {
        "source_balanced_mean": float(np.mean(list(source_means.values()))) if source_means else None,
        "by_source": source_means,
        "by_task": values,
        "tasks_scored": len(values),
        "tasks_without_value": sorted(eligible - values.keys()),
    }


def score_core(rows, predictions, eligible, track):
    eligible = set(eligible)
    cells = defaultdict(list)
    cell_sources = {}
    offered = defaultdict(set)
    d2_rows, d2_answers = [], []
    for row in rows:
        source = row["source"]
        cell = core_task_id(row)
        if cell not in eligible:
            continue
        question_id = row["question_id"]
        if question_id not in predictions:
            raise ValueError(f"Missing prediction {question_id}")
        options = row["options"]
        offered[cell].update(options)
        choice = predictions[question_id]
        if choice is None:
            if track != "D4":
                raise ValueError(f"{question_id}: null is a D4 refusal")
        elif type(choice) is not int or not 0 <= choice < len(options):
            raise ValueError(f"{question_id}: prediction_index must be an offered option index")
        cell_sources[cell] = source
        if track == "D4":
            decision = row["gold_decision"]
            if decision not in ("answer", "refuse"):
                raise ValueError(f"{question_id}: invalid D4 gold decision")
            cells[cell].append((tuple(sorted(options)), decision == "refuse", choice is None))
            continue

        answer = options.index(row["answer"])
        if track == "D3" and (row.get("ordered") is not True or row["answer_index"] != answer):
            raise ValueError(f"{question_id}: invalid ordered-bin answer")
        cells[cell].append((options[answer], options[choice], abs(answer - choice)))
        if track == "D2":
            d2_rows.append(dict(row, answer_index=answer))
            d2_answers.append(choice)

    if not cells:
        raise ValueError(f"No eligible {track} questions in the requested split")

    if track == "D4":
        criteria = {}
        unavailable = 0
        for cell, decisions in cells.items():
            groups = defaultdict(lambda: [[], []])
            for options, gold_refuse, predicted_refuse in decisions:
                groups[options][int(gold_refuse)].append(predicted_refuse)
            scores = []
            for controls, refusals in groups.values():
                if controls and refusals:
                    scores.append(0.5 * (float(np.mean(refusals)) + 1.0 - float(np.mean(controls))))
                else:
                    unavailable += 1
            if scores:
                criteria[cell] = float(np.mean(scores))
        return {
            "group_balanced_refusal": aggregate_cells(criteria, cell_sources, eligible),
            "option_groups_without_both_decisions": unavailable,
        }

    balanced, accuracy, distances = {}, {}, {}
    for cell, entries in cells.items():
        answers = Counter(ground_truth for ground_truth, _, _ in entries)
        correct = Counter(ground_truth for ground_truth, predicted, _ in entries if ground_truth == predicted)
        if min(answers.get(label, 0) for label in offered[cell]) >= 10:
            balanced[cell] = float(np.mean([correct[label] / count for label, count in answers.items()]))
        if track == "D3":
            accuracy[cell] = sum(correct.values()) / len(entries)
            distances[cell] = float(np.mean([distance for _, _, distance in entries]))
    out = {"balanced_accuracy": aggregate_cells(balanced, cell_sources, eligible)}
    if track == "D2":
        # The declared options determine a final argmax. The original joint
        # scorer combines the per-phrasing correct/incorrect decisions.
        width = max(len(row["options"]) for row in d2_rows)
        logits = np.zeros((len(d2_rows), width), dtype=np.float32)
        logits[np.arange(len(d2_rows)), d2_answers] = 1.0
        out["joint"] = score_bundle(d2_rows, logits, np.ones(len(d2_rows), dtype=bool), eligible)
    if track == "D3":
        out["accuracy"] = aggregate_cells(accuracy, cell_sources, eligible)
        out["mean_bin_distance"] = aggregate_cells(distances, cell_sources, eligible)
    return out
