"""The benchmark v1.3 joint D2 metric over attributes, phrasings, tasks and sources."""
from collections import defaultdict

import numpy as np


def window_scores(rows, logits, scored, phrasing="all"):
    """{(source, a1, a2, L): {window_id: (score, (answer1, answer2))}} over the D2 rows."""
    per = defaultdict(lambda: defaultdict(dict))         # window -> attribute -> phrasing -> correct
    meta, unscored = {}, set()
    for i, r in enumerate(rows):
        if r["track"] != "D2":
            continue
        a1, a2 = r["pair"]
        key = (r["source"], a1, a2, r["length_s"])
        w = (key, r["window_id"])
        if not scored[i]:
            unscored.add(w)
            continue
        k = len(r["options"])
        z = logits[i, :k]
        correct = float(int(np.argmax(np.where(np.isfinite(z), z, -np.inf))) == r["answer_index"])
        per[w][r["attribute"]][r.get("phrasing", 0)] = correct
        meta.setdefault(w, {})[r["attribute"]] = r["answer"]
    out = defaultdict(dict)
    for w, attrs in per.items():
        key, window = w
        a1, a2 = key[1], key[2]
        if w in unscored or set(attrs) != {a1, a2}:
            continue
        if phrasing == "first":
            if 0 not in attrs[a1] or 0 not in attrs[a2]:
                continue
            score = attrs[a1][0] * attrs[a2][0]
        else:
            score = float(np.mean(list(attrs[a1].values())) * np.mean(list(attrs[a2].values())))
        out[key][window] = (score, (meta[w][a1], meta[w][a2]))
    return out


def pair_cells(windows, attribute_cells=None, minimum=10):
    """Balanced joint score per pair cell; attribute_cells = allowed per-attribute D2 cells ('source-attr-Ls')."""
    cells = {}
    for (source, a1, a2, L), wins in sorted(windows.items()):
        name = f"{source}-{a1}+{a2}-{L}s"
        entry = {"source": source, "windows": len(wins)}
        if attribute_cells is not None and not {f"{source}-{a1}-{L}s", f"{source}-{a2}-{L}s"} <= set(attribute_cells):
            entry["excluded"] = True
            cells[name] = entry
            continue
        combos = defaultdict(list)
        for score, combo in wins.values():
            combos[combo].append(score)
        entry["combinations"] = {f"{c[0]} | {c[1]}": len(v) for c, v in sorted(combos.items())}
        entry["minimum_combination_windows"] = min(len(v) for v in combos.values())
        entry["passes_minimum10"] = entry["minimum_combination_windows"] >= minimum
        entry["balanced_joint"] = float(np.mean([np.mean(v) for v in combos.values()]))
        entry["raw_joint"] = float(np.mean([s for s, _ in wins.values()]))
        cells[name] = entry
    return cells


def aggregate(cells):
    passing = {n: c for n, c in cells.items() if c.get("passes_minimum10")}
    by_source = defaultdict(list)
    for n, c in passing.items():
        by_source[c["source"]].append(c["balanced_joint"])
    sb = float(np.mean([np.mean(v) for v in by_source.values()])) if by_source else None
    return {"source_balanced_mean": sb,
            "pair_cell_mean": float(np.mean([c["balanced_joint"] for c in passing.values()])) if passing else None,
            "pair_cells_present": len(cells), "excluded": sum(1 for c in cells.values() if c.get("excluded")),
            "failing_minimum10": sum(1 for c in cells.values() if "passes_minimum10" in c and not c["passes_minimum10"]),
            "passing": len(passing), "sources": len(by_source)}


def score_bundle(rows, logits, scored, attribute_cells=None, phrasing="all"):
    cells = pair_cells(window_scores(rows, logits, scored, phrasing), attribute_cells)
    return {"phrasing": phrasing, "aggregate": aggregate(cells), "cells": cells}
