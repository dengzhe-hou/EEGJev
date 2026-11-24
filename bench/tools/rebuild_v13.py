"""Materialize EEGJev-Bench v1.3 from native files and frozen recipes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_jev.benchmark.v13_signal import FS, check_window, cut, flat_runs, preprocess, read_native


def jsonl(path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def start_seconds(row, record, fs, long_window):
    if not long_window and "abs_start_s" in row:
        # Legacy positions in the window index are rounded for reporting. Use the
        # original segment's native start sample before converting to 200 Hz.
        return record["segment_start_samples"][row["segment"]] / fs + row["start_s"]
    return row["start_s"]


def build_windows(output, roots, records, source, part, short, long):
    """Write one source's cache arrays with rows in input-index order."""
    jobs = defaultdict(list)
    counts = Counter()
    channels = None
    for long_window, rows in ((False, short), (True, long)):
        lengths = {8, 16, 30} if long_window else {1, 2, 4}
        for row in rows:
            record = records[source, row["recording"]]
            if channels is None:
                channels = record["channels"]
            if channels != record["channels"] or row["length_s"] not in lengths:
                raise ValueError(f"{source}: channel order or supported duration differs")
            key = ("T2" if long_window else "core", int(row["length_s"]))
            jobs[row["recording"]].append((row, key, counts[key]))
            counts[key] += 1
    if not jobs:
        return
    target = output / "cache" / "rebuilt" / part / source
    target.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.lib.format.open_memmap(
            target / f"{key[0]}_{key[1]}s.npy",
            mode="w+", dtype=np.float32, shape=(count, len(channels), key[1] * FS),
        )
        for key, count in counts.items()
    }
    for recording_number, (recording, entries) in enumerate(jobs.items(), 1):
        record = records[source, recording]
        x, fs, boundaries = read_native(roots[source], record)
        runs = flat_runs(x, fs)
        for row, key, _ in entries:
            is_long = key[0] == "T2"
            start = start_seconds(row, record, fs, is_long)
            try:
                check_window(x, fs, runs, boundaries, start, key[1],
                             2.0 if is_long else record.get("drop_s", 2.0), is_long)
            except ValueError as error:
                name = row.get("window_id", row.get("question_id"))
                raise ValueError(f"{source}/{name}: frozen window failed validation: {error}") from error
        x200 = preprocess(x, fs, channels, record["reference_channels"], record["notch_hz"])
        del x
        for row, key, index in entries:
            array = arrays[key]
            ref = row.get("cache_ref", {})
            if "shape" in ref and tuple(ref["shape"]) != array.shape[1:]:
                raise ValueError(f"{source}: rebuilt EEG shape differs from its frozen index")
            array[index] = cut(x200, start_seconds(row, record, fs, key[0] == "T2"), key[1])
            row["cache_ref"] = {
                "file": str(Path(array.filename).relative_to(output)),
                "row": index,
                "format": "npy",
                "shape": list(array.shape[1:]),
            }
        del x200
        if recording_number % 25 == 0:
            for array in arrays.values():
                array.flush()
    for array in arrays.values():
        array.flush()


def rebuild(metadata_root, raw_config, output, partition):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    manifest = json.loads((metadata_root / "MANIFEST.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    if len(set(sources)) != len(sources) or any(not re.fullmatch(r"[A-Za-z0-9_]+", source) for source in sources):
        raise ValueError("Manifest source names must be unique letters, digits or underscore")
    if not (metadata_root / "FROZEN.json").is_file():
        raise FileNotFoundError("Expected the v1.3 frozen metadata bundle with a FROZEN.json")
    config = json.loads(raw_config.read_text(encoding="utf-8"))
    roots = {name: raw_config.parent / path for name, path in config["roots"].items()}
    records = {}
    for record in jsonl(raw_config.parent / config["recordings"]):
        key = record["source"], record["recording"]
        if key in records:
            raise ValueError(f"Duplicate recording recipe {key}")
        records[key] = record
    if not records:
        raise ValueError("The recording recipe file is missing or empty")

    parts = ("train", "test") if partition == "all" else (partition,)
    splits = {"train", "val"} if parts == ("train",) else set(parts)
    if "train" in splits:
        splits.add("val")
    totals = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rebuild-v13-", dir=output.parent) as directory:
        root = Path(directory) / "corpus"
        (root / "metadata").mkdir(parents=True)
        shutil.copyfile(metadata_root / "MANIFEST.json", root / "MANIFEST.json")
        shutil.copyfile(metadata_root / "metadata/tasks_v13.json", root / "metadata/tasks_v13.json")
        for name in ("scoring_exclusions.json", "exclusions_v13.json", "phrasings_v13.json"):
            origin = metadata_root / "metadata" / name
            if origin.is_file():
                shutil.copyfile(origin, root / "metadata" / name)
        built = set()
        for source in sources:
            t2_relative = Path("new_tracks/validated") / f"T2_{source}.jsonl"
            t2 = [row for row in jsonl(metadata_root / t2_relative) if row["split"] in splits]
            for part in parts:
                origin = metadata_root / "frozen_v13" / part
                target = root / "frozen_v13" / part
                filename = f"{source}_windows.jsonl"
                short = list(jsonl(origin / filename))
                expected = {"test"} if part == "test" else {"train", "val"}
                for row in short:
                    key = source, row["split"], row["window_id"]
                    if row["source"] != source or row["split"] not in expected or key in built:
                        raise ValueError(f"Invalid or duplicate short-window row {key}")
                    built.add(key)
                long = [row for row in t2 if row["split"] in expected]
                build_windows(root, roots, records, source, part, short, long)
                if (origin / filename).is_file():
                    write_rows(target / filename, short)
                for questions in origin.glob(f"{source}_questions*.jsonl"):
                    target.mkdir(parents=True, exist_ok=True)
                    with questions.open(encoding="utf-8") as source_file, (target / questions.name).open("w", encoding="utf-8") as destination:
                        for line in source_file:
                            question = json.loads(line)
                            key = question["source"], question["split"], question["window_id"]
                            if question["source"] != source or question["split"] not in expected or key not in built:
                                raise ValueError(f"Question references a missing rebuilt window {key}")
                            destination.write(line)
                totals.update(core_windows=len(short), T2_presentations=len(long))
            if t2:
                events = defaultdict(list)
                for row in t2:
                    if row["source"] != source:
                        raise ValueError(f"{t2_relative}: mismatched T2 source")
                    key = row["split"], row["task"], row["subject"], row["group_id"]
                    events[key].append(row["answer_index"])
                if any(sorted(answers) != [0, 1] for answers in events.values()):
                    raise ValueError(f"{t2_relative}: each event requires two complementary presentations")
                write_rows(root / t2_relative, t2)

        for file in sorted((metadata_root / "new_tracks/data").glob("t1_*.jsonl")):
            rows = []
            for row in jsonl(file):
                if row["split"] not in splits:
                    continue
                if any((row["source"], row["split"], window) not in built for window in row["windows"]):
                    raise ValueError(f"T1 pair {row['pair_id']} refers to a missing rebuilt window")
                rows.append(row)
            if rows:
                write_rows(root / "new_tracks/data" / file.name, rows)
                totals["T1_pairs"] += len(rows)

        if not totals["core_windows"] and not totals["T2_presentations"]:
            raise ValueError("No windows were selected for the requested partition")
        (root / "REBUILD.json").write_text(
            json.dumps({"partition": partition, "sampling_hz": FS, "counts": dict(totals)}, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(metadata_root / "FROZEN.json", root / "FROZEN.json")
        root.rename(output)
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--raw-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "test", "all"), default="train",
                        help="train rebuilds both train and validation windows from their shared index")
    args = parser.parse_args()
    counts = rebuild(args.metadata_root.resolve(), args.raw_config.resolve(), args.out.resolve(), args.partition)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
