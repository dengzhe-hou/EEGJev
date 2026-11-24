# Rebuild EEGJev-Bench v1.3

This guide rebuilds the signal cache from native recordings and the frozen question and window metadata of `1.3-final-20261115`. The available [dataset card](https://huggingface.co/datasets/dengzhe-hou/EEGJev-Bench) and [data-quality notes](../paper/v1.3/data-quality-v13.md) describe the upstream recordings and corrections.

## Inputs

1. Obtain the original recordings from their providers at the versions recorded in your benchmark manifest. Preserve the companions of each BrainVision header and EEGLAB file.
2. Use the corresponding frozen metadata bundle. It needs `MANIFEST.json`, `FROZEN.json`, `metadata/tasks_v13.json`, the `frozen_v13/{train,test}` window and question JSONL files, `new_tracks/data/t1_*.jsonl`, and `new_tracks/validated/T2_*.jsonl` for the desired partition. The original `cache_ref` entries may point to absent caches during a rebuild.
3. Map each recording ID in that bundle to its native file using a recording recipe. The public source code and synthetic example are available here. Reproducing the paper's exact corpus additionally requires the matching frozen participant assignments, windows, questions, labels, and native recording recipes from the benchmark maintainers.

The original build retained the accepted short-window recordings and labels for the legacy sources, indexed the additional sources from their native files, applied the frozen person-level split, and generated the linked T1 pairs and validated T2 presentations. Preserve the metadata and its order to keep question IDs, splits, task eligibility, and linked view IDs consistent with saved predictions.

## Recording recipes

Save a raw configuration with one root per source and the path to a JSONL file of recording recipes. Paths in this configuration are relative to the configuration file or absolute.

```json
{"roots":{"demo":"native"},"recordings":"recordings.jsonl"}
```

Every line in `recordings.jsonl` identifies a source and recording ID, and records the reader, file relative to the source root, original sampling rate, and ordered model channels. The reference channels and notch frequencies are the settings fixed for the recording.

```json
{"source":"demo","recording":"demo-recording","path":"demo.vhdr","reader":"brainvision","fs":250,"channels":["C3","C4"],"reference_channels":[],"notch_hz":[]}
```

The readers are `brainvision`, `eeglab`, `edf`, `gal_ws`, and `bciciv_mat`. A GAL recipe supplies the original one-based `lift`. Use `channel_map` to map an output channel to an input label or a pair of labels whose difference gives a bipolar channel. The MNE readers convert volts to µV by default; a recipe can supply its recorded `stored_unit_scale`. Native MAT readings use the stored values. Supply `boundaries_s` when the original recording index specifies the discontinuities. Otherwise the readers use native boundary and new-segment annotations when present.

For legacy short windows with `abs_start_s` in the metadata, also supply `segment_start_samples`. This maps the recorded segment IDs to the exact native start samples. The builder adds each window's segment-relative offset before rounding to sample indices. For other short windows and T2, `start_s` is recording-relative.

## Build and read

Install the CPU build requirements with Python 3.11 or later. Run from the repository root.

```sh
python -m pip install -r bench/requirements-build.txt
python -I -B bench/tools/rebuild_v13.py \
  --metadata-root /path/to/frozen-metadata \
  --raw-config /path/to/raw.json \
  --out /path/to/new-v13-corpus --partition train
python -I -B bench/tools/benchmark_v13.py inspect --root /path/to/new-v13-corpus
```

The `train` partition contains the train and validation splits. Use `test` or `all` with the corresponding authorized inputs. The rebuild applies the native-channel flat and dead-channel checks, recorded reference (including the scalp average for ERP CORE), notch filtering and whole-recording resampling to 200 Hz before windowing and per-channel standardization. Each cache contains float32 arrays. The output is created in a new directory after all requested windows pass validation. `REBUILD.json` records the rebuilt partition and row counts.

The rebuild holds one native recording and its resampled signal in memory at a time. Allow disk space for the output and memory for the largest recording in the selected partition. `BenchmarkV13` reads these arrays with `core_eeg` and `new_eeg`. It uses the same `core_questions` and `new_questions` interfaces as the frozen corpus.

## Score final decisions

Write predictions as JSONL with one row per question from the reader. Each `prediction_index` is the zero-based position in that question's offered options. For D4, use `null` to decline or an option index to answer. When reproducing a model's D4 result, apply that model's chosen rule to obtain these decisions before running the scorer.

```sh
python -I -B bench/tools/benchmark_v13.py score-core \
  --root /path/to/new-v13-corpus --split val --track D1 \
  --predictions predictions.jsonl --out d1-score.json
python -I -B bench/tools/benchmark_v13.py score-new \
  --root /path/to/new-v13-corpus --split val --track T2 \
  --predictions temporal-predictions.jsonl --out t2-score.json
```

`score-core` uses the frozen eligible task list for each track. It reports balanced accuracy for D1, D1h, D2 and D3, the original joint D2 score, accuracy and mean bin distance for D3, and the matched-option-group refusal score for D4. Scores average tasks within each source and then weight sources equally. `score-new` averages related views within each pair or event group before averaging over people, tasks and sources. `score-core` consumes final decisions; calibration and threshold fitting use the separately documented model protocol.

## Run the synthetic example

```sh
python -I -B bench/examples/rebuild_linked_demo.py
```

The example uses an invented BrainVision recording and the bundled T1/T2 questions. It checks the rebuilt samples through the public reader and scores fixed decisions with expected T1 and T2 means of 0.50 and 1.00.
