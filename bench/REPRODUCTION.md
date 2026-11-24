# Reproduce EEGJev-Bench v1.3

1. Find the original recordings and their versions in the [source catalog](sources_v13.json). Get the recordings directly from their providers under the applicable access terms.
2. Use the [Hub release](https://huggingface.co/datasets/dengzhe-hou/EEGJev-Bench) for training and validation coverage and source-level preprocessing settings. Recreating the exact corpus also requires the frozen window and question indexes and the per-recording recipes for the chosen split.
3. Try the build and scorer from the repository root with the [CPU build dependencies](requirements-build.txt) installed.

```sh
python -I -B bench/examples/rebuild_linked_demo.py
```

The synthetic BrainVision example rebuilds the windows and scores fixed decisions. The T1 comparison scores 0.50 and the T2 event location scores 1.00.

For a corpus built from original recordings, follow the [native recording recipe and scoring instructions](BUILDING.md). Keep the frozen row order and supply each prediction with its question ID and zero-based option index.
