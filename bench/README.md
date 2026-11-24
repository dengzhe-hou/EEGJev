# EEGJev-Bench

This folder includes a reader for the v1.3 benchmark and the linked-view scorer for T1 and T2. The scorer first averages the linked views of a pair or event group, then averages equally over participants within each task, tasks within each source, and sources.

Run the synthetic scoring example from the repository root with NumPy installed.

```sh
python -I -B bench/examples/score_linked_demo.py
```

The example gives the scorer predictions selecting one option for each generated question. It reports 0.50 for the window comparison and 1.00 for the event location. The examples use an invented person, synthetic questions and hand-written predictions.

Use [the reader](eeg_jev/benchmark/v13_dataset.py) with a built EEGJev-Bench v1.3 corpus. The [scoring CLI](tools/benchmark_v13.py) accepts the corpus root, question type, split and a JSONL file of predictions. Each prediction supplies a question ID and a zero-based option index. The [dataset card](https://huggingface.co/datasets/dengzhe-hou/EEGJev-Bench) describes the source data and access.
