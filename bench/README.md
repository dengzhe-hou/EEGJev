# EEGJev-Bench

EEGJev-Bench v1.3 includes a corpus reader, native-recording rebuild, and scoring commands for all six question types. See the [rebuild and scoring guide](BUILDING.md) for the frozen metadata contract, raw recording recipes, and the synthetic CPU check. Start with the [one-page reproduction guide](REPRODUCTION.md) and [source catalog](sources_v13.json).

Run the synthetic scoring example from the repository root with NumPy installed.

```sh
python -I -B bench/examples/score_linked_demo.py
```

The example gives the scorer predictions selecting one option for each generated question. It reports 0.50 for the window comparison and 1.00 for the event location. The examples use an invented person, synthetic questions and hand-written predictions.

Use [the reader](eeg_jev/benchmark/v13_dataset.py) with a built EEGJev-Bench v1.3 corpus. The [scoring CLI](tools/benchmark_v13.py) accepts the corpus root, question type, split and a JSONL file of predictions. Each prediction supplies a question ID and a zero-based option index. The [dataset card](https://huggingface.co/datasets/dengzhe-hou/EEGJev-Bench) describes the source data and access.
