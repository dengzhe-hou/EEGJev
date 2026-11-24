# EEGJev inference

The CPU entry point restores the shared v1.3 checkpoint, including its trained CBraMod encoder and decision head. It reads standardized 200 Hz waveforms in the [montage order used during training](eegjev/montages.json). The loaded MPNet snapshot encodes the question and candidate text.

Use Python 3.13. Install the CPU PyTorch build before the remaining dependencies.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
python -m pip install -r requirements.txt
```

Run from the repository root with a locally available v1.3 shared checkpoint and an all-mpnet-base-v2 snapshot. Choose a new output directory for each example.

```sh
CHECKPOINT=/path/to/eegjev_seed0.pt
MPNET=/path/to/all-mpnet-base-v2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -I -B examples/question_switch.py --checkpoint "$CHECKPOINT" --mpnet "$MPNET" --output question-output
python -I -B examples/synthetic_linked.py --checkpoint "$CHECKPOINT" --mpnet "$MPNET" --output temporal-output
```

The first example asks two questions of the same synthetic window with the same six candidates. The second generates synthetic T1 and T2 questions, runs the model and scores its answers with EEGJev-Bench. The examples illustrate inputs and outputs. The [paper results](../paper/v1.3/README.md) report performance on recorded EEG. The [Hugging Face model card](https://huggingface.co/dengzhe-hou/EEGJev) describes the weight files. See the [upstream CBraMod license](eegjev/cbramod/LICENSE).
