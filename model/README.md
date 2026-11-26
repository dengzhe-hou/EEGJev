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

## Choice, Noul, Score and answerability

Use the matching decision, D3 and refusal weights for one variant and seed. The example prints a JSON object with all three output types on a simulated COG-BCI window.

~~~sh
WEIGHTS=/path/to/weights/shared/seed0
python -I -B examples/typed_outputs.py \
  --decision "$WEIGHTS/decision.pt" --d3 "$WEIGHTS/d3.pt" \
  --refusal "$WEIGHTS/refusal.pt" --mpnet "$MPNET"
~~~

With the repository's `model/` directory on Python's import path, pass a mapping from names to requests. Each request uses the same EEG input and makes its own forward pass.

~~~python
from eegjev.inference import LinkedEEGJev
from eegjev.typed import Choice, Noul, Score, answer_typed

decision = LinkedEEGJev(weights / "decision.pt", mpnet)
decision.load_refusal(weights / "refusal.pt")
d3 = LinkedEEGJev(weights / "d3.pt", mpnet)
question = "Were the participant's eyes open or closed?"
eyes = ("eyes open", "eyes closed")
answers = answer_typed(decision, "cogbci", [signal], {
    "eye_state": Choice(question, eyes, with_answerability=True),
    "eyes_closed": Noul(question, eyes, true_option="eyes closed"),
    "load": Score("Which n-back level was the participant doing?",
                  ("0-back", "1-back", "2-back"), values=(0, 1, 2)),
}, ordered_model=d3)
~~~

- **Choice** returns the selected label and a probability for each offered label. With `with_answerability=True` it also returns the learned Act head's `answerability_probability`. The selected label is `None` when Act refuses.
- **Noul** returns `probability_true` for the label designated by `true_option` in a supported two-label question. The probability is conditional on that pair.
- **Score** returns the D3 bin probabilities and their weighted mean on the supplied scale. Give one strictly increasing value per bin. For Score alone, you can pass the D3 checkpoint as the first model.

The probabilities are softmax outputs before calibration. Use the source's trained labels and montage. Choice and Noul also accept `track="T1"` (two short windows, options `window A` and `window B`) and `track="T2"` (one long window, options `first half` and `second half`). Act and Score use short windows.
