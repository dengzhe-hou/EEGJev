"""Score a short EEG window against all offered text labels with the trained joint head."""
import numpy as np
import torch

from . import decision
from .inference import FS
from .text import Question


@torch.inference_mode()
def answer_short_window(model, source, signal, prompt, options):
    """Return the chosen option index and logits in the original order on CPU."""
    labels = tuple(options)
    count = len(labels)
    if not 2 <= count <= 12:
        raise ValueError("The joint choice head expects 2 to 12 candidates")
    eeg = np.ascontiguousarray(signal, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] != len(model.montages[source]):
        raise ValueError("Expected an EEG window in the registered source montage order")
    patches, remainder = divmod(eeg.shape[1], FS)
    if remainder or patches not in model.lengths[source]:
        raise ValueError(f"Unsupported length for the {source} short-window readout")

    question = Question(
        qtype="choice", question_parts=(prompt,), question_sentence=prompt,
        options=labels, canon=tuple(range(count)), n_canon=count,
        target=(0.0,) * count, values=(0.0,) * count,
    )
    head = model.model.base
    view = model.bank.view([question])
    head.bank_tokens = view.tokens
    batch = decision.pack_questions(
        [question], view, eeg.shape[0], patches, [0], "joint",
        head_max_len=head.head_max_len, option_max_tokens=head.option_max_tokens,
    )
    del batch["target"]
    batch["source"] = source
    batch["channel_ids"] = model.channel_ids[source]
    logits = head(torch.from_numpy(eeg)[None], batch)["logits"][0]
    if logits.shape != (count,) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite score per offered label")
    return int(logits.argmax()), logits.tolist()
