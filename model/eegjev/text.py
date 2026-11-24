"""Typed questions and frozen MPNet token states used by the v1.3 head."""
from dataclasses import dataclass

import torch

QTYPES = ("choice", "score", "noul", "scalar")
QTYPE_ID = {name: index for index, name in enumerate(QTYPES)}
ORDINAL_QTYPES = ("score", "scalar")

@dataclass(frozen=True)
class Question:
    qtype: str
    question_parts: tuple     # token-level question = concatenation of these strings' token states
    question_sentence: str    # sentence vector used by cosine / FiLM / pairwise
    options: tuple            # presented option strings
    canon: tuple              # canonical option index of each presented option
    n_canon: int
    target: tuple             # canonical target distribution (length n_canon)
    values: tuple             # numeric value of each presented option
    meta: tuple = ()          # (key, value) pairs recorded for audits (prompt kind, opaque, subset)


def pair_string(question_sentence, option):
    """(question, option) text for the PhysioBench-style pair scorer."""
    return f"{question_sentence} {option}"


class BankView:
    """The strings of one microbatch, in the interface pack_questions expects."""

    def __init__(self, states):
        self.rows, start = {}, 0
        for text, state in states.items():
            self.rows[text] = range(start, start + len(state))
            start += len(state)
        self.tokens = torch.cat(list(states.values()))

    def token_rows(self, text):
        return self.rows[text]

    def sentence_row(self, text):
        return 0      # sentence vectors serve only the cosine and FiLM readouts


class LiveTextBank:
    """Frozen MPNet token states without <s> and </s>. Fixed strings stay in memory.

    Per-example candidate-name mappings have too many forms to cache, so they are encoded when drawn.
    """

    def __init__(self, path, device, keep):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval().requires_grad_(False)
        self.device, self.keep, self.kept = device, set(keep), {}
        self.limit = 384

    @torch.no_grad()
    def state(self, text):
        if text in self.kept:
            return self.kept[text]
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] >= self.limit:
            raise ValueError(f"{text!r} would be truncated at {self.limit} MPNet tokens")
        state = self.encoder(**inputs).last_hidden_state[0, 1:-1].float()
        if text in self.keep:
            self.kept[text] = state
        return state

    def view(self, questions):
        strings = [part for q in questions for part in (*q.question_parts, *q.options)]
        return BankView({text: self.state(text) for text in dict.fromkeys(strings)})
