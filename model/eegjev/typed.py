"""Choice, binary judgments, and ordered scores from the trained v1.3 heads."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Literal

import numpy as np
import torch

from .inference import LinkedEEGJev
from .short_window import answer_or_refuse, answer_short_window


Track = Literal["short", "T1", "T2"]


@dataclass(frozen=True)
class Choice:
    question: str
    options: Sequence[str]
    track: Track = "short"
    with_answerability: bool = False


@dataclass(frozen=True)
class Noul:
    question: str
    options: Sequence[str]
    true_option: str
    track: Track = "short"


@dataclass(frozen=True)
class Score:
    question: str
    options: Sequence[str]
    values: Sequence[float]


def _short_window(signals):
    if len(signals) != 1:
        raise ValueError("A short-window question takes one EEG window")
    return signals[0]


def _score_choices(model, source, signals, question, options, track):
    if track == "short":
        return answer_short_window(model, source, _short_window(signals), question, options)
    return model.answer(track, source, signals, question, options)


def _probabilities(options, logits):
    probabilities = torch.tensor(logits, dtype=torch.float64).softmax(0).tolist()
    return dict(zip(options, probabilities, strict=True))


def answer_typed(
    model: LinkedEEGJev,
    source: str,
    signals: Sequence[np.ndarray],
    questions: Mapping[str, Choice | Noul | Score],
    *,
    ordered_model: LinkedEEGJev | None = None,
) -> dict[str, dict]:
    """Answer named questions in order on the same EEG input.

    For Score, supply the matching D3 checkpoint as ordered_model, or pass it as
    model for a Score-only call. Choice can also report the separately trained
    Act head's answerability probability.
    """
    answers = {}
    for name, request in questions.items():
        if not isinstance(request, (Choice, Noul, Score)):
            raise TypeError("Use Choice, Noul, or Score for each named question")
        options = tuple(request.options)
        if len(set(options)) != len(options):
            raise ValueError("Candidate labels must be unique")

        if isinstance(request, Score):
            d3 = model if ordered_model is None else ordered_model
            if d3.temporal:
                raise ValueError("Score requires the D3 checkpoint")
            values = tuple(request.values)
            if (len(values) != len(options) or not all(map(isfinite, values))
                    or any(a >= b for a, b in zip(values, values[1:]))):
                raise ValueError("Give one finite value per bin in strictly increasing order")
            _, logits = answer_short_window(d3, source, _short_window(signals), request.question, options)
            probabilities = _probabilities(options, logits)
            answers[name] = {
                "type": "score",
                "score": fsum(probability * value
                              for probability, value in zip(probabilities.values(), values, strict=True)),
                "probabilities": probabilities,
            }
        elif isinstance(request, Noul):
            if len(options) != 2 or request.true_option not in options:
                raise ValueError("Noul takes two labels and a true_option from those labels")
            _, logits = _score_choices(model, source, signals, request.question, options, request.track)
            answers[name] = {
                "type": "noul",
                "true_option": request.true_option,
                "probability_true": _probabilities(options, logits)[request.true_option],
            }
        else:
            if request.with_answerability:
                if request.track != "short":
                    raise ValueError("The trained Act head takes a short EEG window")
                index, logits, act_logits = answer_or_refuse(
                    model, source, _short_window(signals), request.question, options,
                )
            else:
                index, logits = _score_choices(model, source, signals, request.question, options, request.track)
            answer = {
                "type": "choice",
                "choice": None if index is None else options[index],
                "probabilities": _probabilities(options, logits),
            }
            if request.with_answerability:
                answer["answerability_probability"] = _probabilities(("answer", "refuse"), act_logits)["answer"]
            answers[name] = answer
    return answers
