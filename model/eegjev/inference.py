"""CPU inference for the trained v1.3 T1 and T2 paths, on Bench-standardized 200 Hz arrays."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import decision
from .cbramod.cbramod import CBraMod
from .spatial import SpatialTokenModel, hidden_stack
from .text import LiveTextBank, Question
from .windows import MultiWindowReadout, pack_multi_questions

FS = 200
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalized_name(name):
    return CHANNEL_ALIASES.get(name.upper(), name.upper())


class LinkedEEGJev:
    def __init__(self, checkpoint, text_encoder):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        config = saved["config"]
        layout = config["spatial_token"]["lengths"]
        self.montages = json.loads(Path(__file__).with_name("montages.json").read_text())
        if set(self.montages) != set(layout) or config["model"]["start"] != "shared":
            raise ValueError("Expected the shared v1.3 model and its registered channel layouts")
        names = sorted({normalized_name(c) for s in layout for c in self.montages[s]})
        indices = {name: i for i, name in enumerate(names)}
        self.channel_ids = {s: torch.tensor([indices[normalized_name(c)] for c in cs])
                            for s, cs in self.montages.items()}
        self.lengths = layout
        backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                           seq_len=30, n_layer=12, nhead=8)
        backbone.proj_out = nn.Identity()
        base = SpatialTokenModel(backbone=backbone, readout="joint", backbone_trainable=False,
                                 max_channels=len(names), max_patches=4)
        # All trained stacks are restored, while mmap lets this run touch just the selected montage.
        with torch.device("meta"):
            base.decision.spatial = nn.ModuleDict({
                f"{s}_{length}": hidden_stack(len(self.montages[s]) * length * FS)
                for s, lengths in layout.items() for length in lengths
            })
            base.decision.spatial_proj = nn.Linear(200, base.d_model)
            base.decision.spatial_emb = nn.Parameter(torch.empty(base.d_model))
        self.model = MultiWindowReadout(base, decision)
        state = {"base." + name: value for name, value in saved["model"].items()}
        state.update(saved["new_input_embeddings"])
        matched = self.model.load_state_dict(state, strict=True, assign=True)
        self.model.bind_options_to_windows = config["unified"]["t1_option_window_binding"]
        self.model.bind_options_to_halves = config["unified"]["t2_option_half_binding"]
        if not (self.model.bind_options_to_windows and self.model.bind_options_to_halves):
            raise ValueError("Use the final checkpoint with both trained option bindings")
        self.model.eval().requires_grad_(False)
        self.bank = LiveTextBank(text_encoder, torch.device("cpu"), keep=())
        self.receipt = {
            "seed": config["seed"], "checkpoint_epoch_zero_based": saved["epoch"],
            "strict_load": True, "model_state_entries": len(state),
            "backbone_state_entries": sum(k.startswith("base.backbone.") for k in state),
            "new_input_embeddings": sorted(saved["new_input_embeddings"]),
            "missing_keys": matched.missing_keys, "unexpected_keys": matched.unexpected_keys,
            "t1_window_binding": self.model.bind_options_to_windows,
            "t2_half_binding": self.model.bind_options_to_halves,
            "text_encoder": self.bank.encoder.config.model_type,
            "device": str(next(self.model.parameters()).device),
        }

    def _blocks(self, track, source, signals):
        expected = 2 if track == "T1" else 1
        if track not in ("T1", "T2") or len(signals) != expected:
            raise ValueError("T1 takes two short windows and T2 takes one long window")
        arrays = [torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))[None]
                  for signal in signals]
        for x in arrays:
            if x.ndim != 3 or x.shape[1] != len(self.montages[source]) or x.shape[2] % FS:
                raise ValueError("Use [channels, samples] at 200 Hz in the checkpoint montage order")
        lengths = self.lengths[source]
        if track == "T1":
            if arrays[0].shape != arrays[1].shape or arrays[0].shape[-1] // FS not in lengths:
                raise ValueError("The two T1 windows must have the same supported duration")
            return arrays, [0, 1], [0, 0]
        signal = arrays[0]
        remaining = signal.shape[-1] // FS
        blocks = []
        start = 0
        while remaining:
            length = max(k for k in lengths if k <= min(4, remaining))
            blocks.append(signal[:, :, start:start + length * FS])
            start += length * FS
            remaining -= length
        if len(blocks) > self.model.block_embedding.num_embeddings:
            raise ValueError("Window exceeds the trained number of time positions")
        return blocks, [0] * len(blocks), list(range(len(blocks)))

    @torch.inference_mode()
    def answer(self, track, source, signals, prompt, options):
        """Return the chosen option and its two logits. No label or answer index is accepted."""
        if len(options) != 2:
            raise ValueError("T1 and T2 require two presented choices")
        if track == "T2" and tuple(options) != ("first half", "second half"):
            raise ValueError("The trained T2 binding expects first half followed by second half")
        if track == "T1" and tuple(options) != ("window A", "window B"):
            raise ValueError("The trained T1 binding expects window A followed by window B")
        blocks, owners, positions = self._blocks(track, source, signals)
        question = Question(qtype="choice", question_parts=(prompt,), question_sentence=prompt,
                            options=tuple(options), canon=(0, 1), n_canon=2,
                            target=(0., 0.), values=(0., 1.))
        text = self.bank.view([question])
        self.model.base.bank_tokens = text.tokens
        counts = [x.shape[1] * (x.shape[-1] // FS) + 1 for x in blocks]
        batch = pack_multi_questions(decision, [question], text, counts)
        batch["channel_ids"] = self.channel_ids[source]
        # The shared packer also supports training. No target from the fixture enters the forward.
        del batch["target"]
        output = self.model(blocks, batch, source, owners, positions)
        logits = output["logits"][0]
        if logits.shape != (2,) or not torch.isfinite(logits).all():
            raise ValueError("EEGJev did not return two finite choice logits")
        return int(logits.argmax()), logits.tolist()
