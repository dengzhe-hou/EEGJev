"""Source- and window-specific spatial readout used by the v1.3 decision head."""
import torch
from torch import nn
from torch.nn import functional as Fn

from . import decision as M

def hidden_stack(width, dropout=0.1):
    """Official all_patch_reps hidden layers (classifier[1:7]); the K-way output layer is a separate head."""
    return nn.Sequential(nn.Linear(width, 4 * 200), nn.ELU(), nn.Dropout(dropout),
                         nn.Linear(4 * 200, 200), nn.ELU(), nn.Dropout(dropout))


class SpatialTokenModel(M.TypedSystem1Model):
    """Joint 5c head whose window token rows are its C x P CBraMod tokens plus one spatial token (decision.spatial*)."""
    use_spatial_token = True     # False only in tests, to compare the readout copy with the base class

    def forward(self, x, batch, *, return_context=False):
        if self.use_spatial_token:
            batch = dict(batch)
            m, n_eeg = len(batch["seq_src"]), x.shape[1] * (x.shape[2] // M.PATCH_SAMPLES)
            for key, value in (("seq_src", M.SPECIAL_PAD), ("segment", M.SEG_EEG), ("opt_of_pos", M.OPT_NONE),
                               ("eeg_pos", n_eeg)):
                batch[key] = torch.cat([batch[key], batch[key].new_full((m, 1), value)], 1)
        return super().forward(x, batch, return_context=return_context)

    def _head_readout(self, feats, owner, batch):
        """TypedSystem1Model._head_readout (joint) with C x P + 1 token rows per window; the last is the spatial token."""
        dec = self.decision
        o, channels, patches, _ = feats.shape
        eeg = dec.eeg_proj(feats.reshape(o, channels * patches, M.CBRAMOD_WIDTH))
        eeg = eeg + self._channel_patch(channels, patches, batch.get("channel_ids"))
        if self.use_spatial_token:
            stack = dec.spatial[f"{batch['source']}_{patches}"]
            spatial = dec.spatial_proj(stack(feats.reshape(o, -1))) + dec.spatial_emb
            eeg = torch.cat([eeg, spatial[:, None, :]], 1)
        n_tok = eeg.shape[1]
        text = dec.text_proj(self.bank_tokens[batch["text_rows"]])
        specials = torch.cat([dec.special_emb, dec.special_emb.new_zeros(1, self.d_model)])
        table = torch.cat([specials, text, eeg.reshape(o * n_tok, -1)])
        eeg_p = batch["eeg_pos"]
        base = M.N_SPECIAL_ROWS + text.shape[0]
        src = torch.where(eeg_p >= 0, base + owner[:, None] * n_tok + eeg_p, batch["seq_src"])
        segments = torch.cat([dec.segment_emb.new_zeros(1, self.d_model), dec.segment_emb])
        rows = Fn.embedding(src, table)
        rows = rows + self._bind_markers(rows, batch)
        h = rows + Fn.embedding(batch["segment"], segments) + Fn.embedding(batch["qtype"], dec.type_emb)[:, None, :]
        allowed = self.attention_mask(batch["opt_of_pos"])
        for layer in dec.head:
            h = layer(h, allowed)
        index = batch["marker_pos"][:, :, None].expand(-1, -1, h.shape[-1])
        return dec.scorer(torch.gather(h, 1, index)).squeeze(-1), h[:, 0]
