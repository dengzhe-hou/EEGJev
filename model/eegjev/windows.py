"""CPU-developed new-track input path. This module does not modify a frozen run."""
import torch
from torch import nn
from torch.nn import functional as F


def split_long_window(x, fs=200, chunk_seconds=4):
    """Keep every sample, in order. The final shorter chunk is retained."""
    if x.ndim != 3 or x.shape[-1] % fs:
        raise ValueError("Expected [batch, channels, whole seconds at200Hz]")
    return list(x.split(fs * chunk_seconds, dim=-1))


def pack_multi_questions(M, questions, bank, token_counts):
    """Use the existing text/option packer, replacing its single EEG block."""
    out = M.pack_questions(questions, bank, 1, 1, list(range(len(questions))), "joint")
    count = sum(token_counts)
    grids = []
    for i in range(len(questions)):
        valid = out["opt_of_pos"][i] != M.OPT_PAD
        fields = torch.stack([out[k][i, valid] for k in
                              ("seq_src", "segment", "opt_of_pos", "eeg_pos")], -1)
        at = int(torch.nonzero(fields[:, 3] >= 0)[0])
        eeg = fields.new_tensor([[M.SPECIAL_PAD, M.SEG_EEG, M.OPT_NONE, j]
                                 for j in range(count)])
        grids.append(torch.cat([fields[:at], eeg, fields[at + 1:]], 0))
    longest = max(len(x) for x in grids)
    pad = grids[0].new_tensor([M.SPECIAL_PAD, M.SEG_NONE, M.OPT_PAD, -1])
    grid = torch.stack([torch.cat([g, pad.expand(longest-len(g), -1)]) for g in grids])
    for j, key in enumerate(("seq_src", "segment", "opt_of_pos", "eeg_pos")):
        out[key] = grid[..., j]
    return out


def half_blocked(blocks, opt, eegpos):
    """[n, query, key] True where an option-k token (k = 0 first half, 1 second half) would attend to an EEG token of a
    block that does not overlap half k. Blocks are consecutive in time; block j spans its seconds after blocks 0..j-1."""
    seconds = [b.shape[-1] // 200 for b in blocks]
    counts = [b.shape[1] * s + 1 for b, s in zip(blocks, seconds)]          # channel x patch tokens + spatial token
    starts = [sum(seconds[:j]) for j in range(len(blocks))]
    mid = sum(seconds) / 2
    of_token = torch.repeat_interleave(torch.arange(len(blocks), device=eegpos.device),
                                       torch.tensor(counts, device=eegpos.device))
    in_half = torch.tensor([[s < mid for s in starts], [s + d > mid for s, d in zip(starts, seconds)]],
                           device=eegpos.device)                              # [2, blocks]
    key = eegpos >= 0
    block = of_token[eegpos.clamp(min=0)]                                    # [n, key]
    blocked = torch.zeros(opt.shape[0], opt.shape[1], eegpos.shape[1], dtype=torch.bool, device=eegpos.device)
    for k in (0, 1):
        outside = key & ~in_half[k][block]                                   # [n, key]
        blocked |= (opt == k)[:, :, None] & outside[:, None, :]
    return blocked


class MultiWindowReadout(nn.Module):
    """Reuse frozen CBraMod and the existing spatial/head modules.

    Each short block keeps its original source/length spatial stack.
    New learned embeddings identify window A/B and the ordered block in a long
    input. Inference restores both embeddings from the trained checkpoint. Long-window context is assembled in the
    decision head; CBraMod processes each <=4s block separately.
    """
    def __init__(self, base, M, max_blocks=8):
        super().__init__()
        if base.readout != "joint" or base.film_layers:
            raise ValueError("New tracks currently support the frozen joint/no-FiLM path")
        self.base, self.M = base, M
        self.owner_embedding = nn.Embedding(2, base.d_model)
        self.block_embedding = nn.Embedding(max_blocks, base.d_model)
        nn.init.normal_(self.owner_embedding.weight, std=.02)
        nn.init.normal_(self.block_embedding.weight, std=.02)
        # 17 Nov 2026: when True, for two-window inputs (T1, owners [0, 1]) the tokens of option k do not attend to
        # the EEG tokens of the other window. Set from config["unified"]["t1_option_window_binding"]; False is the
        # behaviour of the heads trained on 16-17 Nov.
        self.bind_options_to_windows = False
        # 18 Nov 2026: when True, for long inputs (T2: one owner, several ordered blocks; options "first half",
        # "second half" in this order) the tokens of option k do not attend to the EEG tokens of blocks outside half k
        # (a block that straddles the midpoint, 12-16 s of a 30 s window, belongs to both halves). Set from
        # config["unified"]["t2_option_half_binding"]; False is the behaviour of the heads trained on 16-17 Nov.
        self.bind_options_to_halves = False
        for parameter in base.backbone.parameters():
            parameter.requires_grad_(False)
        base.backbone_trainable = False
        base.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.backbone.eval()
        return self

    def encode_blocks(self, blocks, source, owners, positions, channel_ids=None,
                      legacy_single=False):
        if len(blocks) != len(owners) or len(blocks) != len(positions):
            raise ValueError("Every input block needs an owner and time position")
        if hasattr(self.base.backbone, "by_source"):
            self.base.backbone.current = source
        tokens = []
        for x, owner, position in zip(blocks, owners, positions):
            if x.shape[-1] not in (200, 400, 800):
                raise ValueError("Reuse existing spatial stacks for1/2/4s blocks only")
            with torch.no_grad():
                feats = self.base.eeg_prefix(x)
            n, channels, patches, width = feats.shape
            dec = self.base.decision
            eeg = dec.eeg_proj(feats.reshape(n, channels*patches, width))
            eeg = eeg + self.base._channel_patch(channels, patches, channel_ids)
            spatial = dec.spatial_proj(
                dec.spatial[f"{source}_{patches}"](feats.reshape(n, -1))
            ) + dec.spatial_emb
            eeg = torch.cat([eeg, spatial[:, None]], 1)
            if not legacy_single:
                eeg = eeg + self.owner_embedding.weight[owner] + self.block_embedding.weight[position]
            tokens.append(eeg)
        return torch.cat(tokens, 1)

    def forward(self, blocks, batch, source, owners, positions, legacy_single=False):
        eeg = self.encode_blocks(blocks, source, owners, positions,
                                 batch.get("channel_ids"), legacy_single)
        base, M, dec = self.base, self.M, self.base.decision
        n, ntok, width = eeg.shape
        text = dec.text_proj(base.bank_tokens[batch["text_rows"]])
        specials = torch.cat([dec.special_emb, dec.special_emb.new_zeros(1, width)])
        table = torch.cat([specials, text, eeg.reshape(n*ntok, width)])
        eegpos = batch["eeg_pos"]
        owner = torch.arange(n, device=eeg.device)
        src = torch.where(eegpos >= 0,
                          M.N_SPECIAL_ROWS + len(text) + owner[:, None]*ntok + eegpos,
                          batch["seq_src"])
        segments = torch.cat([dec.segment_emb.new_zeros(1, width), dec.segment_emb])
        rows = F.embedding(src, table)
        rows = rows + base._bind_markers(rows, batch)
        h = rows + F.embedding(batch["segment"], segments)
        h = h + F.embedding(batch["qtype"], dec.type_emb)[:, None]
        mask = base.attention_mask(batch["opt_of_pos"])
        if self.bind_options_to_windows and list(owners) == [0, 1]:
            opt = batch["opt_of_pos"]
            first = blocks[0].shape[1] * (blocks[0].shape[-1] // 200) + 1      # EEG tokens of window A
            window = torch.where(eegpos >= 0, (eegpos >= first).long(), torch.full_like(eegpos, -1))
            blocked = (opt[:, :, None] >= 0) & (window[:, None, :] >= 0) & (opt[:, :, None] != window[:, None, :])
            mask = mask & ~blocked[:, None]
        if self.bind_options_to_halves and len(blocks) > 1 and all(o == 0 for o in owners):
            mask = mask & ~half_blocked(blocks, batch["opt_of_pos"], eegpos)[:, None]
        for layer in dec.head:
            h = layer(h, mask)
        positions = batch["marker_pos"][:, :, None].expand(-1, -1, width)
        logits = dec.scorer(torch.gather(h, 1, positions)).squeeze(-1).float()
        logits = logits.masked_fill(~batch["option_mask"], M.LOGIT_FILL)
        return {"logits": logits,
                "act_logits": base._act(logits, batch["option_mask"], h[:, 0])}
