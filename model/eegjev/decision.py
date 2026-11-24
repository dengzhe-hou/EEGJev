"""v1.3 model and question packing. The trained backbone is restored by the checkpoint loader."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .text import ORDINAL_QTYPES, QTYPE_ID, QTYPES, pair_string

READOUTS = ("joint", "independent", "late_binding", "pairwise", "cosine")
HEAD_READOUTS = ("joint", "independent", "late_binding")
FILM_CONDITIONS = ("question", "constant", "other_family")
PATCH_SAMPLES = 200
CBRAMOD_WIDTH = 200
CBRAMOD_LAYERS = 12
# Specials table rows. PAD is a fixed zero row, never trained.
SPECIAL_CLS, SPECIAL_SEP, SPECIAL_MASK, SPECIAL_PAD = 0, 1, 2, 3
N_SPECIAL_ROWS = 4
SEG_NONE, SEG_QUESTION, SEG_OPTION, SEG_EEG = 0, 1, 2, 3
OPT_NONE, OPT_PAD = -1, -2
TEMP_MIN, TEMP_MAX = 0.5, 5.0      # laya/common.py TEMP_MIN, TEMP_MAX
CALIBRATION_PROTOCOL = "positive_unclipped_temperature_20261021"
LOGIT_FILL = -1e4                  # laya DecisionModel masked_fill value


class HeadLayer(nn.TransformerEncoderLayer):
    """Laya head layer: nn.TransformerEncoderLayer(d, h, ffn, dropout, batch_first, norm_first).

    Same parameters and math (default ReLU FFN, as in Laya). Attention runs
    through scaled_dot_product_attention so one boolean mask [M,1,S,S] or
    [M,1,1,S] (True = may attend) carries padding and the I8 option cuts.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)

    def _masked_attention(self, x, allowed):
        n, s, d = x.shape
        heads = self.self_attn.num_heads
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.view(n, s, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.self_attn.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(n, s, d)
        return self.dropout1(self.self_attn.out_proj(out))

    def forward(self, src, allowed):
        x = src + self._masked_attention(self.norm1(src), allowed)
        return x + self._ff_block(self.norm2(x))


class QFormerLayer(nn.Module):
    """LEAF.py QFormerLayer: pre-norm self-attention, cross-attention to tokens of width dim_kv, FFN."""

    def __init__(self, dim_q, dim_kv, n_heads, ff_scale=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_q, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_q)
        self.cross_attn = nn.MultiheadAttention(dim_q, n_heads, batch_first=True, kdim=dim_kv, vdim=dim_kv)
        self.norm2 = nn.LayerNorm(dim_q)
        self.ff = nn.Sequential(nn.Linear(dim_q, ff_scale * dim_q), nn.GELU(), nn.Linear(ff_scale * dim_q, dim_q))
        self.norm3 = nn.LayerNorm(dim_q)

    def forward(self, q, kv):
        normed = self.norm1(q)
        q = q + self.self_attn(normed, normed, normed, need_weights=False)[0]
        q = q + self.cross_attn(self.norm2(q), kv, kv, need_weights=False)[0]
        return q + self.ff(self.norm3(q))


class TextConditionedQFormer(nn.Module):
    """LEAF.py TextConditionedQFormer with the same parameter names and math.

    max_pos learnable positions (LEAF: max_tokens * 2; here C x P = 650), FiLM
    of the tokens by Linear(text_dim -> 2 dim_kv), learnable queries, final
    LayerNorm, Dropout(0.3) and Linear(num_q dim_q -> text_dim). Returns the
    output embedding and the final normalized queries (the act head context).
    A LEAF state_dict loads with strict=True (tests/test_typed_system1.py).
    """

    def __init__(self, num_q=8, dim_q=256, dim_kv=200, n_heads=8, n_layers=4, ff_scale=4, text_dim=768,
                 max_pos=650):
        super().__init__()
        self.base_queries = nn.Parameter(torch.randn(num_q, dim_q) / dim_q ** 0.5)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_pos, dim_kv))
        self.film = nn.Linear(text_dim, 2 * dim_kv)
        self.layers = nn.ModuleList(QFormerLayer(dim_q, dim_kv, n_heads, ff_scale) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(dim_q)
        self.output_embed = nn.Sequential(nn.Flatten(1), nn.Dropout(0.3), nn.Linear(dim_q * num_q, text_dim))

    def forward(self, tokens, instruct):
        n = tokens.shape[1]   # <= max_pos: eeg_prefix bounds C and P
        gamma, beta = self.film(instruct).chunk(2, dim=-1)
        tokens = (1 + gamma[:, None, :]) * tokens + beta[:, None, :]
        tokens = tokens + self.pos_embedding[:, :n]
        q = self.base_queries[None].expand(len(tokens), -1, -1)
        for layer in self.layers:
            q = layer(q, tokens)
        q = self.final_norm(q)
        return self.output_embed(q), q


def _mlp_scorer(d, out):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))


class TypedSystem1Model(nn.Module):
    def __init__(self, *, backbone, text_dim=768, readout="joint", d_model=256, head_layers=2, head_heads=4,
                 head_ffn=1024, dropout=0.1, backbone_trainable=True, film_layers=0,
                 film_condition="question", max_channels=65, max_patches=10, n_act=2,
                 head_max_len=192, option_max_tokens=48, late_hidden=128,
                 cosine_temperature="none", cosine_blend_probability=0.0, cosine_eval_blend=False,
                 qformer_queries=8, qformer_layers=4, qformer_heads=8, qformer_ff_scale=4,
                 pairwise_hidden=(256, 128), pairwise_dropout=0.1):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if film_condition not in FILM_CONDITIONS:
            raise ValueError(f"film_condition must be one of {FILM_CONDITIONS}")
        if not 0 <= film_layers <= CBRAMOD_LAYERS:
            raise ValueError("film_layers must be in 0..12")
        if cosine_temperature not in ("none", "learnable"):
            raise ValueError("cosine_temperature must be none or learnable")
        self.readout = readout
        self.d_model = d_model
        self.text_dim = text_dim
        self.backbone_trainable = bool(backbone_trainable)
        self.film_layers = int(film_layers)
        self.film_condition = film_condition
        self.max_channels, self.max_patches = max_channels, max_patches
        self.head_max_len, self.option_max_tokens = head_max_len, option_max_tokens
        self.cosine_blend_probability = float(cosine_blend_probability)
        self.cosine_eval_blend = bool(cosine_eval_blend)

        self.backbone = backbone
        for parameter in self.backbone.parameters():
            # Official full fine-tuning enables every backbone parameter.
            parameter.requires_grad_(self.backbone_trainable)

        dec = nn.Module()
        d = d_model
        if self.film_layers:
            dec.film = nn.ModuleList(nn.Linear(text_dim, 2 * CBRAMOD_WIDTH) for _ in range(self.film_layers))
            for layer in dec.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        if readout in HEAD_READOUTS:
            dec.text_proj = nn.Linear(text_dim, d)
            dec.eeg_proj = nn.Linear(CBRAMOD_WIDTH, d)
            dec.type_emb = nn.Parameter(torch.randn(len(QTYPES), d) * 0.02)
            dec.channel_emb = nn.Parameter(torch.randn(max_channels, d) * 0.02)
            dec.patch_emb = nn.Parameter(torch.randn(max_patches, d) * 0.02)
            dec.special_emb = nn.Parameter(torch.randn(3, d) * 0.02)      # CLS, SEP, MASK
            dec.segment_emb = nn.Parameter(torch.randn(3, d) * 0.02)      # question, option, EEG
            dec.head = nn.ModuleList(HeadLayer(d, head_heads, head_ffn, dropout) for _ in range(head_layers))
            if readout == "late_binding":
                # Replaces A's scorer: 513 * late_hidden + 769 parameters; 128
                # gives 66,433 against the scorer's 66,561 (plan 1.7 parity).
                dec.late_context = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, late_hidden), nn.GELU(),
                                                 nn.Linear(late_hidden, d))
                dec.late_logit_scale = nn.Parameter(torch.tensor(-0.5 * math.log(d)))
            else:
                dec.scorer = _mlp_scorer(d, 1)
        elif readout == "pairwise":
            dec.pair_eeg = nn.Linear(CBRAMOD_WIDTH, d)
            dec.pair_text = nn.Linear(text_dim, d)
            layers, width = [], 4 * d
            for hidden in pairwise_hidden:
                layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(pairwise_dropout)]
                width = hidden
            dec.pair_mlp = nn.Sequential(*layers, nn.Linear(width, 1))
        else:  # cosine: LEAF's Q-Former directly on CBraMod tokens (dim_kv 200), plan 1.7
            dec.qformer = TextConditionedQFormer(qformer_queries, d, CBRAMOD_WIDTH, qformer_heads, qformer_layers,
                                                 qformer_ff_scale, text_dim, max_channels * max_patches)
            if cosine_temperature == "learnable":
                dec.cos_logit_scale = nn.Parameter(torch.zeros(()))  # exp(0) = 1 = LEAF's unscaled cosine
        dec.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.decision = dec
        self.register_buffer("bank_tokens", torch.zeros(0, text_dim), persistent=False)
        self.register_buffer("bank_sentence", torch.zeros(0, text_dim), persistent=False)
        self.text_bank = None
        if not self.backbone_trainable:
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()  # a frozen backbone never applies its dropout
        return self

    def attach_text_bank(self, bank):
        """Use a TextBank's frozen states. They are buffers outside the state_dict."""
        if bank.text_dim != self.text_dim:
            raise ValueError("Text cache width does not match the model's text_dim")
        device = self.bank_tokens.device
        self.bank_tokens = torch.from_numpy(bank.token_states).to(device)
        self.bank_sentence = torch.from_numpy(bank.sentence).to(device)
        self.text_bank = bank
        return self

    # ------------------------------------------------------------------ EEG
    def eeg_prefix(self, x):
        """Patch embedding and the first 12 - film_layers CBraMod layers, once per window."""
        n, channels, samples = x.shape
        if samples % PATCH_SAMPLES or not 1 <= samples // PATCH_SAMPLES <= self.max_patches:
            raise ValueError(f"EEG length must be 200 x P with 1 <= P <= {self.max_patches}: {samples}")
        if channels > self.max_channels:
            raise ValueError(f"At most {self.max_channels} channels are supported: {channels}")
        h = x.reshape(n, channels, samples // PATCH_SAMPLES, PATCH_SAMPLES)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.backbone_trainable):
            h = self.backbone.patch_embedding(h, None)
            for layer in self.backbone.encoder.layers[:CBRAMOD_LAYERS - self.film_layers]:
                h = layer(h)
        return h

    def film_condition_vectors(self, batch):
        if self.film_condition == "other_family":
            # B-null: a prompt of another task family per row, chosen by the
            # runner (pack_questions null_prompts), never the row's own family.
            return self.bank_sentence.index_select(0, batch["null_sentence"])
        cond = self.bank_sentence.index_select(0, batch["question_sentence"])
        if self.film_condition == "constant":
            return torch.zeros_like(cond)
        return cond

    def encode_eeg(self, x, batch):
        """Return CBraMod features [O, C, P, 200] and the owner row of every question."""
        h = self.eeg_prefix(x)
        eeg_index = batch["eeg_index"]
        if not self.film_layers:
            return h, eeg_index
        cond = self.film_condition_vectors(batch)
        h = h.index_select(0, eeg_index)
        for layer, film in zip(self.backbone.encoder.layers[CBRAMOD_LAYERS - self.film_layers:],
                               self.decision.film):
            h = layer(h)  # grad stays enabled so FiLM learns even with a frozen backbone
            gamma, beta = film(cond).chunk(2, dim=-1)
            h = (1 + gamma[:, None, None, :]) * h + beta[:, None, None, :]
        return h, torch.arange(len(eeg_index), device=h.device)

    def _channel_patch(self, channels, patches, channel_ids=None):
        """channel_ids names each row's electrode when sources differ; otherwise rows index the table directly."""
        dec = self.decision
        rows = dec.channel_emb[:channels] if channel_ids is None else dec.channel_emb[channel_ids]
        return (rows[:, None, :] + dec.patch_emb[None, :patches, :]).reshape(channels * patches, -1)

    # ------------------------------------------------------------- readouts
    def forward(self, x, batch, *, return_context=False):
        feats, owner = self.encode_eeg(x, batch)
        if self.readout in HEAD_READOUTS:
            logits, context = self._head_readout(feats, owner, batch)
        elif self.readout == "pairwise":
            logits, context = self._pairwise_readout(feats, owner, batch)
        else:
            logits, context = self._cosine_readout(feats, owner, batch)
        option_mask = batch["option_mask"]
        logits = logits.float().masked_fill(~option_mask, LOGIT_FILL)
        output = {"logits": logits, "act_logits": self._act(logits, option_mask, context)}
        if return_context:
            output["alignment_context"] = context
        return output

    def _head_readout(self, feats, owner, batch):
        dec = self.decision
        o, channels, patches, _ = feats.shape
        eeg = dec.eeg_proj(feats.reshape(o, channels * patches, CBRAMOD_WIDTH))
        eeg = eeg + self._channel_patch(channels, patches, batch.get("channel_ids"))
        text = dec.text_proj(self.bank_tokens[batch["text_rows"]])   # frozen bank: no gradient
        specials = torch.cat([dec.special_emb, dec.special_emb.new_zeros(1, self.d_model)])
        table = torch.cat([specials, text, eeg.reshape(o * channels * patches, -1)])
        eeg_p = batch["eeg_pos"]
        base = N_SPECIAL_ROWS + text.shape[0]
        src = torch.where(eeg_p >= 0, base + owner[:, None] * (channels * patches) + eeg_p, batch["seq_src"])
        segments = torch.cat([dec.segment_emb.new_zeros(1, self.d_model), dec.segment_emb])
        # F.embedding: its CPU backward is deterministic; advanced indexing with
        # repeated rows accumulates in a thread-dependent order.
        rows = F.embedding(src, table)
        if self.readout != "late_binding":
            rows = rows + self._bind_markers(rows, batch)
        h = rows + F.embedding(batch["segment"], segments) + F.embedding(batch["qtype"], dec.type_emb)[:, None, :]
        allowed = self.attention_mask(batch["opt_of_pos"])
        for layer in dec.head:
            h = layer(h, allowed)
        context = h[:, 0]
        if self.readout == "late_binding":
            ctx = dec.late_context(context)
            tokens = F.embedding(batch["option_tokens"], text) * batch["option_token_mask"][..., None]
            options = tokens.sum(2) / batch["option_token_mask"].sum(2, keepdim=True).clamp(min=1)
            logits = (ctx[:, None, :] * options).sum(-1) * dec.late_logit_scale.exp()
        else:
            index = batch["marker_pos"][:, :, None].expand(-1, -1, h.shape[-1])
            logits = dec.scorer(torch.gather(h, 1, index)).squeeze(-1)
        return logits, context

    @staticmethod
    def _bind_markers(rows, batch):
        """Per position: for the [MASK] of option k, the mean of option k's projected tokens; else 0.

        rows [M, S, d] are the projected inputs before segment and type
        embeddings. Padded option slots have no marker and add nothing.
        """
        opt = batch["opt_of_pos"]
        k = batch["option_mask"].shape[1]
        option_ids = torch.arange(k, device=opt.device)[None, :, None]
        member = ((opt[:, None, :] == option_ids) & (batch["seq_src"] >= N_SPECIAL_ROWS)[:, None, :]).to(rows.dtype)
        pooled = torch.bmm(member, rows) / member.sum(-1, keepdim=True).clamp(min=1)          # [M, K, d]
        marker = F.one_hot(batch["marker_pos"], rows.shape[1]).to(rows.dtype) * batch["option_mask"][..., None]
        return torch.bmm(marker.transpose(1, 2), pooled)                                      # [M, S, d]

    def attention_mask(self, opt_of_pos):
        valid = opt_of_pos != OPT_PAD
        if self.readout != "independent":
            return valid[:, None, None, :]
        key_is_option = opt_of_pos >= 0
        allowed = (~key_is_option[:, None, :]) | (opt_of_pos[:, :, None] == opt_of_pos[:, None, :])
        return (allowed & valid[:, None, :])[:, None, :, :]

    def _pairwise_readout(self, feats, owner, batch):
        dec = self.decision
        e = dec.pair_eeg(feats.mean(dim=(1, 2))).index_select(0, owner)       # [M, d]
        t = dec.pair_text(self.bank_sentence[batch["pair_sentence"]])          # [M, K, d]
        e = e[:, None, :].expand_as(t)
        logits = dec.pair_mlp(torch.cat([e, t, e * t, (e - t).abs()], dim=-1)).squeeze(-1)
        return logits, e[:, 0]

    def _cosine_readout(self, feats, owner, batch):
        dec = self.decision
        o, channels, patches, _ = feats.shape
        tokens = feats.reshape(o, channels * patches, CBRAMOD_WIDTH).index_select(0, owner)
        question = self.bank_sentence.index_select(0, batch["question_sentence"])
        options = F.normalize(self.bank_sentence[batch["option_sentence"]], dim=-1)
        mask = batch["option_mask"]
        mean = F.normalize((options * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)
        blended = F.normalize((question + mean) / 2.0, dim=-1)
        if self.training:
            use = torch.rand(len(question), device=question.device) < self.cosine_blend_probability
        else:
            use = torch.full((len(question),), self.cosine_eval_blend, device=question.device)
        question = torch.where(use[:, None], blended, question)
        output, q = dec.qformer(tokens, question)
        embedding = F.normalize(output, dim=-1)
        logits = (embedding[:, None, :] * options).sum(-1)
        if hasattr(dec, "cos_logit_scale"):
            logits = logits * dec.cos_logit_scale.exp()
        return logits, q.mean(1)

    def _act(self, logits, option_mask, context):
        """Laya act head; every input is detached so it never moves the decision."""
        p = torch.softmax(logits.detach(), -1)
        k = option_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], -1)
        return self.decision.act_head(torch.cat([context.detach().float(), feats], -1))

    def act_parameters(self):
        return list(self.decision.act_head.parameters())

    def decision_parameters(self):
        act = {id(p) for p in self.act_parameters()}
        return [p for p in self.parameters() if id(p) not in act]


def pack_questions(questions, bank, channels, patches, eeg_index, readout,
                   head_max_len=192, option_max_tokens=48, null_prompts=None):
    """Turn Question objects into index tensors (CPU). See TypedSystem1Model.forward.

    Laya length rule (laya/common.py L84-96): each option keeps its marker plus
    at most option_max_tokens tokens; if all options leave fewer than 16 of the
    head_max_len budget, each option is cut to max(4, (head_max_len - 16) // K);
    the question keeps max(8, remaining budget) tokens. EEG tokens are never cut.
    null_prompts (one string per question) feed the B-null 'other_family' FiLM.
    """
    m = len(questions)
    if len(eeg_index) != m or (null_prompts is not None and len(null_prompts) != m):
        raise ValueError("Every question needs its EEG window index (and its null prompt, if given)")
    k_max = max(len(q.options) for q in questions)
    if min(len(q.options) for q in questions) < 2:
        raise ValueError("Every question needs at least two options")
    c_max = max(q.n_canon for q in questions)
    out = {
        "eeg_index": torch.as_tensor(np.asarray(eeg_index, dtype=np.int64)),
        "qtype": torch.tensor([QTYPE_ID[q.qtype] for q in questions], dtype=torch.long),
        "option_mask": torch.zeros(m, k_max, dtype=torch.bool),
        "canon_index": torch.zeros(m, k_max, dtype=torch.long),
        "values": torch.zeros(m, k_max, dtype=torch.float32),
        "target": torch.zeros(m, c_max, dtype=torch.float32),
        "canon_mask": torch.zeros(m, c_max, dtype=torch.bool),
        "is_ordinal": torch.tensor([q.qtype in ORDINAL_QTYPES for q in questions]),
        "question_sentence": torch.tensor([bank.sentence_row(q.question_sentence) for q in questions]),
        "option_sentence": torch.zeros(m, k_max, dtype=torch.long),
    }
    for i, q in enumerate(questions):
        k = len(q.options)
        out["option_mask"][i, :k] = True
        out["canon_index"][i, :k] = torch.tensor(q.canon)
        out["values"][i, :k] = torch.tensor(q.values, dtype=torch.float32)
        out["target"][i, :q.n_canon] = torch.tensor(q.target, dtype=torch.float32)
        out["canon_mask"][i, list(q.canon)] = True
        out["option_sentence"][i, :k] = torch.tensor([bank.sentence_row(o) for o in q.options])
    if null_prompts is not None:
        out["null_sentence"] = torch.tensor([bank.sentence_row(p) for p in null_prompts], dtype=torch.long)
    if readout == "pairwise":
        pair = torch.zeros(m, k_max, dtype=torch.long)
        for i, q in enumerate(questions):
            pair[i, :len(q.options)] = torch.tensor(
                [bank.sentence_row(pair_string(q.question_sentence, o)) for o in q.options])
        out["pair_sentence"] = pair
    if readout not in HEAD_READOUTS:
        return out

    n_eeg = channels * patches
    local = {}

    def localize(rows):
        return [local.setdefault(r, len(local)) for r in rows]

    seqs, option_lists = [], []
    for q in questions:
        opts = [list(bank.token_rows(o))[:option_max_tokens] for o in q.options]
        budget = head_max_len - sum(len(o) + 1 for o in opts)
        if budget < 16:
            per = max(4, (head_max_len - 16) // len(opts))
            opts = [o[:per - 1] for o in opts]   # per counts the marker, as in Laya
            budget = head_max_len - sum(len(o) + 1 for o in opts)
        q_rows = [r for part in q.question_parts for r in bank.token_rows(part)][:max(8, budget)]
        q_local = localize(q_rows)
        opt_local = [localize(o) for o in opts]
        # (src, segment, option id, eeg position)
        seq = [(SPECIAL_CLS, SEG_NONE, OPT_NONE, -1)]
        seq += [(N_SPECIAL_ROWS + r, SEG_QUESTION, OPT_NONE, -1) for r in q_local]
        seq.append((SPECIAL_SEP, SEG_NONE, OPT_NONE, -1))
        markers = []
        if readout != "late_binding":
            for j, rows in enumerate(opt_local):
                markers.append(len(seq))
                seq.append((SPECIAL_MASK, SEG_OPTION, j, -1))
                seq += [(N_SPECIAL_ROWS + r, SEG_OPTION, j, -1) for r in rows]
            seq.append((SPECIAL_SEP, SEG_NONE, OPT_NONE, -1))
        seq += [(SPECIAL_PAD, SEG_EEG, OPT_NONE, p) for p in range(n_eeg)]
        seq.append((SPECIAL_SEP, SEG_NONE, OPT_NONE, -1))
        seqs.append((seq, markers))
        option_lists.append(opt_local)
    s_max = max(len(seq) for seq, _ in seqs)
    pad = (SPECIAL_PAD, SEG_NONE, OPT_PAD, -1)
    grid = torch.tensor([seq + [pad] * (s_max - len(seq)) for seq, _ in seqs], dtype=torch.long)
    out.update(seq_src=grid[..., 0], segment=grid[..., 1], opt_of_pos=grid[..., 2], eeg_pos=grid[..., 3],
               text_rows=torch.tensor(sorted(local, key=local.get), dtype=torch.long))
    if readout == "late_binding":
        l_max = max(len(rows) for opts in option_lists for rows in opts)
        tokens = torch.zeros(m, k_max, l_max, dtype=torch.long)
        token_mask = torch.zeros(m, k_max, l_max, dtype=torch.float32)
        for i, opts in enumerate(option_lists):
            for j, rows in enumerate(opts):
                tokens[i, j, :len(rows)] = torch.tensor(rows)
                token_mask[i, j, :len(rows)] = 1.0
        out.update(option_tokens=tokens, option_token_mask=token_mask)
    else:
        marker_pos = torch.zeros(m, k_max, dtype=torch.long)
        for i, (_, markers) in enumerate(seqs):
            marker_pos[i, :len(markers)] = torch.tensor(markers)
        out["marker_pos"] = marker_pos
    return out


def batch_to(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
