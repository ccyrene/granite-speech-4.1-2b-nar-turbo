"""GraniteSpeechNarForASR — pure-torch single-pass NAR speech recognizer.

Pipeline (matches the reference exactly):
  1. Conformer CTC encoder -> BPE CTC logits + multi-layer hidden states.
  2. CTC greedy collapse -> initial token hypothesis.
  3. Q-Former projector -> audio embeddings (downsampled).
  4. Build packed LLM input [audio_i ; interleaved-hypothesis_i] per sample.
  5. Bidirectional LLM -> per-position logits over the editing slots.
  6. CTC greedy collapse on the text positions -> final transcript token ids.

P2 batching notes (P2 optimization pass): the per-sample loops (collapse, packing, confidence)
are replaced by flat batched equivalents that are bit-exact by construction and cost a handful
of host syncs per BATCH instead of several per sample. Host-known lengths (``encoder_lengths``
from the feature extractor) replace ``attention_mask.sum().tolist()`` round-trips when provided.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import GraniteSpeechNarConfig
from .conformer import CTCEncoder
from .projector import Projector
from .llm import GraniteLM
from . import attn_backends


@dataclass
class ASROutput:
    preds: list[torch.Tensor] | None = None
    logits: list[torch.Tensor] | None = None
    encoder_logits: torch.Tensor | None = None
    encoder_preds: list[torch.Tensor] | None = None
    routes: list[str] | None = None            # per-sample route (adaptive path only)
    preds_host: list[list[int]] | None = None  # host copies of preds (ONE batched D2H) for decode


# --------------------------------------------------------------------------- #
# Flat batched index machinery (P2.2/P2.3). All host inputs are python ints, so
# building the index tensors enqueues async H2D copies but never blocks on the GPU.
# --------------------------------------------------------------------------- #
def _as_idx(vals, device) -> torch.Tensor:
    return torch.as_tensor(vals, dtype=torch.long, device=device)


def _excl_cumsum(lens: list[int]) -> list[int]:
    out, off = [], 0
    for x in lens:
        out.append(off)
        off += x
    return out


def _ranges_concat(starts: list[int], lens: list[int], device, total: int | None = None) -> torch.Tensor:
    """Concatenation of half-open integer ranges ``[s_k, s_k + l_k)`` as one index tensor."""
    if total is None:
        total = sum(lens)
    if total == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    ln = _as_idx(lens, device)
    off = torch.repeat_interleave(_as_idx(starts, device), ln, output_size=total)
    base = torch.repeat_interleave(_as_idx(_excl_cumsum(lens), device), ln, output_size=total)
    return off + torch.arange(total, dtype=torch.long, device=device) - base


def _flatten_dense_rows(x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """``[B, P, ...] -> [sum(lengths), ...]``: drop pad rows using host lengths (no sync)."""
    B, P = x.shape[0], x.shape[1]
    idx = _ranges_concat([i * P for i in range(B)], lengths, x.device)
    return x.reshape(B * P, *x.shape[2:]).index_select(0, idx)


def _collapse_ids_flat(flat_ids: torch.Tensor, seg_lengths: list[int], blank_id: int):
    """CTC collapse (unique_consecutive -> drop blank) on a flat argmax sequence with per-segment
    boundary force-keep. Bit-exact per segment vs the per-sample loop; costs 2 host syncs for the
    WHOLE batch (one data-dependent nonzero + one tiny index D2H).

    Returns ``(per_segment_tensors, per_segment_counts, kept_flat)``.
    """
    device = flat_ids.device
    lens = [int(l) for l in seg_lengths]
    N = sum(lens)
    if N == 0:
        empty = flat_ids[:0]
        return [empty for _ in lens], [0] * len(lens), empty
    offsets = _excl_cumsum(lens) + [N]
    change = torch.zeros(N, dtype=torch.bool, device=device)
    if N > 1:
        change[1:] = flat_ids[1:] != flat_ids[:-1]
    starts = [o for o, l in zip(offsets[:-1], lens) if l > 0]
    change[_as_idx(starts, device)] = True
    keep = change & (flat_ids != blank_id)
    kpos = keep.nonzero(as_tuple=False).flatten()            # sync (data-dependent count)
    kept = flat_ids.index_select(0, kpos)
    kpos_host = kpos.cpu()                                   # sync (tiny D2H)
    bounds = torch.searchsorted(kpos_host, torch.as_tensor(offsets, dtype=torch.long))
    counts = (bounds[1:] - bounds[:-1]).tolist()
    return list(kept.split(counts)), counts, kept


def _preds_to_host(preds: list[torch.Tensor]) -> list[list[int]]:
    """Materialize all per-sample pred tensors with ONE batched D2H copy."""
    if not preds:
        return []
    lens = [int(p.shape[0]) for p in preds]
    flat = torch.cat(preds) if len(preds) > 1 else preds[0]
    flat_host = flat.cpu()                                   # ONE sync for the whole batch
    return [t.tolist() for t in flat_host.split(lens)]


class GraniteSpeechNarForASR(nn.Module):
    def __init__(self, config: GraniteSpeechNarConfig):
        super().__init__()
        self.config = config
        self.encoder = CTCEncoder(config.encoder_config)
        self.projector = Projector(config.projector_config)
        self.language_model = GraniteLM(config.text_config)

    # ----- discrete helpers ------------------------------------------------- #
    def _ctc_collapse_decode(self, bpe_logits: torch.Tensor, bpe_lengths: list[int]) -> list[torch.Tensor]:
        """argmax -> unique_consecutive -> drop blank, per sample (flat batched, bit-exact).

        Accepts packed flat ``[N, V]`` logits or dense padded ``[B, P, V]`` (P3.1 encoder)."""
        blank_id = self.config.blank_token_id
        lens = [int(l) for l in bpe_lengths]
        if bpe_logits.dim() == 3:
            ids = _flatten_dense_rows(bpe_logits.argmax(dim=-1), lens)
        else:
            ids = bpe_logits.argmax(dim=-1)
        per, _counts, _kept = _collapse_ids_flat(ids, lens, blank_id)
        return per

    def _add_insertion_slots(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Interleave blank editing slots between tokens (blank, tok, blank, tok, ..., blank)."""
        blank_id = self.config.blank_token_id
        n = token_ids.numel()
        total_len = max(2 * n + 1, self.config.min_edit_sequence_length)
        idx = torch.arange(n, device=token_ids.device)
        out_idx = 2 * idx + 1
        out = torch.full((total_len,), fill_value=blank_id, dtype=token_ids.dtype, device=token_ids.device)
        out[out_idx] = token_ids
        return out

    def _build_flat_inputs(self, ctc_token_ids, audio_embeds, audio_lengths):
        """Pack [audio_i ; text_i] for all samples into one length dimension.

        Reference per-sample implementation, kept for the golden/profile scripts; the inference
        paths use the batched :meth:`_pack_editor_segments` (identical outputs)."""
        embed_tokens = self.language_model.model.embed_tokens
        embeds_list, position_ids_list, text_lengths = [], [], []
        for i, audio_len in enumerate(audio_lengths):
            audio_emb = audio_embeds[i, :audio_len]
            text_ids = self._add_insertion_slots(ctc_token_ids[i])
            text_emb = embed_tokens(text_ids)
            sample_embeds = torch.cat([audio_emb, text_emb], dim=0)
            embeds_list.append(sample_embeds)
            position_ids_list.append(torch.arange(sample_embeds.shape[0], device=audio_embeds.device))
            text_lengths.append(text_ids.shape[0])
        flat_embeds = torch.cat(embeds_list, dim=0).unsqueeze(0)
        flat_position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)
        return flat_embeds, flat_position_ids, text_lengths

    def _pack_editor_segments(self, audio_embeds, seg_audio, seg_tokens, seq_grid: int = 0):
        """Batched packed-[audio;text] builder (P2.3).

        ``seg_audio[k] = (row, a0, a1)`` selects ``audio_embeds[row, a0:a1]``; ``seg_tokens[k]``
        is the segment's (un-interleaved) token ids. ONE embedding gather + two index_copy
        scatters + arithmetic position_ids replace the per-segment python loop (~6 kernels vs
        ~6 per segment). Bit-exact vs ``adaptive.editor_pack.build_packed_segments`` /
        :meth:`_build_flat_inputs`, including the ``seq_grid`` isolated zero filler segment.

        Returns ``(flat_embeds[1,L,D], flat_position_ids[1,L], layout=[(a_len,t_len), ...])``.
        """
        cfg = self.config
        blank_id = cfg.blank_token_id
        min_len = cfg.min_edit_sequence_length
        embed_tokens = self.language_model.model.embed_tokens
        device = audio_embeds.device
        A = audio_embeds.shape[1]
        D = audio_embeds.shape[-1]

        a_lens = [int(a1 - a0) for (_r, a0, a1) in seg_audio]
        n_toks = [int(t.shape[0]) for t in seg_tokens]
        t_lens = [max(2 * n + 1, min_len) for n in n_toks]
        layout = list(zip(a_lens, t_lens))
        seg_lens = [x for at in layout for x in at]
        L = sum(seg_lens)
        L_pad = L
        if seq_grid and L % seq_grid:
            L_pad = L + (seq_grid - L % seq_grid)

        seg_starts = _excl_cumsum(seg_lens)
        audio_starts = seg_starts[0::2]
        text_starts = seg_starts[1::2]

        # interleaved text ids: blank-filled flat tensor, tokens scattered at odd local slots
        T_total = sum(t_lens)
        K_total = sum(n_toks)
        text_ids = torch.full((T_total,), blank_id, dtype=torch.long, device=device)
        if K_total:
            tok_flat = torch.cat(seg_tokens) if len(seg_tokens) > 1 else seg_tokens[0]
            nt = _as_idx(n_toks, device)
            tstart_per_tok = torch.repeat_interleave(_as_idx(text_starts_local := _excl_cumsum(t_lens), device),
                                                     nt, output_size=K_total)
            tokbase_per_tok = torch.repeat_interleave(_as_idx(_excl_cumsum(n_toks), device),
                                                      nt, output_size=K_total)
            j = torch.arange(K_total, dtype=torch.long, device=device) - tokbase_per_tok
            text_ids[tstart_per_tok + 2 * j + 1] = tok_flat.to(torch.long)
        text_embeds = embed_tokens(text_ids)

        flat = audio_embeds.new_zeros(L_pad, D)   # untouched rows = the seq_grid zero filler
        A_total = sum(a_lens)
        if A_total:
            dst_a = _ranges_concat(audio_starts, a_lens, device, A_total)
            src_a = _ranges_concat([r * A + a0 for (r, a0, _a1) in seg_audio], a_lens, device, A_total)
            flat.index_copy_(0, dst_a, audio_embeds.reshape(-1, D).index_select(0, src_a))
        if T_total:
            dst_t = _ranges_concat(text_starts, t_lens, device, T_total)
            flat.index_copy_(0, dst_t, text_embeds)

        # position_ids restart at 0 at each PACKED SEGMENT (= one audio+text pair — the LLM's
        # pos==0 boundary walls off samples, audio and its text must share a segment); the
        # trailing seq_grid filler restarts at 0 as its own isolated segment.
        pair_lens = [a + t for (a, t) in layout] + ([L_pad - L] if L_pad > L else [])
        pair_starts = _excl_cumsum(pair_lens)
        start_per_row = torch.repeat_interleave(_as_idx(pair_starts, device), _as_idx(pair_lens, device),
                                                output_size=L_pad)
        flat_pos = torch.arange(L_pad, dtype=torch.long, device=device) - start_per_row
        return flat.unsqueeze(0), flat_pos.unsqueeze(0), layout

    def _host_lengths(self, input_features, attention_mask, encoder_lengths) -> list[int]:
        """Per-sample encoder frame counts as host ints (P2.3). Prefers the feature extractor's
        ``encoder_lengths`` (no sync); falls back to one ``attention_mask.sum().tolist()``."""
        if encoder_lengths is not None:
            return [int(l) for l in encoder_lengths]
        if attention_mask is None:
            return [int(input_features.shape[1])] * int(input_features.shape[0])
        return [int(x) for x in attention_mask.sum(dim=1).tolist()]

    def _project_audio(self, hidden_states_list):
        """Projector + reference scaling/cast. ``hidden_states_list`` is the list of encoder
        hidden states (the projector cats them post-norm — skips one (B,T,4096) copy)."""
        cfg = self.config
        audio_embeds = self.projector(hidden_states_list)
        if cfg.scale_projected_embeddings:
            audio_embeds = audio_embeds / cfg.text_config.embedding_multiplier
        return audio_embeds.to(self.language_model.model.embed_tokens.weight.dtype)

    # ----- forward ---------------------------------------------------------- #
    @torch.inference_mode()
    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None,
                output_encoder_logits: bool = False, text_only_head: bool = False,
                encoder_lengths: list[int] | None = None) -> ASROutput:
        cfg = self.config
        enc_out = self.encoder(input_features, attention_mask=attention_mask, output_hidden_states=True,
                               hidden_state_indices=cfg.encoder_layer_indices)

        enc_lens = self._host_lengths(input_features, attention_mask, encoder_lengths)
        pool_window = self.encoder.config.bpe_pooling_window
        bpe_lengths = [-(-l // pool_window) for l in enc_lens]
        ctc_token_ids = self._ctc_collapse_decode(enc_out.logits, bpe_lengths)

        encoder_logits = enc_out.logits if output_encoder_logits else None

        audio_embeds = self._project_audio(
            [enc_out.all_hidden_states[idx] for idx in cfg.encoder_layer_indices]
        )

        downsample_rate = self.projector.config.downsample_rate
        audio_lengths = [l // downsample_rate for l in enc_lens]

        seg_audio = [(i, 0, audio_lengths[i]) for i in range(len(audio_lengths))]
        flat_embeds, flat_position_ids, layout = self._pack_editor_segments(
            audio_embeds, seg_audio, ctc_token_ids
        )
        text_lengths = [t for (_a, t) in layout]
        segment_lengths = [l for at in layout for l in at]

        editor_mask = attn_backends.build_editor_block_mask_from_lengths(
            segment_lengths, flat_embeds.device, flat_embeds.shape[1]
        )
        if text_only_head:
            # Fast path: run the tied LM head ONLY on the text segments, skipping the
            # full-vocab GEMM over the (discarded) audio rows. Argmax/transcript-exact vs
            # the default path; intermediate `text_logits` are bit-identical on the single
            # sample, and argmax-identical (small GEMM-shape numeric drift) on packed batches.
            hidden = self.language_model.model(
                inputs_embeds=flat_embeds, position_ids=flat_position_ids, attention_mask=editor_mask
            ).squeeze(0)
            text_hidden = torch.cat(list(hidden.split(segment_lengths)[1::2]))
            text_logits = F.linear(
                text_hidden, self.language_model.model.embed_tokens.weight
            ) / self.language_model.logits_scaling
        else:
            llm_out = self.language_model(
                inputs_embeds=flat_embeds, position_ids=flat_position_ids, attention_mask=editor_mask
            )
            all_logits = llm_out.logits.squeeze(0)
            text_logits = torch.cat(list(all_logits.split(segment_lengths)[1::2]))
        logits_per_sample = list(text_logits.split(text_lengths))

        return ASROutput(
            logits=logits_per_sample,
            encoder_logits=encoder_logits,
            encoder_preds=ctc_token_ids,
        )

    @torch.inference_mode()
    def transcribe(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None,
                   output_encoder_logits: bool = False, text_only_head: bool = False,
                   encoder_lengths: list[int] | None = None) -> ASROutput:
        output = self.forward(input_features, attention_mask, output_encoder_logits, text_only_head,
                              encoder_lengths=encoder_lengths)
        blank_id = self.config.blank_token_id
        # vectorized final collapse (P2.2): per-sample argmax launches, ONE flat collapse
        text_lengths = [int(l.shape[0]) for l in output.logits]
        flat_ids = torch.cat([l.argmax(-1) for l in output.logits])
        preds, counts, _kept = _collapse_ids_flat(flat_ids, text_lengths, blank_id)
        preds_host = _preds_to_host(preds)
        return ASROutput(
            preds=preds,
            preds_host=preds_host,
            logits=output.logits,
            encoder_logits=output.encoder_logits,
            encoder_preds=output.encoder_preds,
        )

    # ----- adaptive path (OPTIMIZATION_PLAN Phases 2-4) --------------------- #
    def _editor_text_hidden(self, audio_embeds, seg_audio, seg_tokens,
                            early_exit=None, seq_grid: int = 0):
        """Texthead-style editor pass WITHOUT the tied LM head: pack segments, run the base
        LLM, return ``(flat_text_hidden, text_lens)``. Callers apply the head (full logits)
        or the fused argmax (Exp2)."""
        from .adaptive.editor_pack import text_segment_lengths, text_position_mask
        flat_embeds, flat_pos, layout = self._pack_editor_segments(
            audio_embeds, seg_audio, seg_tokens, seq_grid=seq_grid
        )
        seg_lengths = text_segment_lengths(layout, flat_embeds.shape[1])
        editor_mask = attn_backends.build_editor_block_mask_from_lengths(
            seg_lengths, flat_embeds.device, flat_embeds.shape[1]
        )
        # Phase-5 early-exit must judge stability on TEXT positions only (not audio / filler rows).
        tpm = None
        if early_exit is not None and getattr(early_exit, "enabled", False):
            tpm = text_position_mask(layout, flat_embeds.shape[1], device=flat_embeds.device)
        text_lens = [t for (_a, t) in layout]
        hidden = self.language_model.model(
            inputs_embeds=flat_embeds, position_ids=flat_pos, attention_mask=editor_mask,
            early_exit=early_exit, text_position_mask=tpm).squeeze(0)
        text_hidden = torch.cat(list(hidden.split(seg_lengths)[1::2]))
        return text_hidden, text_lens

    def _editor_text_logits(self, audio_embeds, seg_audio, seg_tokens, text_only_head: bool,
                            early_exit=None, seq_grid: int = 0):
        """Run the LLM editor on packed segments; returns ``(flat_text_logits, text_lens, scaled)``.

        ``scaled=False`` (texthead path) means the ``/logits_scaling`` division is SKIPPED —
        argmax/collapse are invariant to it (exact power-of-two positive scale); divide before
        any softmax/confidence use (P2.4g). The non-texthead path returns scaled logits."""
        if text_only_head:
            text_hidden, text_lens = self._editor_text_hidden(
                audio_embeds, seg_audio, seg_tokens, early_exit=early_exit, seq_grid=seq_grid)
            text_logits = F.linear(text_hidden, self.language_model.model.embed_tokens.weight)
            return text_logits, text_lens, False
        from .adaptive.editor_pack import text_segment_lengths, text_position_mask
        flat_embeds, flat_pos, layout = self._pack_editor_segments(
            audio_embeds, seg_audio, seg_tokens, seq_grid=seq_grid
        )
        seg_lengths = text_segment_lengths(layout, flat_embeds.shape[1])
        editor_mask = attn_backends.build_editor_block_mask_from_lengths(
            seg_lengths, flat_embeds.device, flat_embeds.shape[1]
        )
        tpm = None
        if early_exit is not None and getattr(early_exit, "enabled", False):
            tpm = text_position_mask(layout, flat_embeds.shape[1], device=flat_embeds.device)
        text_lens = [t for (_a, t) in layout]
        all_logits = self.language_model(
            inputs_embeds=flat_embeds, position_ids=flat_pos, attention_mask=editor_mask,
            early_exit=early_exit, text_position_mask=tpm).logits.squeeze(0)
        text_logits = torch.cat(list(all_logits.split(seg_lengths)[1::2]))
        return text_logits, text_lens, True

    def _collapse_logits(self, text_logits: torch.Tensor) -> torch.Tensor:
        blank_id = self.config.blank_token_id
        collapsed = torch.unique_consecutive(text_logits.argmax(-1))
        return collapsed[collapsed != blank_id]

    @torch.inference_mode()
    def transcribe_adaptive(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None,
                            routing=None, verifier=None, text_only_head: bool = False,
                            early_exit=None, seq_grid: int = 0,
                            encoder_lengths: list[int] | None = None) -> ASROutput:
        """CTC-first adaptive inference.

        Per utterance: compute CTC confidence, then route to
          ctc_fast : return the CTC hypothesis (skip the editor)  -- the big latency win
          local    : edit only suspicious windows (if routing.local_crop_audio) + verifier + fallback
          full     : run the full NAR editor (the baseline path)
        With ``routing.enabled=False`` (default) this is EXACTLY ``transcribe`` (lossless).

        P2.1: the collapsed hypothesis is taken from the confidence bundle (the previous separate
        full-vocab CTC argmax pass was a duplicate — unit-tested identical); span detection and
        the confidence scalars sync once per batch, not per sample.
        """
        from .adaptive import (compute_ctc_confidence, VerifierConfig, Route, route_decision,
                               detect_suspicious_spans_batch, window_spans, span_audio_frame_range,
                               merge_local_edits, verifier_accepts)
        from .adaptive.verifier import Candidate
        from collections import defaultdict

        cfg = self.config
        if routing is None or not getattr(routing, "enabled", False):
            return self.transcribe(input_features, attention_mask, text_only_head=text_only_head,
                                   encoder_lengths=encoder_lengths)
        if verifier is None:
            verifier = VerifierConfig()

        enc_out = self.encoder(input_features, attention_mask=attention_mask, output_hidden_states=True,
                               hidden_state_indices=cfg.encoder_layer_indices)
        if enc_out.logits is None:                       # no BPE CTC head -> cannot route, be safe
            return self.transcribe(input_features, attention_mask, text_only_head=text_only_head,
                                   encoder_lengths=encoder_lengths)
        enc_lens = self._host_lengths(input_features, attention_mask, encoder_lengths)
        pool_window = self.encoder.config.bpe_pooling_window
        bpe_lengths = [-(-l // pool_window) for l in enc_lens]

        confs = compute_ctc_confidence(enc_out.logits, bpe_lengths, cfg.blank_token_id)
        ctc_token_ids = [c.token_ids for c in confs]     # == _ctc_collapse_decode (invariant, unit-tested)
        spans_per = detect_suspicious_spans_batch(confs, routing)
        routes = [route_decision(c, sp, routing) for c, sp in zip(confs, spans_per)]

        B = len(ctc_token_ids)
        preds: list[torch.Tensor | None] = [None] * B
        for i in range(B):
            if routes[i] == Route.CTC_FAST:
                preds[i] = ctc_token_ids[i]

        need_editor = [i for i in range(B) if routes[i] != Route.CTC_FAST]
        if need_editor:
            hs = [enc_out.all_hidden_states[idx] for idx in cfg.encoder_layer_indices]
            if attn_backends.projector_slice_enabled() and len(need_editor) < B:
                # P2.3 (ulp-class, env-gated): projector runs only on the rows that reach the
                # editor; the ctc_fast rows' multilayer features are never projected.
                rows = _as_idx(need_editor, hs[0].device)
                hs = [h.index_select(0, rows) for h in hs]
                row_of = {i: r for r, i in enumerate(need_editor)}
            else:
                row_of = {i: i for i in need_editor}
            audio_embeds = self._project_audio(hs)
            downsample = self.projector.config.downsample_rate
            audio_lengths = [l // downsample for l in enc_lens]

            seg_audio, seg_tokens, seg_owner, windows_by_sample = [], [], [], {}
            for i in need_editor:
                a_len = audio_lengths[i]
                use_local = (routes[i] == Route.LOCAL and routing.local_crop_audio
                             and spans_per[i] and confs[i].num_tokens > 0)
                if use_local:
                    wins = window_spans(spans_per[i], confs[i].num_tokens,
                                        routing.left_context_tokens, routing.right_context_tokens)
                    windows_by_sample[i] = wins
                    for (ws, we) in wins:
                        fs = int(confs[i].token_frame_start[ws]); fe = int(confs[i].token_frame_end[we - 1])
                        a0, a1 = span_audio_frame_range(fs, fe, pool_window, downsample, a_len,
                                                        routing.audio_crop_margin)
                        seg_audio.append((row_of[i], a0, a1))
                        seg_tokens.append(ctc_token_ids[i][ws:we])
                        seg_owner.append((i, "local"))
                else:
                    seg_audio.append((row_of[i], 0, a_len))
                    seg_tokens.append(ctc_token_ids[i])
                    seg_owner.append((i, "full"))

            fused_head = text_only_head and attn_backends.texthead_argmax_enabled()
            if fused_head:
                # Exp2: chunked GEMM + running rowwise argmax — the [T_text, V] logits tensor
                # never hits HBM. Confidence for LOCAL segments is recomputed on their (small)
                # hidden slices below.
                from cuda_kernels.texthead_argmax import linear_argmax
                text_hidden, text_lens = self._editor_text_hidden(
                    audio_embeds, seg_audio, seg_tokens, early_exit=early_exit, seq_grid=seq_grid)
                head_w = self.language_model.model.embed_tokens.weight
                flat_ids = linear_argmax(text_hidden, head_w)
            else:
                text_logits, text_lens, scaled = self._editor_text_logits(
                    audio_embeds, seg_audio, seg_tokens, text_only_head,
                    early_exit=early_exit, seq_grid=seq_grid)
                flat_ids = text_logits.argmax(-1)
            seg_tok_list, _counts, _kept = _collapse_ids_flat(
                flat_ids, text_lens, cfg.blank_token_id)

            # per-segment confidence is only consumed for LOCAL segments (verifier); it needs the
            # true scaled distribution, so divide lazily when the texthead path skipped it.
            seg_conf = [1.0] * len(seg_owner)
            if any(kind == "local" for (_i, kind) in seg_owner):
                sc = self.language_model.logits_scaling
                if fused_head:
                    th_views = text_hidden.split(text_lens)
                    for k, (i, kind) in enumerate(seg_owner):
                        if kind == "local" and th_views[k].numel():
                            tl = F.linear(th_views[k], head_w).float() / sc
                            seg_conf[k] = float(torch.softmax(tl, -1).max(-1).values.mean())
                else:
                    tl_views = text_logits.split(text_lens)
                    for k, (i, kind) in enumerate(seg_owner):
                        if kind == "local" and tl_views[k].numel():
                            tl = tl_views[k].float()
                            if not scaled:
                                tl = tl / sc
                            seg_conf[k] = float(torch.softmax(tl, -1).max(-1).values.mean())

            local_items = defaultdict(list)
            for k, (i, kind) in enumerate(seg_owner):
                if kind == "full":
                    preds[i] = seg_tok_list[k]
                else:
                    local_items[i].append((seg_tok_list[k], seg_conf[k]))

            fallback = []
            for i, items in local_items.items():
                edited = [t for (t, _) in items]
                merged = merge_local_edits(ctc_token_ids[i], windows_by_sample[i], edited)
                mean_conf = sum(c for (_, c) in items) / max(1, len(items))
                cand = Candidate(tokens=merged, ctc_tokens=ctc_token_ids[i], mean_confidence=mean_conf)
                if verifier_accepts(cand, verifier):
                    preds[i] = merged
                elif routing.fallback_to_full_editor:
                    fallback.append(i)
                else:
                    preds[i] = merged

            if fallback:
                fb_audio = [(row_of[i], 0, audio_lengths[i]) for i in fallback]
                fb_tokens = [ctc_token_ids[i] for i in fallback]
                if fused_head:
                    fb_hidden, fb_lens = self._editor_text_hidden(
                        audio_embeds, fb_audio, fb_tokens, early_exit=early_exit, seq_grid=seq_grid)
                    fb_ids = linear_argmax(fb_hidden, head_w)
                else:
                    fb_logits, fb_lens, _sc = self._editor_text_logits(
                        audio_embeds, fb_audio, fb_tokens, text_only_head,
                        early_exit=early_exit, seq_grid=seq_grid)
                    fb_ids = fb_logits.argmax(-1)
                fb_toks, _c, _k = _collapse_ids_flat(fb_ids, fb_lens, cfg.blank_token_id)
                for i, t in zip(fallback, fb_toks):
                    preds[i] = t

        preds_host = _preds_to_host(preds) if all(p is not None for p in preds) else None
        return ASROutput(preds=preds, preds_host=preds_host, encoder_preds=ctc_token_ids, routes=routes)
