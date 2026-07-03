"""Verifier for the two-tier cascade (OPTIMIZATION_PLAN Phase 4 "Verifier Logic").

After the cheap tier (local span editor) produces a candidate transcript, the verifier decides
whether to accept it or escalate to the full NAR editor. It is a *safety gate*: it should reject
anything that looks like a confident-wrong edit, because a bad accept is worse than spending the
extra compute on the full editor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifierConfig:
    min_confidence: float = 0.88            # candidate mean confidence must be >=
    max_safe_edit_distance: int = 8         # token edit distance from the CTC hypothesis must be <=
    max_confident_wrong_score: float = 0.15
    reject_unstable_span: bool = True       # reject if any edited span is flagged unstable


def token_edit_distance(a, b) -> int:
    """Levenshtein distance between two token-id sequences (lists or 1-D tensors)."""
    a = a.tolist() if hasattr(a, "tolist") else list(a)
    b = b.tolist() if hasattr(b, "tolist") else list(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


@dataclass
class Candidate:
    """What the tier-1 (local) editor hands the verifier."""

    tokens: object                 # 1-D tensor / list of the candidate transcript token ids
    ctc_tokens: object             # the CTC hypothesis it edited (for edit-distance)
    mean_confidence: float = 1.0   # candidate mean token confidence (from the editor logits)
    has_unstable_span: bool = False
    confident_wrong_score: float = 0.0


def verifier_accepts(candidate: Candidate, cfg: VerifierConfig) -> bool:
    """True => accept the cheap candidate; False => escalate to the full editor."""
    if candidate.mean_confidence < cfg.min_confidence:
        return False
    if token_edit_distance(candidate.tokens, candidate.ctc_tokens) > cfg.max_safe_edit_distance:
        return False
    if cfg.reject_unstable_span and candidate.has_unstable_span:
        return False
    if candidate.confident_wrong_score > cfg.max_confident_wrong_score:
        return False
    return True
