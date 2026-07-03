"""Early-exit controller for the NAR editor (OPTIMIZATION_PLAN Phase 5).

The bidirectional editor runs all 40 layers unconditionally. When the edited transcript has already
stabilized (the text-position argmax stops changing across layers), the remaining layers are wasted.
This module holds the *policy* (config + stability tracker); the actual per-layer projection + exit
lives behind a flag in ``llm.py`` (``GraniteModel.forward(early_exit=...)``), so the default path is
untouched.

Safety: only high-confidence samples may early-exit; the projection check is applied to the TEXT
positions only, and the exit requires N consecutive stable checks (``patience``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EarlyExitConfig:
    enabled: bool = False
    min_layers: int = 20        # never exit before this many layers have run
    check_every: int = 4        # project + compare every K layers after min_layers
    patience: int = 2           # require this many consecutive stable checks to exit
    conf_threshold: float = 0.90  # exiting also requires mean text-token prob >= this


def stable_argmax(prev_ids: torch.Tensor | None, cur_ids: torch.Tensor) -> bool:
    """True if the two argmax token sequences are identical (same length + equal)."""
    if prev_ids is None:
        return False
    if prev_ids.shape != cur_ids.shape:
        return False
    return bool(torch.equal(prev_ids, cur_ids))


class StabilityTracker:
    """Counts consecutive stable checks; signals exit once patience is reached."""

    def __init__(self, cfg: EarlyExitConfig):
        self.cfg = cfg
        self.prev: torch.Tensor | None = None
        self.stable_count = 0

    def update(self, cur_ids: torch.Tensor, mean_conf: float) -> bool:
        """Feed the current text-position argmax; return True to exit now."""
        if stable_argmax(self.prev, cur_ids) and mean_conf >= self.cfg.conf_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        self.prev = cur_ids
        return self.stable_count >= self.cfg.patience
