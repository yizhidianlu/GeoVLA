# -*- coding: utf-8 -*-
"""GeoVLA tensor-shape interface contracts.

Locked at master synthesis (see docs/master_synthesis.md §3). All cross-module
tensors flowing between {Perception, VLM, FlowHead, WorldModel, GeoSpec, Action}
MUST conform to these dataclass-defined shapes. Violations raise at construction
time, before any GPU compute is wasted.

These are *contracts*, not data containers — they validate shape/dtype and
optionally device, then expose the underlying tensor.

See:
    .paic/specs/00_master_synthesis.md  §3 Interface Contracts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ──────────────────────────────────────────────────────────────────────────────
#  Architectural constants (locked at master synthesis)
# ──────────────────────────────────────────────────────────────────────────────

#: Number of multi-view cameras (wrist + 3rd-person + tactile-RGB)
N_CAMERAS = 3

#: 3DGS dense reconstruction — max Gaussians produced by perception front-end
N_DENSE_MAX = 80_000

#: Number of Gaussian control particles after FPS downsampling
N_CONTROL = 2048

#: Number of geo-tokens after spatial-bucket pool (4x4x4 grid → 64; we pool
#: into 256 = 4× redundancy for resolution headroom)
N_GEO_TOKENS = 256

#: VLM hidden dimension (Qwen-2.5-1.5B = 1536)
D_HIDDEN = 1536

#: Action chunk horizon (8 actions emitted per inference; coordinates with MoH)
H_ACTION = 8

#: Action dimension (7 = 6-DoF pose delta + 1 gripper)
D_ACTION = 7

#: Number of flow-matching proposals per step (top-K re-ranking)
K_PROPOSALS = 4

#: Per-Gaussian feature dim: 3 pos + 4 rot quat + 3 scale + 3 color + 1 opacity
D_GAUSSIAN = 14


# ──────────────────────────────────────────────────────────────────────────────
#  Interface dataclasses (validated on construction)
# ──────────────────────────────────────────────────────────────────────────────

def _check(t: torch.Tensor, name: str, shape: tuple, dtype: torch.dtype | None = None):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(t).__name__}")
    if t.shape != shape:
        raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {shape}")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name}: dtype {t.dtype} != expected {dtype}")


@dataclass
class I1_DenseGaussians:
    """3DGS front-end → 3D tokeniser.

    Per-frame dense Gaussians produced by the feed-forward reconstruction.
    Variable count up to N_DENSE_MAX; we pass actual count via .n_active.
    """
    tensor: torch.Tensor      # (N_DENSE_MAX, D_GAUSSIAN), FP16, zero-padded
    n_active: int             # how many primitives are real (rest are padding)

    def __post_init__(self):
        _check(self.tensor, "I1_DenseGaussians.tensor",
               (N_DENSE_MAX, D_GAUSSIAN), torch.float16)
        if not (0 < self.n_active <= N_DENSE_MAX):
            raise ValueError(f"I1.n_active {self.n_active} out of (0, {N_DENSE_MAX}]")


@dataclass
class I2_GeoTokens:
    """3D tokeniser → VLM.

    Spatial-bucket-pooled geo-tokens, projected to VLM hidden dim.
    """
    tokens: torch.Tensor      # (1, N_GEO_TOKENS, D_HIDDEN), BF16

    def __post_init__(self):
        _check(self.tokens, "I2_GeoTokens.tokens",
               (1, N_GEO_TOKENS, D_HIDDEN), torch.bfloat16)


@dataclass
class I3_VLMHidden:
    """VLM hidden → Flow head.

    Hidden state at the [ACT] anchor token; emitted at end of VLM forward.
    T_vlm includes geo-tokens + image tokens + language tokens + [ACT].
    """
    hidden: torch.Tensor      # (1, T_vlm, D_HIDDEN), BF16
    act_anchor_idx: int       # index of the [ACT] token in T_vlm

    def __post_init__(self):
        if self.hidden.dim() != 3:
            raise ValueError(f"I3.hidden must be 3D, got shape {tuple(self.hidden.shape)}")
        if self.hidden.shape[2] != D_HIDDEN:
            raise ValueError(f"I3.hidden last dim {self.hidden.shape[2]} != {D_HIDDEN}")
        if self.hidden.dtype != torch.bfloat16:
            raise ValueError(f"I3.hidden dtype {self.hidden.dtype} != bfloat16")
        if not (0 <= self.act_anchor_idx < self.hidden.shape[1]):
            raise ValueError("I3.act_anchor_idx out of bounds")


@dataclass
class I4_FlowProposals:
    """Flow head → World model.

    K top-K candidate action chunks (sampled with different temperatures).
    """
    proposals: torch.Tensor   # (K_PROPOSALS, H_ACTION, D_ACTION), FP16
    log_probs: torch.Tensor   # (K_PROPOSALS,) FP32 — for kinematic Kalman score

    def __post_init__(self):
        _check(self.proposals, "I4_FlowProposals.proposals",
               (K_PROPOSALS, H_ACTION, D_ACTION), torch.float16)
        _check(self.log_probs, "I4_FlowProposals.log_probs",
               (K_PROPOSALS,), torch.float32)


@dataclass
class I5_PredictedGaussians:
    """World model → GeoSpec.

    Predicted next-frame Gaussians for each of K proposals (parallel batch).
    REUSED by both L3 re-ranking AND GeoSpec L_geo computation.
    """
    next_gaussians: torch.Tensor   # (K_PROPOSALS, N_CONTROL, D_GAUSSIAN), FP16
    rollout_confidence: torch.Tensor  # (K_PROPOSALS,) FP32 — WM's own confidence

    def __post_init__(self):
        _check(self.next_gaussians, "I5_PredictedGaussians.next_gaussians",
               (K_PROPOSALS, N_CONTROL, D_GAUSSIAN), torch.float16)
        _check(self.rollout_confidence, "I5_PredictedGaussians.rollout_confidence",
               (K_PROPOSALS,), torch.float32)


@dataclass
class I6_CachedFeatures:
    """World model → next-step 3DGS-tokeniser (cache for next frame)."""
    cached_tokens: torch.Tensor    # (1, N_GEO_TOKENS, D_HIDDEN), BF16

    def __post_init__(self):
        _check(self.cached_tokens, "I6_CachedFeatures.cached_tokens",
               (1, N_GEO_TOKENS, D_HIDDEN), torch.bfloat16)


@dataclass
class I7_EmittedAction:
    """GeoSpec → action emitter (motor)."""
    action_chunk: torch.Tensor     # (H_ACTION, D_ACTION), FP16
    chosen_idx: int                # which of K_PROPOSALS was selected
    accepted: bool                 # GeoSpec gate verdict — False triggers fallback
    gate_score: float              # composite score from GeoSpec α∙s

    def __post_init__(self):
        _check(self.action_chunk, "I7_EmittedAction.action_chunk",
               (H_ACTION, D_ACTION), torch.float16)
        if not (0 <= self.chosen_idx < K_PROPOSALS):
            raise ValueError("I7.chosen_idx out of bounds")
        if not isinstance(self.accepted, bool):
            raise TypeError(f"I7.accepted must be bool, got {type(self.accepted).__name__}")


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience: a single canonical message-passing record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoVLAStep:
    """One inference step's full I/O record. Useful for tests + replay."""
    i1: Optional[I1_DenseGaussians] = None
    i2: Optional[I2_GeoTokens] = None
    i3: Optional[I3_VLMHidden] = None
    i4: Optional[I4_FlowProposals] = None
    i5: Optional[I5_PredictedGaussians] = None
    i6: Optional[I6_CachedFeatures] = None
    i7: Optional[I7_EmittedAction] = None

    latency_ms: dict[str, float] = field(default_factory=dict)
