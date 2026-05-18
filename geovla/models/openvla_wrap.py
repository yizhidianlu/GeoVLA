# -*- coding: utf-8 -*-
"""VariantE — OpenVLA-7B (Prismatic VLM) + LoRA + action chunk head.

OpenVLA-7B is a Prismatic VLM combining DINOv2 + SigLIP vision + LLaMA-2 7B
backbone, pretrained on 970K Open X-Embodiment trajectories. We use it as the
M2.5 "production-grade VLA" baseline, mirroring the VariantD setup but with a
7B pretrained-on-robot-actions backbone instead of a 1.5B text-only Qwen.

Architecture:
- OpenVLA-7B forward (vision tower + projector + LLaMA backbone) — frozen weights
- LoRA on LLaMA q/k/v/o_proj (r=64) — ~60M trainable
- Last hidden state → linear action chunk head
- Optional proprio token appended to the LLM input (E-RGBProprio)

Two flavours (mirror VariantD naming convention):
  VariantE_RGB         — RGB only (the "stronger A baseline" for M2.5)
  VariantE_RGBProprio  — RGB + 15-d proprio appended

We deliberately do NOT fine-tune OpenVLA's discrete action vocabulary (would
require re-formatting demos as action-token sequences). Instead we treat
OpenVLA as a frozen-with-LoRA vision-language backbone and learn a fresh
continuous action head — this preserves comparability with VariantD.

Implementation notes:
- `trust_remote_code=True` required for the prismatic_vla custom modeling.
- Bf16 throughout to fit 7B + LoRA in <20 GB on A800.
- A fixed text prompt is used at every forward to keep batched inference fast:
  "In: What action should the robot take?\\nOut:"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize


_DEFAULT_OPENVLA = "/root/autodl-tmp/hf/openvla-7b"
# Prismatic / OpenVLA visual preprocessing — equivalent to SigLIP + DINOv2 means
_OVLA_MEAN = [0.5, 0.5, 0.5]
_OVLA_STD = [0.5, 0.5, 0.5]
_OVLA_PROMPT = "In: What action should the robot take?\nOut:"


def _load_openvla_with_lora(path: str, r: int, alpha: int, dropout: float):
    from transformers import AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    vla = AutoModelForVision2Seq.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # freeze entire model first
    for p in vla.parameters():
        p.requires_grad = False
    # apply LoRA to LLaMA backbone q/k/v/o
    cfg = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # OpenVLA exposes the LLM as `vla.language_model` (PrismaticForActionPrediction wraps it)
    try:
        vla.language_model = get_peft_model(vla.language_model, cfg)
    except AttributeError:
        # Fallback: apply to top-level vla if naming differs
        vla = get_peft_model(vla, cfg)
    return vla


class _OpenVLABase(nn.Module):
    """Shared backbone wrapper. Subclasses set with_proprio."""

    def __init__(self,
                 action_dim: int = 7,
                 chunk_size: int = 8,
                 with_proprio: bool = True,
                 proprio_dim: int = 15,
                 lora_r: int = 64,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 openvla_path: str = _DEFAULT_OPENVLA):
        super().__init__()
        if not Path(openvla_path).exists():
            raise FileNotFoundError(
                f"OpenVLA weights not at {openvla_path}; run M2.5 download step first.")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.with_proprio = with_proprio
        self.openvla_path = openvla_path

        self.vla = _load_openvla_with_lora(openvla_path, lora_r, lora_alpha, lora_dropout)
        # Determine LLM hidden dim
        try:
            llm_hidden = self.vla.config.text_config.hidden_size
        except AttributeError:
            llm_hidden = self.vla.config.hidden_size
        self.llm_hidden = llm_hidden

        # Action head + optional proprio adapter (both bf16 to match backbone)
        if with_proprio:
            self.proprio_proj = nn.Linear(proprio_dim, llm_hidden).to(torch.bfloat16)
        self.head = nn.Linear(llm_hidden, action_dim * chunk_size).to(torch.bfloat16)

        # Tokenize the fixed prompt once
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
        prompt_ids = self.processor.tokenizer(
            _OVLA_PROMPT, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0]
        self.register_buffer("prompt_ids", prompt_ids, persistent=False)

        self.preprocess = Compose([
            Resize((224, 224), antialias=True),
            Normalize(_OVLA_MEAN, _OVLA_STD),
        ])

    def _forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        """Return last hidden state at the last token of the LLM, (B, llm_hidden)."""
        B = image.shape[0]
        x = self.preprocess(image).to(torch.bfloat16)
        # Expand the fixed prompt to batch
        input_ids = self.prompt_ids.unsqueeze(0).expand(B, -1).to(image.device)
        # forward
        with torch.no_grad():
            out = self.vla(
                input_ids=input_ids,
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            )
        hs = out.hidden_states[-1]                # (B, T, H)
        return hs[:, -1, :]                       # last token

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        h = self._forward_backbone(image)
        if self.with_proprio:
            p = self.proprio_proj(proprio.to(torch.bfloat16))
            h = h + p     # additive fusion (cheapest, no architectural risk)
        action = self.head(h)
        return action.view(-1, self.chunk_size, self.action_dim).float()


class VariantE_RGB(_OpenVLABase):
    """OpenVLA-7B + LoRA + linear action head. RGB only."""

    def __init__(self, **kwargs):
        super().__init__(with_proprio=False, **kwargs)

    def forward(self, image: torch.Tensor, proprio: Optional[torch.Tensor] = None):
        # proprio ignored
        h = self._forward_backbone(image)
        action = self.head(h)
        return action.view(-1, self.chunk_size, self.action_dim).float()


class VariantE_RGBProprio(_OpenVLABase):
    """OpenVLA-7B + LoRA + linear action head. RGB + 15-d proprio."""
    def __init__(self, **kwargs):
        super().__init__(with_proprio=True, **kwargs)


def trainable_param_count_e(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
