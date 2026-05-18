# -*- coding: utf-8 -*-
"""VariantD — Qwen-2.5-1.5B (LoRA) + SigLIP-Base vision + action chunk head.

This is the M2 milestone model: replace the ResNet18+MLP baseline with a real
VLA-style backbone. We follow LLaVA-style minimal fusion — SigLIP gives
(B, 196, 768) patch tokens, a single linear projects them to Qwen's 1536-d
hidden space, optional proprio is appended as one extra token. Qwen forward
proceeds with causal attention; the last token's hidden state is the action
"summary" fed into a linear chunk-action head.

LoRA touches q/k/v/o_proj across all Qwen layers with r=64 → ~5M trainable.
The full 1.5B Qwen weights stay frozen + bf16. SigLIP also frozen.

Two flavours sharing the same backbone:
  VariantD_RGB        — vision-only (RGB → SigLIP → Qwen → action)
  VariantD_RGBProprio — RGB + 15-d proprio appended as a token

The RGB-only flavour is the "stronger A baseline" for M2; the proprio
flavour answers the same question Gate-1 mini did, now on a real VLA.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize


_DEFAULT_QWEN = "/root/autodl-tmp/hf/qwen2.5-1.5b"
_DEFAULT_SIGLIP = "/root/autodl-tmp/hf/siglip-base"
_SIGLIP_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_STD = [0.5, 0.5, 0.5]


def _load_qwen_with_lora(qwen_path: str, r: int, alpha: int, dropout: float):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    cfg = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    qwen = get_peft_model(qwen, cfg)
    return qwen


def _load_siglip(siglip_path: str):
    from transformers import SiglipVisionModel
    vis = SiglipVisionModel.from_pretrained(siglip_path, torch_dtype=torch.bfloat16)
    for p in vis.parameters():
        p.requires_grad = False
    vis.eval()
    return vis


class _QwenVLABase(nn.Module):
    def __init__(self,
                 action_dim: int = 7,
                 chunk_size: int = 8,
                 with_proprio: bool = True,
                 proprio_dim: int = 15,
                 lora_r: int = 64,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 qwen_path: str = _DEFAULT_QWEN,
                 siglip_path: str = _DEFAULT_SIGLIP):
        super().__init__()
        if not Path(qwen_path).exists():
            raise FileNotFoundError(f"Qwen model not found at {qwen_path} — "
                                    f"run the download step in scripts/m2/setup.sh first")
        if not Path(siglip_path).exists():
            raise FileNotFoundError(f"SigLIP model not found at {siglip_path}")

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.with_proprio = with_proprio

        # Vision
        self.vision = _load_siglip(siglip_path)
        vis_dim = self.vision.config.hidden_size  # 768

        # Qwen w/ LoRA
        self.qwen = _load_qwen_with_lora(qwen_path, lora_r, lora_alpha, lora_dropout)
        qwen_dim = self.qwen.config.hidden_size   # 1536

        # Projections (kept fp32 weights but matmul in bf16 via input cast — safer init)
        self.vis_proj = nn.Linear(vis_dim, qwen_dim)
        if with_proprio:
            self.proprio_proj = nn.Linear(proprio_dim, qwen_dim)
        self.head = nn.Linear(qwen_dim, action_dim * chunk_size)

        # Convert these small adapter layers to bf16 to match Qwen
        self.vis_proj = self.vis_proj.to(torch.bfloat16)
        if with_proprio:
            self.proprio_proj = self.proprio_proj.to(torch.bfloat16)
        self.head = self.head.to(torch.bfloat16)

        self.preprocess = Compose([
            Resize((224, 224), antialias=True),
            Normalize(_SIGLIP_MEAN, _SIGLIP_STD),
        ])

    def trainable_parameters(self):
        # peft-wrapped Qwen + small adapter layers
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # image: (B,3,H,W) float in [0,1]
        x = self.preprocess(image).to(torch.bfloat16)
        with torch.no_grad():
            vis_out = self.vision(pixel_values=x, output_hidden_states=False)
        vis_feats = vis_out.last_hidden_state                    # (B, 196, 768) bf16
        vis_tokens = self.vis_proj(vis_feats)                    # (B, 196, 1536)

        if self.with_proprio:
            prop = proprio.to(torch.bfloat16)
            prop_tok = self.proprio_proj(prop).unsqueeze(1)      # (B, 1, 1536)
            input_embeds = torch.cat([vis_tokens, prop_tok], dim=1)
        else:
            input_embeds = vis_tokens

        # Qwen forward (causal mask applied internally)
        out = self.qwen(inputs_embeds=input_embeds,
                        output_hidden_states=True,
                        use_cache=False)
        last_hidden = out.hidden_states[-1][:, -1, :]            # (B, 1536) bf16

        action = self.head(last_hidden)                          # (B, A*K) bf16
        return action.view(-1, self.chunk_size, self.action_dim).float()


class VariantD_RGB(_QwenVLABase):
    """RGB-only via Qwen-2.5-1.5B + SigLIP + LoRA. The M2 baseline."""
    def __init__(self, **kwargs):
        super().__init__(with_proprio=False, **kwargs)

    def forward(self, image, proprio=None):  # proprio ignored
        return super().forward(image, torch.zeros(1, device=image.device))


class VariantD_RGBProprio(_QwenVLABase):
    """RGB + 15-d proprio via Qwen-2.5-1.5B + SigLIP + LoRA. M2 H1 test."""
    def __init__(self, **kwargs):
        super().__init__(with_proprio=True, **kwargs)


def trainable_param_count_d(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
