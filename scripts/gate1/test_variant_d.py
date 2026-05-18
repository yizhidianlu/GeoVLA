#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny forward sanity check for VariantD before kicking off M2 training.

Loads the model, runs one forward, prints trainable-param count + output shape.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from geovla.models.qwen_vla import VariantD_RGBProprio, trainable_param_count_d


def main():
    dev = torch.device("cuda")
    print("=== Building VariantD_RGBProprio (Qwen-2.5-1.5B + SigLIP-Base + LoRA r=64) ===")
    t0 = time.time()
    m = VariantD_RGBProprio(chunk_size=8).to(dev)
    print(f"  built in {time.time()-t0:.1f}s")
    print(f"  trainable params: {trainable_param_count_d(m):,}")
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    img = torch.randn(2, 3, 128, 128, device=dev)
    prop = torch.randn(2, 15, device=dev)
    t1 = time.time()
    with torch.no_grad():
        out = m(img, prop)
    torch.cuda.synchronize()
    print(f"  forward batch=2: out shape={tuple(out.shape)}, dt={1000*(time.time()-t1):.1f} ms")

    # Backward sanity (small step)
    t2 = time.time()
    out = m(img, prop)
    loss = out.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"  forward+backward dt={1000*(time.time()-t2):.1f} ms")
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\nOK")


if __name__ == "__main__":
    main()
