#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3DGS perception front-end feasibility check on A800.

Why this matters: GeoVLA's per-frame latency budget allocates 9 ms to feed-forward
3DGS reconstruction (master synthesis §2). We need to know early — *before* W3 —
whether the GraspSplats-style feed-forward reconstruction at our target resolution
fits that budget on A800, or whether we need to fall back to GenieSim PanoRecon
(Plan B for §3.1 of A1's architecture spec).

This script does NOT train anything. It simulates the perception path's compute
profile:
  - allocate the dense-Gaussian buffer (N_DENSE_MAX, D_GAUSSIAN) in FP16
  - run a representative dense-to-sparse downsampling proxy (FPS over N_DENSE=12k → N_CONTROL=2048)
  - run a small GNN-like message-passing proxy (5 layers × 2048 nodes × 16 neighbours)
  - measure throughput in FPS on the current GPU

Numbers establish the GPU "headroom" lower-bound. If A800 cannot sustain >100 FPS
on the proxy, real 3DGS reconstruction definitely won't fit.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from geovla import interfaces as I


def bench(fn, n_iter: int = 100, warmup: int = 20, label: str = ""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    total = (time.time() - t0)
    per = total / n_iter * 1000
    print(f"  {label:40s} {per:7.2f} ms/iter   {1000/per:7.1f} FPS")
    return per


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"\nLocked interface constants:")
    print(f"  N_DENSE_MAX  = {I.N_DENSE_MAX}")
    print(f"  N_CONTROL    = {I.N_CONTROL}")
    print(f"  N_GEO_TOKENS = {I.N_GEO_TOKENS}")
    print(f"  D_GAUSSIAN   = {I.D_GAUSSIAN}")
    print(f"  D_HIDDEN     = {I.D_HIDDEN}")
    print(f"  K_PROPOSALS  = {I.K_PROPOSALS}")
    print()

    print("=== 3DGS perception proxy benchmarks ===\n")

    # 1. dense buffer allocation
    def alloc_dense():
        x = torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN, device=dev, dtype=torch.float16)
        return x
    bench(alloc_dense, label="alloc dense (80k, 14) FP16")

    # 2. FPS-style downsampling proxy (sort-by-distance heuristic)
    dense = torch.randn(12000, 3, device=dev)
    def fps_proxy():
        # simulated: k random anchors + distance-to-anchor scoring
        anchor = dense[torch.randint(0, 12000, (8,), device=dev)]  # (8, 3)
        d = torch.cdist(dense, anchor)
        d_min = d.min(dim=1).values
        idx = d_min.topk(I.N_CONTROL, largest=True).indices
        return dense[idx]
    bench(fps_proxy, label="FPS downsample 12k -> 2048 (proxy)")

    # 3. message-passing GNN proxy (5 layers, 2048 nodes, hidden 256)
    nodes = torch.randn(I.N_CONTROL, 256, device=dev, dtype=torch.bfloat16)
    edge_idx = torch.randint(0, I.N_CONTROL, (2, I.N_CONTROL * 16), device=dev)
    weights = [torch.randn(256, 256, device=dev, dtype=torch.bfloat16) for _ in range(5)]
    def gnn_proxy():
        x = nodes
        for w in weights:
            # naive scatter-mean aggregation
            src = x[edge_idx[0]]
            agg = torch.zeros_like(x)
            agg.index_add_(0, edge_idx[1], src)
            x = (agg @ w).relu()
        return x
    bench(gnn_proxy, label="GNN 5-layer message-pass (2048,256)")

    # 4. spatial-bucket pool (geo-tokeniser proxy)
    points = torch.randn(I.N_CONTROL, 3, device=dev)
    feats = torch.randn(I.N_CONTROL, I.D_HIDDEN, device=dev, dtype=torch.bfloat16)
    def pool_proxy():
        # bucket by integer quantisation of position into 4x4x4
        idx = ((points - points.min(0).values) / (points.max(0).values - points.min(0).values + 1e-9) * 4).long().clamp_(0, 3)
        flat = idx[:, 0] * 16 + idx[:, 1] * 4 + idx[:, 2]
        pooled = torch.zeros(256, I.D_HIDDEN, device=dev, dtype=torch.bfloat16)
        for c in range(64):
            mask = flat == c
            if mask.any():
                pooled[c] = feats[mask].mean(0)
        return pooled
    bench(pool_proxy, label="spatial-bucket pool 2048 -> 256")

    # 5. K=4 parallel WM forward proxy
    def wm_k4_proxy():
        # 4 copies of GNN running in parallel batch
        x = nodes.unsqueeze(0).expand(I.K_PROPOSALS, -1, -1)  # (4, 2048, 256)
        for w in weights:
            x = (x @ w).relu()  # batched matmul
        return x
    bench(wm_k4_proxy, label="WM K=4 parallel batch forward")

    # 6. backbone-size FP16 matmul as proxy for VLM forward cost
    a = torch.randn(2048, I.D_HIDDEN, device=dev, dtype=torch.bfloat16)
    b = torch.randn(I.D_HIDDEN, I.D_HIDDEN, device=dev, dtype=torch.bfloat16)
    def vlm_layer_proxy():
        for _ in range(28):  # ~1.5B model layer count
            _ = a @ b
        return _
    bench(vlm_layer_proxy, label="VLM 28-layer matmul proxy (BF16)")

    print("\n=== Summary ===")
    print("If all proxies are >100 FPS each, A800 has headroom to hit 30 Hz GeoVLA target.")
    print("If GNN K=4 < 80 FPS or VLM proxy < 40 FPS, expect to need optimisation.")


if __name__ == "__main__":
    main()
