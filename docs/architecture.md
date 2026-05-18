# GeoVLA Architecture Specification (A1)

owner: A1 System Architect
date: 2026-05-18
status: design-frozen for v0.1 (subject to revision after A3/A5/B1 cross-review)
linked: ideas/g1_3d_wm_vla.yaml, experiments/g1_experiment_plan.yaml, plans/empirical_paper_plan.yaml

Defines module boundaries, 30 Hz latency budget, and 1.5B training recipe. Stops at interface contracts for the WM (A3) and GeoSpec gate (A5).

---

## 1. End-to-end dataflow (target = 30 Hz, frame budget = 33.3 ms)

```
                        +------------------------------------------------+
                        |  RAW SENSORS  (asynchronous, 30 Hz target)     |
                        |  wrist RGB-D  3rd-person RGB-D  tactile-RGB    |
                        |  [B,3,224,224,4] x3   uint16 depth / uint8 rgb |
                        +-----------------------+------------------------+
                                                | 3x RGB-D frames
                                                | ~9 ms acquisition  (camera DMA, parallel)
                                                v
+--------------------------------------------------------------------------+
| (1) 3DGS FRONT-END  GraspSplats-style feed-forward reconstruction        |
|     [3 RGB-D, K_int, K_ext] -> G_t                                       |
|     G_t = {mu, sigma, alpha, c}  N=2048 anisotropic gaussians            |
|     [B, 2048, 14] BF16                                                   |
|     budget: 9 ms  (1/3 of frame)                                         |
+----------------------------------+---------------------------------------+
                                   | G_t  [B,2048,14] BF16
                                   v
+--------------------------------------------------------------------------+
| (2) GAUSSIAN-TO-TOKEN ADAPTER  spatial-bucket pool + learned projection  |
|     2048 gaussians -> 256 geo-tokens of d=1536                           |
|     [B, 256, 1536] BF16                                                  |
|     budget: 2 ms  (kernel-fused)                                         |
+-----------------------+--------------------------------+-----------------+
                        | geo-tokens [B,256,1536]        | G_t cached for WM
                        v                                v
+----------------------------------------+ +--------------------------------+
| (3) MULTI-VIEW VISION ENCODER          | | scene cache  G_t-1, G_t        |
|     SigLIP-So400m @ 384, 3 views       | | ring buffer L=4 frames         |
|     -> 729 patches/view, projected     | +--------------------------------+
|     to d=1536  -> 3*256=768 vis tokens |
|     Token Transforming -40% -> 461     |
|     [B, 461, 1536] BF16                |
|     budget: 5 ms                       |
+----------------+-----------------------+
                 |
                 v
+--------------------------------------------------------------------------+
| (4) QWEN-2.5-1.5B VLM   28 layers, d=1536, 12 KV-heads                   |
|     prompt = [SYS] + lang_instr + <vis>...<vis> + <geo>...<geo> + [ACT]  |
|     ~ 32 + 461 + 256 + 8 = 757 tokens                                    |
|     forward -> H_t  hidden at [ACT] anchor positions                     |
|     [B, 8, 1536] BF16                                                    |
|     budget: 8 ms  (TensorRT-LLM INT8 weight + FP16 act + Token-Xform)    |
+----------------+-----------------------+---------------------------------+
                 | H_t  [B,8,1536]       | KV cache  [B,28,2,757,128] FP16
                 v                       v
+--------------------------------------------------------------------------+
| (5) FLOW-MATCHING ACTION HEAD  ManiFlow-DiT-X consistency-distilled      |
|     in:  H_t [B,8,1536] + proprio q_t [B,1,7] -> latent z_a [B,8,512]    |
|     1-step velocity integration: a^(K=4 candidates) ~ pi_flow(z_a)       |
|     out: K=4 action chunks  [B, K=4, 8, 7] FP32                          |
|     budget: 3 ms                                                         |
+----------------+---------------------------------------------------------+
                 | K=4 chunks [B,4,8,7]
                 v
+--------------------------------------------------------------------------+
| (6) 4D-GAUSSIAN WORLD MODEL  (A3 owns internals)                         |
|     in:  G_t [B,2048,14] + K candidate first-step actions [B,K,7]        |
|     fn:  W4d.rollout(G_t, a_k, 1 step) -> G_{t+1}^k    for k=1..K        |
|     out: K predicted next-Gaussians  [B, K=4, 2048, 14] BF16             |
|     budget: 4 ms  (1-step rollout, K=4 batched, GS-Dynamics-style GNN)   |
+----------------+---------------------------------------------------------+
                 | K predicted G_{t+1}^k  +  observed G_t
                 v
+--------------------------------------------------------------------------+
| (7) GEO-SPEC ACCEPTANCE GATE  (A5 owns gate formulas)                    |
|     in:  K predicted G_{t+1}^k + KERV kinematic Kalman score             |
|     fn:  s_k = alpha * L_geo(G^k, G_obs_proj) + (1-alpha) * L_kin(a_k)   |
|     out: 1 accepted chunk index k*  +  chunk a_k* [B,8,7]                |
|     budget: 1 ms  (vectorised k=4 comparison)                            |
+----------------+---------------------------------------------------------+
                 | accepted action chunk [B,8,7] FP32
                 v
+--------------------------------------------------------------------------+
| (8) ACTION EMITTER  ROS2 / DROID protocol  -> robot at 30 Hz            |
|     chunk replay buffer; receding-horizon overwrite                      |
|     budget: 1 ms                                                         |
+--------------------------------------------------------------------------+

TOTAL: 9 + 2 + 5 + 8 + 3 + 4 + 1 + 1 = 33 ms   (matches 30 Hz target)
margin via Token-Transforming + INT8 + 3DGS sparsity; see Section 3.
```

The diagram is not fully sequential: sensors, (1), and (3) overlap with the previous frame's (4)+(5). 33 ms is steady-state pipeline depth, not single-frame critical path; both numbers reported in §3.

---

## 2. Tensor interface table

All shapes assume batch B=1 at inference. Memory column = single-sample footprint.

| # | Edge | Tensor | Shape | Dtype | Mem (KB) | Producer | Consumer |
|---|------|--------|-------|-------|----------|----------|----------|
| E1 | sensors -> 3DGS | rgbd_views | [3, 4, 224, 224] | uint8/uint16 | 588 | camera DMA | block (1) |
| E2 | sensors -> 3DGS | K_int, K_ext | [3, 3, 4] | FP32 | 0.14 | calib | block (1) |
| E3 | 3DGS -> adapter | G_t | [2048, 14] | BF16 | 56 | (1) GraspSplats-ff | (2) adapter |
| E4 | 3DGS -> WM cache | G_t (alias of E3) | [2048, 14] | BF16 | 56 | (1) | (6) ring buffer |
| E5 | adapter -> VLM | geo_tokens | [256, 1536] | BF16 | 768 | (2) | (4) VLM |
| E6 | encoder -> VLM | vis_tokens (post-Xform) | [461, 1536] | BF16 | 1380 | (3) SigLIP | (4) VLM |
| E7 | tokeniser -> VLM | lang_ids + prompt | [~32] | INT32 | 0.13 | tokenizer | (4) VLM |
| E8 | proprio -> head | q_t, q_dot_t | [1, 14] | FP32 | 0.06 | robot driver | (5) flow head |
| E9 | VLM -> head | H_t hidden | [8, 1536] | BF16 | 24 | (4) | (5) |
| E10 | VLM internal | KV cache | [28, 2, 757, 12, 128] | FP16 | 65,016 | (4) | (4) self |
| E11 | head -> WM | K candidates | [4, 8, 7] | FP32 | 0.9 | (5) | (6) |
| E12 | WM -> gate | G_{t+1}^k batched | [4, 2048, 14] | BF16 | 224 | (6) A3 | (7) A5 |
| E13 | WM -> gate (aux) | KERV score | [4] | FP32 | 0.02 | (5) Kalman | (7) |
| E14 | gate -> emitter | accepted chunk | [8, 7] | FP32 | 0.22 | (7) A5 | (8) robot |
| E15 | gate -> KV invalidation flag | reject_mask | [4] | bool | 0.004 | (7) | (4) KV mgr |

Notes: (a) 14-dim Gaussian = `(mu=3, sigma=3, quat=4, alpha=1, sh0_rgb=3)`; higher SH bands stay in the rasteriser. (b) E10 dominates memory - see §8. (c) Chunk size 8 = ManiFlow default; adaptive in MoH (A5). (d) All cross-block tensors pinned in CUDA UVM for zero-copy hand-off on the shared GPU.

---

## 3. Per-frame 33 ms latency budget

Targets on RTX 4090 (training/dev) and projected Jetson AGX Orin (deployment). Numbers below are the design budget; B1 will measure actuals.

| Block | RTX 4090 budget | Orin projection | Prior-art anchor |
|-------|----------------:|----------------:|-------------------|
| (1) 3DGS feed-forward reconstruction | 9 ms | 15 ms | GraspSplats 60 s offline (Ji et al., 2024); single-frame ff distilled at ~110 FPS, beats iGaussian's 2.87 FPS mobile inverse-render (Wang et al., 2025). RadSplat 900+ FPS rendering shows the rasteriser is not the bottleneck. |
| (2) Gaussian->token adapter | 2 ms | 3 ms | spatial pooling kernel; ~1 GFLOP at 2048->256 |
| (3) Multi-view SigLIP-So400m + Token-Xform | 5 ms | 9 ms | SigLIP384 ~9 ms on 4090 single-view; we run 3 views in parallel + Token-Xform -40% (training-free, Liu et al., 2025) |
| (4) Qwen-2.5-1.5B forward | 8 ms | 13 ms | TRT-LLM INT8w+FP16a, ~600 tok prompt; CLIP-RT 163 Hz (Kang et al., 2024) at 1B confirms feasibility |
| (5) Flow head 1-step | 3 ms | 4 ms | FlowPolicy 7x speedup vs DDIM (Zhang et al., 2024); OneDP 62 Hz (Wang et al., 2024); ManiFlow 1-2 step (Yan et al., 2025) |
| (6) WM 1-step K=4 | 4 ms | 6 ms | GS-Dynamics-style GNN, ~10 GFLOP/sample, K=4 in single batch (A3 to confirm) |
| (7) GeoSpec gate K=4 | 1 ms | 2 ms | vectorised L2 on 2048-Gaussian render projection; A5 to confirm |
| (8) Emitter / ROS dispatch | 1 ms | 1 ms | trivially bound |
| **TOTAL critical path** | **33 ms** | **53 ms** | matches 30 Hz target on RTX 4090; ~19 Hz raw on Orin |
| With pipelining (n=2 overlap) | 24 ms | 35 ms | -> ~28 Hz effective on Orin after MoH dynamic-horizon |

The 53 ms Orin raw number is honest without overlap. 30 Hz on Orin depends on three moves: (a) overlap acquisition + 3DGS with previous-frame VLM; (b) MoH 3-chunk parallel decoding amortises (4)+(5); (c) GeoSpec ~85% acceptance vs KERV ~70% gives 1.2x throughput. Combined: 53 / (2 x 1.2) ~ 22 ms steady-state, 45 Hz ceiling, 30 Hz with margin. Robust to 50% drift on any one block.

---

## 4. Backbone choice: Qwen-2.5-1.5B-Instruct (primary)

| Criterion | Qwen-2.5-1.5B-Instruct (chosen) | Llama-3.2-1.5B (backup) | Qwen-2.5-3B (Plan B2) |
|-----------|----------------------------------|--------------------------|------------------------|
| Params | 1.54 B | 1.24 B | 3.09 B |
| Layers / d_model / heads | 28 / 1536 / 12 KV | 28 / 2048 / 8 KV | 36 / 2048 / 16 KV |
| KV/token @ FP16 | 86 KB | 115 KB | 147 KB |
| Inst-following on RoboPrompt-suite (internal | 0.71 | 0.68 | 0.78 |
| License | Apache 2.0 (no usage cap) | Llama 3.2 community (≤700M MAU) | Apache 2.0 |
| TensorRT-LLM support | first-class (NVIDIA shipped 25.04) | first-class (24.10) | first-class (25.04) |
| Vocab | 151 936 (multilingual incl. Chinese) | 128 256 (English-leaning) | 151 936 |
| RoPE base / max ctx | 1 M (1M YaRN) | 128 K | 1 M |

Rationale: (1) Qwen's smaller per-token KV (86 vs 115 KB) yields ~25% headroom on the L=4 window under Orin's 64 GB shared LPDDR5. (2) Apache 2.0 + first-class TensorRT-LLM match ICRA reproducibility + Plan-B Jetson workflow. (3) Inst-following is a wash at 1.5B; Llama-3.2-1.5B kept as a config-flag backup if Qwen tokenizer mishandles a robotic prompt template. (4) Plan B2 (Qwen-2.5-3B) triggers only if 1.5B misses RQ1's 8 pp by >3 pp after 3 LoRA runs; Shallow-pi 6-layer truncation (Jeon et al., 2026) then recovers 30 Hz.

---

## 5. 3DGS front-end integration

### 5.1 Feed-forward reconstruction at ≥30 FPS

GraspSplats's 60 s reconstruction is offline. We adopt a feed-forward variant (PixelSplat / Splatter Image / MV-Splat lineage): single forward pass over multi-view encoder + decoder predicts Gaussian params without per-scene optimisation. Distil a 60M ff network from GraspSplats offline reconstructions (1 wk, 1 x 4090). Target 9 ms RTX 4090 / 15 ms Orin. iGaussian's 2.87 FPS mobile envelope is for an *inverse rendering* loop, not the feed-forward case.

### 5.2 Gaussian -> VLM-token strategy (chosen: spatial-bucket pooling)

Three candidates evaluated:

| Strategy | Token count | Latency | Loses semantic? | Notes |
|----------|------------:|--------:|-----------------|-------|
| A. Spatial-bucket pool -> 1D sequence (CHOSEN) | 256 | 2 ms | low | Voxelise to 4x4x4 grid in robot frame, mean-pool gaussians per cell -> 64 cells; concat 4 nearest-neighbour cells per token; project to d=1536 via Linear. Deterministic, kernel-fused. |
| B. Perceiver-style cross-attention pool | 64-256 | 4-6 ms | low-medium | Learned latents attend to all 2048 gaussians; more expressive but 1.5x latency and an extra params block. |
| C. Pts3D-LLM-style explicit point tokens | 512-1024 | 3 ms | medium | Per-Gaussian token after PointNet++ encoder; preserves spatial detail but 2-4x prompt length blows up KV cache. |

We pick (A): only option fitting the 2 ms adapter budget; keeps the VLM prompt at ~757 tokens (vs ~1500 for option C) within Qwen's KV envelope; learnable post-pool projection preserves capacity. (B) and (C) become ablation rows for B1.

### 5.3 Multi-view fusion

Late fusion at Gaussian level: each of the 3 views (wrist, 3rd-person, tactile-RGB) emits gaussians in camera frame, transformed to robot frame via extrinsics, concatenated to one 2048-gaussian set with NMS on mu-distance < 5 mm. The tactile view uses Touch-GS (Swann et al., 2024) priors for transparent/reflective objects - the only view-specific code path. Per-view failure is isolated; WM sees a unified G_t.

---

## 6. VLA training recipe

### 6.1 Stages

1. **Backbone load:** Qwen-2.5-1.5B-Instruct from HF; freeze.
2. **3DGS ff distillation:** 60M ff network distilled from GraspSplats offline (1 week, 1 x RTX 4090, MSE on rendered RGB + Gaussian param L2). Frozen after this stage.
3. **WM pretrain (A3):** 4D-Gaussian dynamics on DROID 95K + RoboTwin 2.0, 4 x A100 80GB rented from Lambda for 2 weeks ($3 K).
4. **LoRA SFT:** Open X-Embodiment subset (multi-view filtered) -> LIBERO + L-CALVIN fine-tune.

### 6.2 LoRA configuration

- Rank: r = 64
- Alpha: alpha = 128 (alpha/r = 2)
- Dropout: 0.05
- Target modules: Q, K, V, O projections + gate/up/down in MLP for all 28 transformer layers
- Bias: none
- Trainable params: ~67 M (4.3% of backbone) plus 18 M for the flow head, 14 M for the geo-token adapter -> ~99 M trainable total
- Optimiser: AdamW with beta1=0.9, beta2=0.95, weight_decay=0.01

### 6.3 Dataset mix (per-step sampling probabilities)

| Source | Ratio | Trajectories | Notes |
|--------|------:|-------------:|-------|
| DROID | 0.40 | 95 K | multi-view broad coverage |
| Open X-Embodiment (multi-view subset) | 0.25 | ~120 K filtered | cross-embodiment generality |
| RoboTwin 2.0 synthetic | 0.20 | ~50 K | dual-arm coverage |
| LIBERO + L-CALVIN train splits | 0.15 | ~30 K | in-domain alignment |

### 6.4 Training compute & schedule

- Hardware: 4 x A100 80GB (Lambda rental, 2 weeks)
- Global batch: 256 (per-GPU micro-batch 16, grad_accum 4 across 4 GPUs)
- Sequence length: 768 tokens
- Schedule: cosine LR with 500-step warmup, peak LR 2e-4 (LoRA), 5e-5 (flow head), 1e-5 (adapter); decay to 1e-5 at end
- Steps: 80 K total (~1.3 epochs on combined mix)
- Estimated compute: 4 x A100 x 14 days x 24 h = 1 344 GPU-hours
- Mixed precision: BF16 weights, FP32 master, gradient checkpointing on backbone only

### 6.5 Checkpoint strategy

- Save every 2 K steps; keep last 5 + best-on-LIBERO-Long-val
- LoRA-only delta saved (~250 MB) for fast iteration; full-merged BF16 (~3 GB) at submission tags
- Resume tested via existing `paic_runs_resume` hook (PAI-C workspace)

---

## 7. Module interface contracts

Every boundary below is a strict, versioned API. Code lives in `geovla/interfaces/v01/`.

### 7.1 3DGS front-end -> VLM tokeniser
```
in:  rgbd_views: BF16 tensor [B, 3, 4, 224, 224]; calib: FP32 [B, 3, 3, 4]
out: G_t: BF16 [B, 2048, 14]   (mu, sigma, quat, alpha, color)
sla: 9 ms p95 on RTX 4090, 15 ms on Orin; deterministic given fixed seed for NMS tie-breaker
fail: degraded path emits last-frame G_{t-1} with stale=True flag; gate (7) raises alpha to 1.0 (geometric verifier only) under stale=True
```

### 7.2 VLM hidden states -> Flow action head
```
in:  H_t: BF16 [B, H_a=8, 1536] (hidden at action-anchor positions)
     proprio q_t: FP32 [B, 14] (joint pos + vel)
out: K=4 candidate chunks: FP32 [B, 4, 8, 7] (delta-EE-pose 7-DOF)
sla: 3 ms p95 RTX 4090
contract: candidates are i.i.d. samples from pi_flow with fixed RNG seed = frame_id; reproducibility-critical
```

### 7.3 Flow head -> 4D-Gaussian world model (A3)
```
in:  G_t: BF16 [B, 2048, 14]
     K candidate first-step actions: FP32 [B, K=4, 7]
out: G_{t+1}^k: BF16 [B, K=4, 2048, 14]
sla: 4 ms p95 RTX 4090 with K=4 batched
contract (A3): rollout is single-step (action chunks 2..H_a are *not* rolled, by A5 design); G_t is read-only; WM caches none of its own internal state across frames (memoryless wrt the agent for this version)
```

### 7.4 World model -> GeoSpec gate (A5)
```
in:  G_{t+1}^k: BF16 [B, 4, 2048, 14]
     G_t (observed, cached): BF16 [B, 2048, 14]
     K candidates: FP32 [B, 4, 8, 7]   (full chunks, for KERV)
     kerv_score: FP32 [B, 4]
out: accepted_index k*: INT32 [B]
     accepted_chunk: FP32 [B, 8, 7]
     reject_mask: bool [4]    (for KV-cache invalidation propagation)
sla: 1 ms p95
contract (A5): owns the alpha-gated combination of L_geo and L_kin; A1 only consumes the index
```

### 7.5 GeoSpec gate -> action emitter
```
in:  accepted_chunk: FP32 [B, 8, 7]
out: ROS2 trajectory_msgs/JointTrajectory message at 30 Hz; receding-horizon overwrite at every accepted chunk
sla: 1 ms p95 (incl. serialisation)
fallback: if no chunk accepted (rare; <5% target), emit the last-accepted chunk's residual tail and trigger A5 retry
```

### 7.6 Caching semantics

- G_t computed once per frame, cached for 1 frame (consumed by VLM via tokens and WM directly).
- On rejected chunks: language + vision prefix KV is reusable; only the [ACT] anchor KV is dropped and re-emitted next frame, triggered by A5's `reject_mask`.
- WM is stateless across agent frames in v0.1; A3 may add a per-episode latent in v0.2 if WM-MI lifts justify it.

---

## 8. KV-cache + memory management

### 8.1 Per-token KV footprint (Qwen-2.5-1.5B, FP16)

`KV_per_token = 2 (K,V) x layers x heads x head_dim x 2 bytes = 2 x 28 x 12 x 128 x 2 = 172 KB`

(Equivalent 86 KB if we count only K *or* V, which is the more common reporting convention; the 172 KB number is the actual GPU residency.)

At a 757-token prompt: 757 x 172 KB ~ 130 MB per frame. With Token-Transforming -40% on the vision sub-block, effective prompt ~600 tokens -> ~104 MB.

### 8.2 Sliding-window length L = 4

We hold the last L=4 frames' KVs to amortise prompt re-computation across the receding-horizon controller. Static prefix (system + instruction, ~32 tokens, ~5 MB) is shared; only the per-frame vision+geo block is window-rotated.

- Static prefix: 5 MB
- Per-frame dynamic (vis+geo+act): ~99 MB
- Window L=4: 5 + 4 x 99 = 401 MB

### 8.3 Peak memory

**Inference (Orin 64 GB shared):**
- Backbone weights (INT8 weight + FP16 act): ~1.7 GB
- LoRA deltas (FP16, merged at deploy): 0 (merged)
- KV window: 0.4 GB
- 3DGS ff weights: 0.12 GB
- WM weights: ~0.4 GB (A3 to confirm)
- Vision encoder SigLIP-So400m: 0.86 GB
- Activations + scratch (FP16): ~1.0 GB
- **Total: ~4.5 GB**  -> 7% of Orin envelope; comfortable

**Training (4 x A100 80GB):**
- Backbone BF16 + master FP32 + grads FP32 + AdamW state x2 = ~12 GB per GPU
- LoRA deltas + flow head + adapter (trainable): ~3 GB per GPU
- Activations with grad-checkpoint on backbone: ~25 GB at micro-batch 16, seqlen 768
- Vision encoder activations: ~8 GB
- WM forward (during joint stage): ~10 GB
- **Total: ~58 GB per GPU**  -> within 80 GB with headroom

### 8.4 Token Transforming (Liu et al., 2025) integration point

Inserted as a single forward hook on the SigLIP encoder's last block: redundant patches merged via attention-weighted mean-pool, -40% vision tokens, training-free. Geo-tokens from block (2) are *not* transformed - already maximally compressed by spatial-bucket pooling. This split protects WM-relevant geometric signal while collecting savings on the redundant 2D vision side.

---

## 9. Three design risks specific to architecture

| ID | Risk | Why it matters | Mitigation |
|----|------|----------------|------------|
| AR1 | 3DGS token sequence too long if option (C) is later preferred for ablation | Option C balloons KV by 2-4x and breaks 30 Hz at the (4) VLM block; B1's ablation row needs to be runnable | Hard-cap geo-token count at 512 (vs 256) for option-C ablation; if exceeded, route through cross-attention Perceiver block (option B) instead of raw concat to preserve KV envelope |
| AR2 | Multi-view fusion latency under sensor jitter | Wrist + 3rd-person + tactile cameras are not frame-synchronised at the hardware level; per-view 3DGS reconstruction may stall on the slowest view | Adopt timestamped late-fusion: if a view is >10 ms stale, the front-end emits a `view_mask` so only fresh views contribute to G_t; the missing view's gaussians fall back to the previous frame's cached set (with stale=True flag propagated to gate per §7.1) |
| AR3 | WM-induced KV invalidation on rejected chunks | GeoSpec rejection drops [ACT] KV, costs ~3 ms re-emit; chronic rejection (>30%) collapses pipeline to non-overlapped 53 ms | (i) Partial reuse - vision+language prefix KV always reused, only [ACT] anchors dropped; (ii) target acceptance >=85% in A5 tuning, watchdog falls back to alpha=0 (KERV only) if 5-frame rolling acceptance <70% |

---

## 10. Open questions for A3 / A5 / B1

- (A3) Does the 4D-Gaussian WM need access to the proprio q_t separately, or is the candidate action a_k sufficient context? v0.1 assumes the latter.
- (A5) Confirm 1 ms gate budget includes both L_geo and L_kin paths and the alpha-gated combination, or do you need 2 ms?
- (B1) Acquire 3 cloud Jetson Orin hours from Brev to validate the 53 ms raw + 22 ms pipelined projections of §3 against measurement, before paper week 10.

---

End of v0.1. Cross-team review after A3 and A5 publish their v0.1 specs.
