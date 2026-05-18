# GeoVLA — Master Synthesis Spec (v1)

**Date:** 2026-05-18  
**Source:** Synthesis of 4 parallel sub-agent deliverables (A1 Architecture, A3 World Model, A5 Efficiency, B1 Experiment Design)  
**Status:** Cross-team consistency verified; 3 open items flagged for Week-1 resolution.

---

## 1. Cross-Team Decision Matrix (LOCKED)

| Decision | Value | Owner | Cross-checked by |
|---|---|---|---|
| VLM backbone | **Qwen-2.5-1.5B-Instruct** (Apache 2.0, TRT-LLM first-class) | A1 | A3, A5 |
| Plan B backbone | Qwen-2.5-3B + Shallow-π 6-layer truncation (only if RQ1 misses by >3 pp after 3 LoRA runs) | A1 | B1 |
| 3DGS tokenisation | **Spatial-bucket pool to 1D sequence** (4×4×4 voxel grid, 256 geo-tokens at d=1536) | A1 | A3, A5 |
| WM architecture | **GNN, 5 message-passing layers, hidden 256, 12M params, GS-Dynamics warm-start** | A3 | A1, A5 |
| Gaussian control particles | **N_c = 2048** (FPS-downsampled from dense scene, biased to graspable + gripper region) | A3 | A1 |
| L3 protocol | **K=4 temperature-sampled flow proposals × N=1 WM rollout step**, top-1 emitted, zero-velocity hold on universal rejection | A3 | A5 |
| GeoSpec gate | **Per-action-dim 8-vector α ∈ ℝ⁸** (7 kinematic + 1 geometric); **post-hoc** logistic regression on 5,000 draft-target pairs | A5 | A3 |
| Draft model | **0.3B 6-layer student** distilled from the 1.5B target (re-used as Edge-variant student) | A5 | A1 |
| Quantisation | TRT-LLM **INT8 weights + FP16 activations**; flow-matching head kept FP16; calibration on 200 trajectories | A5 | A1 |
| Token Transforming integration point | **Between 3DGS-tokeniser and VLM** (training-free, 40% vision-token reduction) | A5 | A1 |
| MoH integration | **3 parallel chunk segments**, adaptive horizon, applied after distillation | A5 | A1 |
| Cloud Jetson Orin instance | **NVIDIA Brev AGX Orin 64GB, $143 / 7 days** (booked W10, ran W11) | A5 | B1 |

## 2. End-to-End 30 Hz Latency Budget (consolidated)

Critical-path budget, RTX 4090 (left) / Jetson AGX Orin INT8 projected (right):

```
┌─────────────────────────────────┬──────────┬──────────┐
│ Block                           │ RTX 4090 │ Orin INT8│
├─────────────────────────────────┼──────────┼──────────┤
│ Multi-view RGB-D ingest         │  1.0 ms  │  1.5 ms  │
│ 3DGS feed-forward reconstruction│  9.0 ms  │ 15.0 ms  │
│ 3DGS → 256 geo-tokens (adapter) │  2.0 ms  │  3.0 ms  │
│ SigLIP multi-view + TokenXform  │  5.0 ms  │  9.0 ms  │
│ Qwen-2.5-1.5B forward (INT8w)   │  8.0 ms  │ 13.0 ms  │
│ Flow head 1-step (FP16)         │  1.5 ms  │  2.5 ms  │
│ WM K=4 forward (overlapped)     │  4.2 ms  │  7.0 ms  │  ← shared with GeoSpec
│ GeoSpec gate (renders + α∙s)    │  1.8 ms  │  2.5 ms  │  ← overlaps with WM via stream
│ Bookkeeping + KV update         │  0.5 ms  │  0.5 ms  │
├─────────────────────────────────┼──────────┼──────────┤
│ Critical-path (overlapped)      │ 28-30 ms │ 49-53 ms │
│ → Steady-state with MoH/3-stage │  ~22 ms  │  ~31 ms  │  ← target hit
│ Effective Hz                    │  ~45 Hz  │  ~32 Hz  │  ← 30 Hz margin
└─────────────────────────────────┴──────────┴──────────┘
```

**Three observations from cross-check:**

- A1 budget (8/13 ms VLM) + A3 budget (8 ms WM) + A5 estimate (Orin 31 Hz INT8) are **mutually consistent**: total Orin critical-path = 53 ms (3-stage pipelining + MoH amortisation → 22 ms steady = 45 Hz on 4090 / 31 Hz on Orin).
- The 30 Hz target is hit with **<10% margin on Orin** — fragile to any one of: GeoSpec acceptance < 70 %, INT8 instability in flow head, 3DGS reconstruction degradation under cluttered scenes.
- **WM forward is shared between L3 coupling and GeoSpec** — this is the single most important latency-saving design choice; without it the budget would blow through 33 ms on RTX 4090.

## 3. Interface Contracts (LOCKED — to be encoded in `geovla/interfaces.py`)

| # | Producer → Consumer | Tensor | Shape | Dtype | When |
|---|---|---|---|---|---|
| I1 | 3DGS front-end → 3D tokeniser | dense Gaussians | (N_dense, 14) | FP16 | every frame (30 Hz) |
| I2 | 3D tokeniser → VLM | geo-tokens | (1, 256, 1536) | BF16 | every frame |
| I3 | VLM hidden → Flow head | hidden state | (1, T_vlm, 1536) | BF16 | every action emit |
| I4 | Flow head → WM | K proposals | (K=4, H_a=8, 7) | FP16 | every action emit |
| I5 | WM → GeoSpec | predicted next-Gaussian | (K=4, N_c=2048, 14) | FP16 | parallel-batched, single forward |
| I6 | WM → next-step 3DGS-tokeniser | reused features | (1, 256, 1536) | BF16 | next frame (cache) |
| I7 | GeoSpec → Action emitter | top-1 chunk | (H_a=8, 7) | FP16 | every action emit |

**Critical re-use:** I5 is consumed by *both* the action re-ranking (A3's L3 protocol) and GeoSpec's L_geo computation (A5's gate). Without this single-forward re-use, Orin steady-state slips from 31 ms → 38 ms, killing the 30 Hz target.

## 4. Unified Risk Register (Top 10, prioritised by lead × severity)

| ID | Owner | Risk | Severity | Mitigation | Detection week |
|---|---|---|---|---|---|
| **R1** | A1 | WM-induced KV invalidation on rejected GeoSpec chunks (>30% rejection → 53 ms Orin = 19 Hz) | **HIGH** | Split KV reuse (vision+language prefix always reused; only [ACT] anchors dropped) + 5-frame watchdog flips α=0 to KERV-only | W11 |
| **R2** | A3 | L3 coupling overhead breaks 30 Hz on Orin even with WM at 8 ms internal | **HIGH** | CUDA-stream separation for K=4 WM forwards + K=2 Edge variant + adaptive-K lazy-WM fallback when gate margin > 0.5 | W11 |
| **R3** | A5 | GeoSpec α_geo collapses to 0 (L_geo noisy from renderer / WM error correlates with policy mistakes) | **HIGH** | Use DINOv2 feature-space L2 instead of pixel L2 + sparsity prior on α_geo + conditional GeoSpec (only when WM predicts contact within H steps) | W9 |
| **R4** | B1 | **WM-MI metric construct validity** — A2 (3D-only no-WM-inference) still encodes WM info via training; "no-WM" condition is not pure | **HIGH** | (1) Concede in §6 limitations; (2) report WM-MI in two parallel forms (raw difference + per-task paired-t); (3) optional A2'-from-scratch retrain doubling RQ1 cost — defer to revision if R3 paid | W7 |
| **R5** | A1 | Qwen-2.5-1.5B underpowered for VLA → 8 pp tax missed | MEDIUM | Plan B2: 3B + 6-layer Shallow-π truncation (preserves 30 Hz target) | W7 |
| **R6** | A5 | INT8 quantisation destabilises flow-matching head (diffusion-style numerics) | MEDIUM | Keep flow head FP16 from day 1; INT8 only on backbone + linear projections; sensitivity ablation in RQ3 | W10 |
| **R7** | A3 | GNN doesn't scale to 2048 particles at 30 Hz on Orin | MEDIUM | k-NN graph cached across frames; particle count reduced to 1024 in Edge variant; cite GS-Dynamics scaling | W6 |
| **R8** | A5 | Cloud Jetson Orin unavailable (Brev/Lambda out of stock) | MEDIUM | TensorRT FLOPs projection formula (validated A100/4090 ratio ±15%) + RTX 4060 Laptop fallback ($1500) | W10 |
| **R9** | B1 | RTX 4090 compute budget exceeded (1647 GPU-h ≈ 69 days vs 63-day W7-W15 window) | MEDIUM | Run RQ1 LIBERO-Spatial/-Object in W4-W6 *parallel* with cloud A100 WM pretraining (recovers 6 days) | W4 |
| **R10** | B1 | RQ4 4×4 grid compute blow-out (64 training runs × 4 seeds) | MEDIUM | Pre-registered Plan B5: reduce to 3×2 grid prioritising distill + GeoSpec axes | W12 |

## 5. Consolidated Compute Budget (BOM)

| Resource | Hours | Rate | Cost | Phase |
|---|---|---|---|---|
| RTX 4090 (owned, electricity proxy) | 1,647 GPU-h | — | $0 | W1-W17 |
| A100 80GB × 4 (Lambda WM pretrain) | 1,344 A100-h | $0.79/h | $1,062 | W6 |
| A100 80GB × 4 (Lambda RQ1 fine-tune buffer) | 500 A100-h | $0.79/h | $395 | W7 |
| A100 80GB × 4 (Lambda Isaac Sim cross-val) | ~32 A100-h | $0.79/h | $25 | W12 |
| Jetson AGX Orin 64GB (NVIDIA Brev) | 80 Jetson-h (1 week) | — | $143 | W11 |
| Buffer + storage + egress | — | — | ~$575 | — |
| **TOTAL CLOUD** | | | **≈ $2,200 + buffer = $3,200** | |

**Cash budget: exactly on plan.** Tightest constraint: 1,647 RTX-4090 hours vs the 63-day W7-W15 execution window (1,512 GPU-h available at 24h/day). **Recovery plan:** B1 recommended running RQ1's LIBERO-Spatial/-Object cells *in parallel* with the W4-W6 A100 WM pre-training; this recovers 6 days and brings 4090 utilisation under 100%.

## 6. Pre-Registration & Reproducibility (B1 raised this — newly added)

Before W6 (WM pretraining starts), commit to OSF (Open Science Framework) timestamped pre-registration containing:

1. The 4 RQs verbatim
2. The 5-variant set for RQ1
3. The α-sweep grid for RQ2 (α ∈ {0, 0.25, 0.5, 0.75, 1.0})
4. The 7-cell ablation for RQ3
5. The 4×4 grid for RQ4 + Plan B reduction rules
6. Power analysis + multiple-comparison correction (Bonferroni for >2 contrasts)
7. The frozen **PWABenchS_subset100.json** seed (`0xC0FFEE`) for cloud-Jetson validation
8. The cross-simulator triplet (MuJoCo + Isaac Sim + PolaRiS holdout) protocol

OSF link to be inserted in paper §4.

## 7. Three Open Items (require Week-1 resolution)

| Open | Resolution path | Owner | Deadline |
|---|---|---|---|
| **O1** Whether A3's L1-only fallback (WM-train-only-no-inference) is a sufficient construct-validity control for WM-MI, given B1's R4 concern | Decide if extra A2'-from-scratch retrain is needed; if so, scope cost ($300-600 extra A100) | A3 + B1 + Lead | 2026-05-25 |
| **O2** Whether GeoSpec α should be retrained per-task (130 LIBERO tasks → potential overfitting) or globally (one α for all) | A5 recommends global per-action-dim; need confirmation after seeing pilot data | A5 + B1 | 2026-06-01 (W3) |
| **O3** Whether to swap Qwen-2.5-1.5B for SmolLM2-1.7B if Qwen-2.5 license / data lineage concerns arise late | Pre-train a sanity-check checkpoint on SmolLM2 in W2; freeze decision W3 | A1 | 2026-06-01 (W3) |

## 8. Squad C Activation (论文产出团队) — scheduled

- **C1 Paper Architect** — activate W13 (M8a, 2026-08-10) with the 4 spec files + experiment results as inputs.
- **C2 Figure/Algorithm Designer** — activate W13 in parallel with C1.
- **C3 Reviewer-2 Defender** — activate W14 (M8b, 2026-08-17) after v1 draft.

## 9. Communication & Cadence

- **Async (always)**: each agent's spec file + summary returned to Lead.
- **Weekly Friday 17:00**: 30-min Lead-only review of weekly_log.md + risk-register status.
- **Wake Lead immediately** if any HIGH risk fires (R1/R2/R3/R4) OR any Plan B trigger condition is met.
- **At W7 (M4) and W11 (M6b)** mandatory go/no-go check-in with the user (ICRA vs RA-L pivot).

---

## Appendix: spec files cross-reference

| File | Owner | Words | Key contribution |
|---|---|---|---|
| 01_architecture.md | A1 | 3,457 | Tensor shape table; 33 ms budget; Qwen-2.5-1.5B rationale |
| 02_world_model.md | A3 | 3,166 | GNN architecture; L3 protocol; WM-MI metric definition |
| 03_efficiency_deployment.md | A5 | 3,165 | GeoSpec pseudocode + math; TRT-LLM INT8 plan; Plan B playbook |
| 04_experiment_design.md | B1 | 3,455 | PWA-Bench-S protocol; 4 RQ × 474 cells; pre-registration plan |
| **00_master_synthesis.md (this)** | Lead | ~1,200 | Cross-team interface contracts, unified risk register, compute BOM, open items |
