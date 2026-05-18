# Gate-1 Smell Test — Earliest Idea Validation

**Target trigger week:** W2 (≈ 2026-06-01)
**Goal:** Earliest possible falsifiability of GeoVLA's core hypothesis H1
("3D-aware tokens improve a 1.5B VLA at iso-parameter") at minimum cost.

> If this gate fails → pivot to Qwen-2.5-3B + Shallow-π 6-layer truncation
> (Plan B2 in master synthesis §1) before sinking 4 weeks into full RQ1.

---

## Hypothesis under test

> A 1.5B-parameter Qwen-2.5 backbone fine-tuned with LoRA on a single
> LIBERO-Spatial task achieves higher success when input tokens include
> a 3DGS-derived geo-token sequence (256 tokens) than when input is RGB-only.

**Acceptance criterion (single seed):** Δ success rate ≥ +3 pp.
**Hard reject criterion:** Δ ≤ -2 pp on 1-task, 5-trajectory eval.

## Why this design

- **Single task** → fast (no LIBERO suite-wide compute), keeps variance high but
  trends visible.
- **Single seed** → not asking for statistical significance yet — this is a *smell
  test*, not RQ1. ICRA-grade is W6-7.
- **LoRA only** → 1-2 hours per fine-tune × 2 variants = 4 GPU-h on A800.
- **No real WM** → isolate the 3D contribution from WM contribution. Adding WM
  is Gate-2 (~W4) testing the *combined* tax.
- **Depth-back-projected pseudo-3DGS** as a surrogate for the full GraspSplats
  feed-forward — full 3DGS infra not yet integrated by W2.

## Variants (2 fine-tunes)

| Variant | Vision tokens | Geo tokens | Backbone | LoRA |
|---|---|---|---|---|
| **(a) RGB-only** | SigLIP image tokens (~ 729) | none | Qwen-2.5-1.5B-Instruct | r=64 α=128 |
| **(b) RGB + 3D** | SigLIP image tokens (~ 729) | **256 geo-tokens from depth-back-projected pseudo-3DGS** | Qwen-2.5-1.5B-Instruct | r=64 α=128 |

**Geo-token recipe for variant (b):**
1. Extract `agentview_depth` from LIBERO obs (shape ~128×128).
2. Back-project to 3D point cloud using camera intrinsics (~16k points).
3. Voxelise into 4×4×4 grid → 64 buckets; pool point features (mean XYZ + mean
   RGB) per bucket → 4 cells × 64 → 256 geo-tokens.
4. Project via single linear layer (3+3=6 dim → 1536 dim) into the VLM hidden
   space.
5. Prepend geo-tokens before image tokens in the prompt sequence.

This is a **proxy** for the eventual GraspSplats feed-forward — it provides
geometry signal without requiring the full GS infrastructure to be done.

## Action head

For both variants, use a minimal MLP head:
- Input: VLM hidden state at the `[ACT]` anchor token.
- Output: 8-step action chunk × 7-dim = 56 floats, regressed via MSE.

This avoids the flow-matching machinery which is W3-4 work.

## Training recipe

| Item | Value |
|---|---|
| Task | `libero_spatial/pick_up_the_alphabet_soup_and_place_it_in_the_basket` |
| Demo data | 50 expert demos from LIBERO-Spatial (built-in) |
| Train steps | 2000 (≈ 1 epoch over 50 × 8-chunk samples) |
| Batch size | 16 (1.5B + LoRA fits A800 easily) |
| LR | 1e-4 (constant) |
| Precision | BF16 |
| Optimiser | AdamW (β1=0.9, β2=0.95, wd=1e-4) |
| Expected wall-clock | 1.5 h per variant on A800 |

## Evaluation

| Item | Value |
|---|---|
| Eval task | same task as train |
| Trajectories | 5 |
| Max horizon | 200 steps |
| Success metric | LIBERO built-in `info["success"]` |
| Render | OffScreen, 128×128 |
| Output | `runs/gate1_smell/<variant>/eval.json` |

## Compute budget

- 2 fine-tunes × 1.5 h = 3 A800-h
- 2 evals × 0.3 h = 0.6 A800-h
- Total: ~ 4 A800-h, ~ \$0 (uses owned A800)

## Deliverable

Single comparison row in a `gate1_results.json`:

```json
{
  "rgb_only": {"success_rate": 0.40, "mean_reward": 0.65, "wall_clock_h": 1.4},
  "rgb_plus_3d": {"success_rate": 0.60, "mean_reward": 0.81, "wall_clock_h": 1.5},
  "delta_pp": 20.0,
  "gate_verdict": "PASS"
}
```

## What happens after the verdict

### PASS (Δ ≥ +3 pp)
→ Proceed to Gate-2 (full LIBERO-Spatial × WM-off, ~ W3-4) without changes.
→ Optionally tighten LoRA rank or try Qwen-2.5-1.5B fully fine-tuned for the same task.

### MARGINAL (-2 pp < Δ < +3 pp)
→ Diagnose: profile geo-token gradient flow; sweep voxel grid resolution; try
  early-fusion (concat) vs late-fusion (cross-attention) of geo-tokens.
→ Add one more LIBERO-Spatial task to detect single-task variance.

### REJECT (Δ ≤ -2 pp)
→ Likely 1.5B is too small for the 3D-token bandwidth to help.
→ **Immediate Plan B2**: scale to Qwen-2.5-3B + Shallow-π 6-layer truncation.
→ Re-run Gate-1 on the 3B backbone.

## Code skeleton (to be filled in W2)

```
scripts/gate1/
├── train_lora.py          # the LoRA fine-tune driver
├── eval_libero_single.py  # 5-trajectory eval
├── geo_token_recipe.py    # depth back-projection + voxel pool
└── compare.py             # produces gate1_results.json
```

## Risks

| ID | Risk | Mitigation |
|---|---|---|
| G1-R1 | LIBERO depth maps are noisy / inconsistent | Use ground-truth XYZ from `geom_quat` instead of depth back-proj |
| G1-R2 | LoRA on Qwen-2.5-1.5B underfits in 2 K steps | Bump to 5 K steps; budget +1 h |
| G1-R3 | 5 trajectories too noisy to see +3 pp | Bump to 20 trajectories at evaluation time only |
| G1-R4 | Single-task overfit means transfer is questionable | Acknowledged; this is a *smell test*. Gate-2 tests broader claim. |

## When to wake the team up

- If Gate-1 outcome is REJECT → immediately, before W3 work starts.
- If PASS → at the regular Friday weekly digest.
