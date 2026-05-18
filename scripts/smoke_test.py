#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeoVLA smoke test — verifies env + interface contracts before any heavy work."""
import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .` finishing first
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print(f"=== Python: {sys.version.split()[0]} ===")
print(f"=== GeoVLA repo: {Path(__file__).resolve().parent.parent} ===\n")


def check_torch():
    import torch
    print(f"[OK] torch {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA device 0: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"     compute {props.major}.{props.minor}, "
              f"mem {props.total_memory / 1e9:.1f} GB, "
              f"SMs {props.multi_processor_count}")
        # Quick BF16 sanity
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
        y = (x @ x).sum().item()
        print(f"[OK] bfloat16 matmul finished: {y:.2f}")


def check_interfaces():
    import torch
    from geovla import interfaces as I
    print(f"\n=== Interfaces (locked at master synthesis) ===")
    print(f"  N_DENSE_MAX={I.N_DENSE_MAX}, N_CONTROL={I.N_CONTROL}")
    print(f"  N_GEO_TOKENS={I.N_GEO_TOKENS}, D_HIDDEN={I.D_HIDDEN}")
    print(f"  H_ACTION={I.H_ACTION}, D_ACTION={I.D_ACTION}, K_PROPOSALS={I.K_PROPOSALS}")

    # Fabricate one full GeoVLA step and validate every interface
    i1 = I.I1_DenseGaussians(
        tensor=torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN, dtype=torch.float16),
        n_active=12_345,
    )
    i2 = I.I2_GeoTokens(
        tokens=torch.zeros(1, I.N_GEO_TOKENS, I.D_HIDDEN, dtype=torch.bfloat16),
    )
    T_vlm = 256 + 729 + 32 + 1  # geo + SigLIP image + lang + [ACT]
    i3 = I.I3_VLMHidden(
        hidden=torch.zeros(1, T_vlm, I.D_HIDDEN, dtype=torch.bfloat16),
        act_anchor_idx=T_vlm - 1,
    )
    i4 = I.I4_FlowProposals(
        proposals=torch.zeros(I.K_PROPOSALS, I.H_ACTION, I.D_ACTION, dtype=torch.float16),
        log_probs=torch.zeros(I.K_PROPOSALS, dtype=torch.float32),
    )
    i5 = I.I5_PredictedGaussians(
        next_gaussians=torch.zeros(I.K_PROPOSALS, I.N_CONTROL, I.D_GAUSSIAN, dtype=torch.float16),
        rollout_confidence=torch.zeros(I.K_PROPOSALS, dtype=torch.float32),
    )
    i6 = I.I6_CachedFeatures(
        cached_tokens=torch.zeros(1, I.N_GEO_TOKENS, I.D_HIDDEN, dtype=torch.bfloat16),
    )
    i7 = I.I7_EmittedAction(
        action_chunk=torch.zeros(I.H_ACTION, I.D_ACTION, dtype=torch.float16),
        chosen_idx=0,
        accepted=True,
        gate_score=0.0,
    )

    step = I.GeoVLAStep(i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, i6=i6, i7=i7,
                        latency_ms={"3dgs": 9.0, "vlm": 8.0, "wm": 4.2, "geospec": 1.8})
    print(f"[OK] all 7 interface dataclasses constructed; latency sketch: {step.latency_ms}")


def main():
    check_torch()
    check_interfaces()
    print("\n=== Smoke test PASSED ===")


if __name__ == "__main__":
    main()
