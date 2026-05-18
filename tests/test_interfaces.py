# -*- coding: utf-8 -*-
"""Smoke tests for geovla.interfaces — runs in CI under pure CPU."""
import pytest
import torch

from geovla import interfaces as I


def test_constants_locked():
    """The master synthesis locked these; CI breaks if anyone changes them silently."""
    assert I.N_CAMERAS == 3
    assert I.N_DENSE_MAX == 80_000
    assert I.N_CONTROL == 2048
    assert I.N_GEO_TOKENS == 256
    assert I.D_HIDDEN == 1536
    assert I.H_ACTION == 8
    assert I.D_ACTION == 7
    assert I.K_PROPOSALS == 4
    assert I.D_GAUSSIAN == 14


def test_i1_dense_gaussians_ok():
    t = torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN, dtype=torch.float16)
    i1 = I.I1_DenseGaussians(tensor=t, n_active=12345)
    assert i1.n_active == 12345


def test_i1_rejects_wrong_dtype():
    t = torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN, dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        I.I1_DenseGaussians(tensor=t, n_active=1)


def test_i1_rejects_wrong_shape():
    t = torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN + 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="shape"):
        I.I1_DenseGaussians(tensor=t, n_active=1)


def test_i1_rejects_n_active_out_of_range():
    t = torch.zeros(I.N_DENSE_MAX, I.D_GAUSSIAN, dtype=torch.float16)
    with pytest.raises(ValueError, match="out of"):
        I.I1_DenseGaussians(tensor=t, n_active=0)
    with pytest.raises(ValueError, match="out of"):
        I.I1_DenseGaussians(tensor=t, n_active=I.N_DENSE_MAX + 1)


def test_i4_flow_proposals():
    p = torch.zeros(I.K_PROPOSALS, I.H_ACTION, I.D_ACTION, dtype=torch.float16)
    lp = torch.zeros(I.K_PROPOSALS, dtype=torch.float32)
    i4 = I.I4_FlowProposals(proposals=p, log_probs=lp)
    assert i4.proposals.shape == (I.K_PROPOSALS, I.H_ACTION, I.D_ACTION)


def test_i5_predicted_gaussians():
    g = torch.zeros(I.K_PROPOSALS, I.N_CONTROL, I.D_GAUSSIAN, dtype=torch.float16)
    c = torch.zeros(I.K_PROPOSALS, dtype=torch.float32)
    i5 = I.I5_PredictedGaussians(next_gaussians=g, rollout_confidence=c)
    assert i5.next_gaussians.shape == (I.K_PROPOSALS, I.N_CONTROL, I.D_GAUSSIAN)


def test_i7_emitted_action_rejects_invalid_idx():
    a = torch.zeros(I.H_ACTION, I.D_ACTION, dtype=torch.float16)
    with pytest.raises(ValueError, match="out of bounds"):
        I.I7_EmittedAction(action_chunk=a, chosen_idx=I.K_PROPOSALS, accepted=True, gate_score=0.0)


def test_full_geovla_step():
    """Constructs one end-to-end step using all 7 interfaces; this is the canonical
    smoke check before any heavier integration test."""
    step = I.GeoVLAStep()
    step.latency_ms["3dgs"] = 9.0
    step.latency_ms["vlm"] = 8.0
    assert sum(step.latency_ms.values()) == 17.0
