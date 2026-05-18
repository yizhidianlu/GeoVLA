# GeoVLA

**Geometry-aware, World-Model-coupled, Edge-real-time Vision-Language-Action policy.**

A 1.5B-parameter Vision-Language-Action (VLA) model that fills a measured empty cell
in the post-OpenVLA literature: **3D-explicit perception + closed-loop world-model
coupling + ≥30 Hz inference on Jetson AGX Orin-tier compute** — three properties no
released policy provides simultaneously.

> Status: **W1 / D1 — scaffold + baseline reproduction in progress.**
> Target venue: ICRA 2027 (deadline 2026-09-15).
> Companion survey: see `docs/companion_survey.md`.

## Why this exists

Mainstream VLAs (OpenVLA, π₀, GR00T-N1, CogACT, MolmoAct) trade off three things
practitioners actually need to deploy:

1. **3D awareness** is treated as emergent rather than designed-in.
2. **Inference latency** is treated as someone else's problem; most VLAs run at < 10 Hz
   on edge hardware.
3. **World models and policies** evolve in parallel; their integration (WM-VLA) only
   became standard in late 2025 (Genie Envisioner, GigaBrain-0, DUST, GPC, World4RL).

GeoVLA is the smallest VLA that closes all three gaps simultaneously, demonstrated on
LIBERO-Long + L-CALVIN + RLBench-12, with a PWA-Bench-S edge card validated on a cloud
Jetson AGX Orin instance.

## Architecture (see `docs/architecture.md`)

```
  multi-view RGB-D                 language
        │                              │
        ▼                              │
  ┌──────────────┐                     │
  │ 3DGS feed-fwd│  3D Gaussian        │
  │ reconstruction│  primitives (2048) │
  └──────┬───────┘                     │
         │                             │
         ▼  (spatial-bucket pool)      │
  ┌──────────────┐                     │
  │ Geo-tokeniser│  256 geo-tokens     │
  └──────┬───────┘                     │
         │                             │
         ▼                             ▼
  ┌──────────────────────────────────────┐
  │   Qwen-2.5-1.5B-Instruct (LoRA)      │
  │   + TokenTransforming (40 % vis-tok) │
  └──────┬───────────────────────────────┘
         │ hidden state
         ▼
  ┌──────────────┐
  │  Flow head   │  K=4 proposals (8-step action chunk)
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────┐
  │  4D-Gaussian WM      │  predicted next-Gaussian
  │  (GNN, 5 layers,     │  (shared with GeoSpec L_geo)
  │   2048 particles)    │
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │   GeoSpec gate       │  α ∈ ℝ⁸ kinematic+geometric
  │   (logistic, ≥85 %   │  acceptance rule
  │   target acceptance) │
  └──────┬───────────────┘
         ▼
     emitted action chunk
         (target ≥ 30 Hz on Jetson AGX Orin)
```

## Five contributions

| # | Title | Owner module |
|---|---|---|
| C1 | GeoVLA-Base architecture (1.5B + 3DGS + 4D-WM + flow head + L3) | `geovla/{perception,world_model,action}` |
| C2 | GeoSpec geometric speculative decoding (+10 pp acceptance vs KERV) | `geovla/geo_spec/` |
| C3 | GeoVLA-Edge: distill 18→6 layers, MoH, TRT-LLM INT8, ≥30 Hz Jetson | `scripts/deploy/` |
| C4 | PWA-Bench-S evaluation protocol + edge card | `geovla/bench/` |
| C5 | 4×4 acceleration-orthogonality ablation grid | `scripts/ablations/` |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yizhidianlu/GeoVLA.git
cd GeoVLA

# 2. Environment (mirrors AutoDL setup)
bash scripts/env_setup.sh        # creates geovla conda env at $GEOVLA_ENV_PREFIX

# 3. Smoke test — verifies module imports + forward pass
python scripts/smoke_test.py

# 4. LIBERO baseline (reproduces CogACT-1.5B baseline for RQ1.A0)
python scripts/baseline_libero.py --variant A0 --suite spatial
```

## Repo layout

```
GeoVLA/
├── geovla/                        # Python package
│   ├── interfaces.py              # tensor-shape contracts (locked at master synthesis)
│   ├── perception/                # 3DGS feed-forward + spatial-bucket tokeniser
│   ├── world_model/               # 4D-Gaussian GNN + L3 protocol
│   ├── action/                    # flow-matching head (consistency-distillable)
│   ├── geo_spec/                  # GeoSpec speculative decoding
│   ├── bench/                     # PWA-Bench-S
│   └── utils/
├── configs/
│   ├── geovla_base.yaml           # GeoVLA-Base full
│   └── geovla_edge.yaml           # deployment variant
├── scripts/
│   ├── env_setup.sh
│   ├── smoke_test.py
│   ├── baseline_libero.py         # RQ1.A0 CogACT-1.5B reimpl
│   ├── train.py                   # full training loop
│   ├── deploy/                    # TRT-LLM compile + INT8 calibration
│   └── ablations/                 # RQ2..RQ4 driver scripts
├── tests/                         # pytest smoke + interface tests
├── docs/
│   ├── architecture.md            # mirrors specs/01_architecture.md
│   ├── master_synthesis.md        # mirrors specs/00_master_synthesis.md
│   └── companion_survey.md        # short pointer to the 3DWAM survey
└── README.md
```

## License

Apache 2.0 (see `LICENSE`).

## Acknowledgements

Built on the shoulders of: OpenVLA, CogACT, GR00T-N1, π₀, GraspSplats, GS-Dynamics,
DeepVerse, Shallow-π, KERV, OneDP, Token Transforming, Mixture-of-Horizons,
Genie Envisioner, GigaBrain-0. Full credits in companion-survey `refs.bib`.
