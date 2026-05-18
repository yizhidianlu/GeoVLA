# W1-D1 Closeout — 2026-05-18

**Status: COMPLETE.** All Day-1 milestones achieved + one significant unexpected positive (A800 proxy benchmarks).

---

## Milestones delivered

| # | Milestone | Evidence |
|---|---|---|
| M0 | SSH + GPU侦察 | A800 80GB, 1 TB RAM, 112 cores Xeon Gold, CUDA 13 driver |
| M1 | 数据盘策略决定 | svla (160 G) 完全冻结；pssa-vla / LIBERO / SAM-2 保留；释放 22 G |
| M2 | GitHub repo 上线 | https://github.com/yizhidianlu/GeoVLA (3 commits, 22 files) |
| M3 | Overleaf 同步 | `~/Dropbox/Apps/Overleaf/3DWAM/` — IEEEtran main.tex + 8 sections + 85-entry refs.bib + 5 specs |
| M4 | Conda env (geovla) | 6.9 GB, cloned from pssa-vla in 23.6 s — bypassed network bottleneck |
| M5 | Smoke test PASS | torch 2.4.1+cu124, A800 BF16 OK, 7 interface dataclasses constructed |
| M6 | 3DGS feasibility PASS | All proxies ≥ 185 FPS — 30 Hz target has 6×–280× headroom |
| M7 | LIBERO eval rail PASS | 6 suites loaded, task instantiated 3.59 s, OffScreenRenderEnv at 43 FPS headless, action_dim = 7 matches `interfaces.D_ACTION` |
| M8 | 监控工具 | `tools/check_remote.py` (summary / watch / logs / smoke) — 用户随时一行命令查状态 |
| M9 | Gate-1 设计 | `docs/gate1_smell_test.md` — 14-day earliest falsifiability protocol |

## Unexpected positives

1. **A800 GPU 比预期强**。所有 3DGS/WM/VLM proxy 在 30 Hz 目标上有 17×–280× 余量
   ⇒ master synthesis §2 的 33 ms 延迟预算很可能是 conservative；实际 GeoVLA-Base 在 A800 上可能跑到 60–100 Hz。
2. **pssa-vla 已装好所有依赖**。conda clone 23.6 秒搞定环境，**省下 1.5 小时安装时间**和 $0 公网下载。
3. **LIBERO action space 与接口完全一致** (action_dim=7 = D_ACTION)，无需 adapter。

## Unexpected obstacles

1. **download.pytorch.org 限速 40 KB/s**。原 env_setup.sh 会跑 5.5 小时。解决：阿里 PyTorch mirror + 直接 conda clone。
2. **pip 直接 URL 模式 hang**（无 443 连接）。解决：放弃 URL，走 conda clone 路径。
3. **PowerShell `cd` 不传到 .NET `WriteAllText`**。本地 scaffold 第一遍漏 6 个 stubs，bash here-doc 补救。

## Stage flow (Day 1 真实路径)

```
SSH probe → 22G cleanup → GitHub init+push → Overleaf sync
   ↓
env_setup.sh v1 (pytorch.org)  ← 40 KB/s 限速，弃用
   ↓
env_setup.sh v2 (Aliyun)        ← pip URL hang，弃用
   ↓
conda clone pssa-vla → geovla   ← 23.6 秒成功 ✓
   ↓
smoke test ✓ → 3DGS bench ✓ → LIBERO rail ✓
```

## Repo state

- **Branch:** main
- **Commits:** 4 (initial scaffold, stubs补丁, Aliyun mirror + sanity scripts, numpy>=1.24<3.0 修复)
- **Tracked files:** 23
- **CI:** none yet (W2 任务)

## Server state

- `/root/autodl-tmp/GeoVLA/` — cloned, env tested, sanity scripts ready
- `/root/autodl-tmp/envs/geovla/` — 6.9 GB, fully functional
- `/root/autodl-tmp/datasets/libero_spatial/` — 5.9 GB ready
- `/root/autodl-tmp/LIBERO/` — installed via pssa-vla editable mount
- `/autodl-fs/data/svla/` — completely untouched (160 G frozen)
- Disk: 21 G used / 50 G total / **30 G free** (was 15 G at start)

## What W1-D2 / W2 looks like

### W1-D2 (Tuesday 2026-05-19)
- Resolve 2 open architecture items (Master synthesis §7):
  - **O1** WM-MI construct validity → decide if A2'-from-scratch retrain needed
  - **O3** Qwen-2.5 vs SmolLM2 license — pull SmolLM2-1.7B sanity checkpoint
- Implement `geovla.perception.gs_feedforward` skeleton — first real module body, can be slow at first; latency optimisation is W3
- Start downloading Qwen-2.5-1.5B-Instruct to `/root/autodl-tmp/hf/`

### W1-D3 — W2 末 (~2026-06-01)
- Build Gate-1 (smell test) per `docs/gate1_smell_test.md`
- Train 2 LoRA variants on 1 LIBERO-Spatial task
- Eval 5 trajectories × 2 variants
- **First go/no-go signal:** ≥ +3 pp success rate for "RGB+3D" variant?

### Risk gate
If Gate-1 REJECTS → immediate Plan B2 pivot to Qwen-2.5-3B + Shallow-π 6-layer truncation.

## Monitoring command for the user

```powershell
cd C:\Users\jielu\Desktop\Academic\WAM
python tools\check_remote.py            # one-shot summary
python tools\check_remote.py --smoke    # rerun smoke test
python tools\check_remote.py --logs -n 50
```

Or live:
```bash
ssh -p 16967 root@connect.nma1.seetacloud.com
nvidia-smi
ls /root/autodl-tmp/GeoVLA/logs/
```

## Sign-off

W1-D1 completed in **single working session**. All sanity gates passed.
Ready to execute W1-D2 + W2 on user signal.
