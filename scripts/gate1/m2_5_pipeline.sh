#!/usr/bin/env bash
# M2.5 — OpenVLA-7B + LoRA on task 0. Variants: E (RGB) vs Ep (RGB + propio).
# Heavier than M2 (7B vs 1.5B): batch 8, 2000 steps, ~30 min train per variant
# + N=20 eval (~5-8 min each). Total ≈ 80-100 min.
set -e
cd /root/autodl-tmp/GeoVLA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/geovla
export MUJOCO_GL=egl
export HF_HOME=/root/autodl-tmp/hf

OUT=runs/M2_5/task0_seed42
KW=between_the_plate_and_the_ramekin
mkdir -p "$OUT"

echo "=== M2.5: OpenVLA-7B + LoRA on $KW (seed 42) ==="
echo "Start: $(date)"

for v in E Ep; do
  echo
  echo "--- variant $v (OpenVLA-7B + LoRA) ---"
  echo "  train"
  python scripts/gate1/train.py --variant $v --task-keyword $KW --seed 42 \
    --chunk-size 8 --steps 2000 --batch 8 --max-demos 50 \
    --out-root "$OUT" 2>&1 | tail -3
  echo "  eval N=20"
  python scripts/gate1/eval_libero.py --variant $v --task-keyword $KW --seed 42 \
    --n-trajs 20 --max-steps 400 --run-root "$OUT" 2>&1 | tail -3
done

echo
echo "=== M2.5 compare (E vs Ep) ==="
python -c "
import json
from pathlib import Path
root = Path('$OUT')
ev_e = json.loads((root / 'variant_E' / 'eval.json').read_text())
ev_ep = json.loads((root / 'variant_Ep' / 'eval.json').read_text())
lg_e = json.loads((root / 'variant_E' / 'log.json').read_text())
lg_ep = json.loads((root / 'variant_Ep' / 'log.json').read_text())
delta_succ = (ev_ep['success_rate'] - ev_e['success_rate']) * 100
delta_val = (lg_ep['history']['val_loss'][-1] / lg_e['history']['val_loss'][-1] - 1) * 100
verdict = 'PASS' if delta_succ >= 3.0 else ('REJECT' if delta_succ <= -2.0 else 'MARGINAL')
out = {
    'M2_5_openvla_E_rgb_only': ev_e,
    'M2_5_openvla_Ep_rgb_proprio': ev_ep,
    'final_val_loss_E': lg_e['history']['val_loss'][-1],
    'final_val_loss_Ep': lg_ep['history']['val_loss'][-1],
    'delta_success_pp': round(delta_succ, 2),
    'delta_val_loss_pct': round(delta_val, 2),
    'gate_verdict': verdict,
    'notes': 'M2.5 on OpenVLA-7B (Prismatic VLM, DINOv2+SigLIP+LLaMA-2 7B) + LoRA. '
             'E=RGB-only, Ep=RGB+15-d propio. H1 PASS at task-success requires Ep > E by >=3 pp.',
}
(root / 'm2_5_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
"

echo
echo "=== M2.5 DONE: $(date) ==="
