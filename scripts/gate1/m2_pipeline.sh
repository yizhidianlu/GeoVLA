#!/usr/bin/env bash
# M2 — Qwen-2.5-1.5B + LoRA + SigLIP on task 0.
# Variants: D (RGB only) vs Dp (RGB + propio). Same H1 test as Gate-1, real VLA.
# Single task, seed 42, batch 16 (Qwen needs more memory than ResNet18),
# 3000 steps (Qwen+LoRA converges faster than MLP), N=20 eval.
set -e
cd /root/autodl-tmp/GeoVLA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/geovla
export MUJOCO_GL=egl
export HF_HOME=/root/autodl-tmp/hf

OUT=runs/M2/task0_seed42
KW=between_the_plate_and_the_ramekin
mkdir -p "$OUT"

echo "=== M2: Qwen-2.5-1.5B + LoRA on $KW (seed 42) ==="
echo "Start: $(date)"

for v in D Dp; do
  echo
  echo "--- variant $v ---"
  echo "  train"
  python scripts/gate1/train.py --variant $v --task-keyword $KW --seed 42 \
    --chunk-size 8 --steps 3000 --batch 16 --max-demos 50 \
    --out-root "$OUT" 2>&1 | tail -3
  echo "  eval N=20"
  python scripts/gate1/eval_libero.py --variant $v --task-keyword $KW --seed 42 \
    --n-trajs 20 --max-steps 400 --run-root "$OUT" 2>&1 | tail -3
done

echo
echo "=== M2 compare (D vs Dp) ==="
# Compare uses A/B/C labels; we'll rename for clarity but use existing logic:
# move D -> variant_A (baseline RGB) and Dp -> variant_B (with proprio) for compare.py
ln -sfn variant_D "$OUT/variant_A_link"   # informational only
ln -sfn variant_Dp "$OUT/variant_B_link"
python -c "
import json
from pathlib import Path
root = Path('$OUT')
ev_d = json.loads((root / 'variant_D' / 'eval.json').read_text())
ev_dp = json.loads((root / 'variant_Dp' / 'eval.json').read_text())
lg_d = json.loads((root / 'variant_D' / 'log.json').read_text())
lg_dp = json.loads((root / 'variant_Dp' / 'log.json').read_text())
delta_succ = (ev_dp['success_rate'] - ev_d['success_rate']) * 100
delta_val = (lg_dp['history']['val_loss'][-1] / lg_d['history']['val_loss'][-1] - 1) * 100
verdict = 'PASS' if delta_succ >= 3.0 else ('REJECT' if delta_succ <= -2.0 else 'MARGINAL')
out = {
    'M2_qwen_lora_D_rgb_only': ev_d,
    'M2_qwen_lora_Dp_rgb_proprio': ev_dp,
    'final_val_loss_D': lg_d['history']['val_loss'][-1],
    'final_val_loss_Dp': lg_dp['history']['val_loss'][-1],
    'delta_success_pp': round(delta_succ, 2),
    'delta_val_loss_pct': round(delta_val, 2),
    'gate_verdict': verdict,
    'notes': 'M2 on Qwen-2.5-1.5B + LoRA + SigLIP. D=RGB-only, Dp=RGB+15-d propio. '
             'H1 PASS at task-success level requires Dp > D by >=3 pp.',
}
(root / 'm2_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
"

echo
echo "=== M2 DONE: $(date) ==="
