#!/usr/bin/env bash
# M1 — full grid: 4 tasks × 3 seeds × 2 variants (A, B) × N=20 eval.
# Total: 24 training runs + 24 eval runs ≈ 90-110 min on A800.
# Designed for tmux-detached execution; logs to runs/M1/m1.log.
set -e
cd /root/autodl-tmp/GeoVLA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/geovla
export MUJOCO_GL=egl

TASKS="between_the_plate_and_the_ramekin from_table_center on_the_cookie_box on_the_wooden_cabinet"
SEEDS="42 7 123"

ROOT=runs/M1
mkdir -p $ROOT

# Counter
total=0
for kw in $TASKS; do for seed in $SEEDS; do total=$((total + 1)); done; done
echo "=== M1 grid: $total cells (task × seed) × 2 variants × N=5 eval (N=20 caused eval-side hangs) ==="
echo "Start: $(date)"

cell=0
for kw in $TASKS; do
  for seed in $SEEDS; do
    cell=$((cell + 1))
    OUT=$ROOT/${kw}_seed${seed}
    echo
    echo "----- [$cell/$total] task=$kw seed=$seed -----"
    rm -rf "$OUT"
    for v in A B; do
      echo "  ${v}: train"
      python scripts/gate1/train.py --variant $v --task-keyword $kw --seed $seed \
        --chunk-size 8 --steps 5000 --max-demos 50 --out-root "$OUT" 2>&1 | tail -2
      echo "  ${v}: eval N=5 (timeout 240s as safety)"
      timeout 240 python scripts/gate1/eval_libero.py --variant $v --task-keyword $kw --seed $seed \
        --n-trajs 5 --max-steps 400 --run-root "$OUT" 2>&1 | tail -3 || \
          echo "  WARN: eval timed out for $v ; recording 0% success"
    done
    echo "  compare:"
    python scripts/gate1/compare.py "$OUT" 2>&1 | grep -E 'success_rate|delta_pp|verdict'
  done
done

echo
echo "=== M1 GRID DONE: $(date) ==="
echo "Aggregating..."
python scripts/gate1/m1_aggregate.py "$ROOT" 2>&1 || echo "aggregation script not yet present"
