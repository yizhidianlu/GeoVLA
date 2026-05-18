#!/usr/bin/env bash
# W2 VariantC pipeline — depth render + train A & C + eval N=10 + compare.
# Run via: tmux new-session -d -s w2run "bash scripts/gate1/w2_pipeline.sh 2>&1 | tee runs/gate1_depth/pipeline.log"
set -e
cd /root/autodl-tmp/GeoVLA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/geovla
export MUJOCO_GL=egl

OUT=runs/gate1_depth
KW=between_the_plate_and_the_ramekin
mkdir -p "$OUT"
rm -rf "$OUT/variant_A" "$OUT/variant_C" "$OUT/gate1_results.json"

echo "=== 1. RENDER DEPTH (timeout 600s) ==="
# 50 demos has been observed to hang around demo 20 due to EGL context leak;
# run with a hard timeout and proceed with whatever depth got saved (partial-save every 5 demos).
timeout 600 python scripts/gate1/render_depth_cache.py --task-keyword "$KW" --max-demos 30 2>&1 || \
    echo "[w2-pipeline] render exited (timeout/failure) -- continuing with partial cache"
DEPTH_FILE=/root/autodl-tmp/datasets/libero_spatial/pick_up_the_black_bowl_${KW}_and_place_it_on_the_plate_demo_depth.npz
ls -la "$DEPTH_FILE" || { echo "no depth cache produced"; exit 1; }

echo "=== 2. TRAIN A (max-demos matched to depth cache) ==="
# Detect how many demos got cached so train uses the same subset for both variants.
N_CACHED=$(python -c "import numpy as np; print(len(np.load('$DEPTH_FILE').files))")
echo "  cached depth demos = $N_CACHED ; training both variants on the same subset"
python scripts/gate1/train.py --variant A --task-keyword "$KW" --max-demos "$N_CACHED" --chunk-size 8 --steps 5000 --out-root "$OUT" 2>&1

echo "=== 3. TRAIN C ==="
python scripts/gate1/train.py --variant C --task-keyword "$KW" --max-demos "$N_CACHED" --chunk-size 8 --steps 5000 --out-root "$OUT" 2>&1

echo "=== 4. EVAL A N=10 ==="
python scripts/gate1/eval_libero.py --variant A --task-keyword "$KW" --n-trajs 10 --max-steps 400 --run-root "$OUT" 2>&1

echo "=== 5. EVAL C N=10 ==="
python scripts/gate1/eval_libero.py --variant C --task-keyword "$KW" --n-trajs 10 --max-steps 400 --run-root "$OUT" 2>&1

echo "=== 6. COMPARE ==="
python scripts/gate1/compare.py "$OUT" 2>&1

echo "=== PIPELINE DONE ==="
