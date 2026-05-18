#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate-1 eval: roll out a trained policy on the matching LIBERO task.

Usage:
    python scripts/gate1/eval_libero.py --variant A --n-trajs 5
    python scripts/gate1/eval_libero.py --variant B --n-trajs 5

Reads runs/gate1/variant_<X>/model.pt + normalisers.npz, runs closed-loop
on the same LIBERO-Spatial task used at training, writes eval.json.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from geovla.data import find_libero_spatial_task, PROPRIO_DIM
from geovla.models import VariantA, VariantB


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--task-keyword", default="between_the_plate_and_the_ramekin")
    p.add_argument("--n-trajs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/libero_spatial")
    p.add_argument("--run-root", default="/root/autodl-tmp/GeoVLA/runs/gate1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_env(task_file: Path):
    # task_file is /root/.../datasets/libero_spatial/<name>_demo.hdf5
    # Need to map back to the matching bddl file
    from libero.libero.benchmark import get_benchmark_dict
    bm_dict = get_benchmark_dict()
    suite = bm_dict["libero_spatial"]()
    # find task whose name matches the dataset file
    task_name = task_file.stem.replace("_demo", "")
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name]
    if not matches:
        raise RuntimeError(f"no LIBERO-Spatial task with name {task_name}")
    task_idx = matches[0]
    bddl = suite.get_task_bddl_file_path(task_idx)

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=128, camera_widths=128)
    return env, suite.get_task(task_idx), task_idx, suite


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    task_file = find_libero_spatial_task(args.data_root, args.task_keyword)
    run_dir = Path(args.run_root) / f"variant_{args.variant}"
    norm = np.load(run_dir / "normalisers.npz")
    action_mean = torch.tensor(norm["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.tensor(norm["action_std"], dtype=torch.float32, device=args.device)
    proprio_mean = torch.tensor(norm["proprio_mean"], dtype=torch.float32, device=args.device)
    proprio_std = torch.tensor(norm["proprio_std"], dtype=torch.float32, device=args.device)

    if args.variant == "A":
        model = VariantA().to(args.device)
    else:
        model = VariantB().to(args.device)
    ckpt = torch.load(run_dir / "model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    env, task, task_idx, suite = make_env(task_file)
    print(f"[eval] task = {task.name}  (idx {task_idx})")
    print(f"[eval] variant = {args.variant}")

    successes, lengths, t0 = [], [], time.time()
    for ti in range(args.n_trajs):
        # different init by repeatedly stepping reset (LIBERO uses a fixed seed otherwise)
        obs = env.reset()
        # 10-step random warmup to perturb init
        rng = np.random.RandomState(args.seed * 1000 + ti)
        for _ in range(5):
            obs, _, _, _ = env.step(rng.uniform(-0.05, 0.05, size=7))
        succ = False
        steps_used = args.max_steps
        for t in range(args.max_steps):
            img = obs["agentview_rgb"]
            if img is None:
                break
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(args.device) / 255.0
            prop = np.concatenate([obs["ee_pos"], obs["ee_ori"], obs["joint_states"], obs["gripper_states"]])
            prop_t = torch.from_numpy(prop).float().to(args.device).unsqueeze(0)
            prop_t = (prop_t - proprio_mean) / proprio_std

            with torch.no_grad():
                a_norm = model(img, prop_t)
            a = (a_norm * action_std + action_mean).squeeze(0).cpu().numpy()
            a = np.clip(a, -1.0, 1.0)  # action clip to safe range

            obs, reward, done, info = env.step(a)
            if reward > 0 or info.get("success", False):
                succ = info.get("success", reward > 0)
                if succ:
                    steps_used = t + 1
                    break
            if done:
                steps_used = t + 1
                succ = info.get("success", False)
                break
        successes.append(int(succ))
        lengths.append(steps_used)
        print(f"  traj {ti+1}/{args.n_trajs}: success={succ}, steps={steps_used}", flush=True)
    env.close()

    out = {
        "variant": args.variant,
        "task": task.name,
        "n_trajs": args.n_trajs,
        "successes": successes,
        "lengths": lengths,
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "eval_wall_s": time.time() - t0,
    }
    with open(run_dir / "eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
