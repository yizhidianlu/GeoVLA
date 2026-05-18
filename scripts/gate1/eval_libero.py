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
from geovla.models import VariantA, VariantB, VariantC
def _build_d_rgb(**kw):
    from geovla.models.qwen_vla import VariantD_RGB
    return VariantD_RGB(**kw)
def _build_d_rgbp(**kw):
    from geovla.models.qwen_vla import VariantD_RGBProprio
    return VariantD_RGBProprio(**kw)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B", "C", "D", "Dp"], required=True)
    p.add_argument("--task-keyword", default="between_the_plate_and_the_ramekin")
    p.add_argument("--n-trajs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/libero_spatial")
    p.add_argument("--run-root", default="/root/autodl-tmp/GeoVLA/runs/gate1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_env(task_file: Path, need_depth: bool = False):
    from libero.libero.benchmark import get_benchmark_dict
    bm_dict = get_benchmark_dict()
    suite = bm_dict["libero_spatial"]()
    task_name = task_file.stem.replace("_demo", "")
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name]
    if not matches:
        raise RuntimeError(f"no LIBERO-Spatial task with name {task_name}")
    task_idx = matches[0]
    bddl = suite.get_task_bddl_file_path(task_idx)

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=128, camera_widths=128,
                             camera_depths=need_depth)
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

    ckpt = torch.load(run_dir / "model.pt", map_location=args.device, weights_only=False)
    chunk_size = ckpt["args"].get("chunk_size", 1)
    if args.variant == "A":
        model = VariantA(chunk_size=chunk_size).to(args.device)
    elif args.variant == "B":
        model = VariantB(chunk_size=chunk_size).to(args.device)
    elif args.variant == "C":
        model = VariantC(chunk_size=chunk_size).to(args.device)
    elif args.variant == "D":
        model = _build_d_rgb(chunk_size=chunk_size).to(args.device)
    elif args.variant == "Dp":
        model = _build_d_rgbp(chunk_size=chunk_size).to(args.device)
    else:
        raise ValueError(args.variant)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    depth_mean = float(norm.get("depth_mean", 0.0))
    depth_std = float(norm.get("depth_std", 1.0))
    print(f"[eval] chunk_size = {chunk_size}, variant = {args.variant}")

    env, task, task_idx, suite = make_env(task_file, need_depth=(args.variant == "C"))
    print(f"[eval] task = {task.name}  (idx {task_idx})")

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
        # Action chunking: predict once, execute K steps, then re-predict
        # (receding-horizon style — execute exec_horizon of the K predicted actions)
        exec_horizon = max(1, chunk_size // 2)
        t = 0
        while t < args.max_steps:
            # robosuite env uses different obs keys than the demo hdf5 — map them.
            img = obs.get("agentview_image", obs.get("agentview_rgb"))
            if img is None:
                break
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(args.device) / 255.0
            if args.variant == "C":
                d = obs.get("agentview_depth")
                if d is None:
                    raise RuntimeError("variant C requires agentview_depth from env")
                d = d.squeeze().astype(np.float32)
                d = (d - depth_mean) / depth_std
                d_t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(args.device)
                img_t = torch.cat([img_t, d_t], dim=1)  # (1,4,128,128)

            ee_pos = obs.get("robot0_eef_pos", obs.get("ee_pos"))
            quat = obs.get("robot0_eef_quat")
            if quat is not None:
                from scipy.spatial.transform import Rotation
                ee_ori = Rotation.from_quat(quat).as_euler("xyz")
            else:
                ee_ori = obs["ee_ori"]
            joint = obs.get("robot0_joint_pos", obs.get("joint_states"))
            gripper = obs.get("robot0_gripper_qpos", obs.get("gripper_states"))
            prop = np.concatenate([ee_pos, ee_ori, joint, gripper])
            prop_t = torch.from_numpy(prop).float().to(args.device).unsqueeze(0)
            prop_t = (prop_t - proprio_mean) / proprio_std

            with torch.no_grad():
                a_norm = model(img_t, prop_t)                  # (1, K, 7)
            a_chunk = (a_norm * action_std + action_mean).squeeze(0).cpu().numpy()  # (K, 7)
            a_chunk = np.clip(a_chunk, -1.0, 1.0)

            # execute exec_horizon actions before re-predicting
            broke = False
            for k in range(exec_horizon):
                if t >= args.max_steps:
                    break
                obs, reward, done, info = env.step(a_chunk[k])
                t += 1
                if reward > 0 or info.get("success", False):
                    succ = info.get("success", reward > 0)
                    if succ:
                        steps_used = t
                        broke = True
                        break
                if done:
                    steps_used = t
                    succ = info.get("success", False)
                    broke = True
                    break
            if broke:
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
