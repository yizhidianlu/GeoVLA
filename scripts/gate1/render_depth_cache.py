#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache ground-truth depth maps for LIBERO demos.

LIBERO demo hdf5 stores `states` (T, 92) — full mujoco model state per step.
We replay these states (no physics integration needed, just sim.set_state +
forward + render) in an env with `camera_depths=True`, dump the agentview
depth maps to a sibling npz file.

Output: <hdf5_basename>_depth.npz containing one (T, 128, 128) float32 array
per demo, keyed by demo_id (e.g. demo_0, demo_1, ...).

Run once per LIBERO task; subsequent training reads the cache instantly.
"""
from __future__ import annotations

import argparse, os, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def render_depth_for_demo(env, states: np.ndarray) -> np.ndarray:
    """Replay a demo's full state trajectory; return depth maps (T, 128, 128)."""
    env.reset()
    sim = env.env.sim
    depths = np.zeros((states.shape[0], 128, 128), dtype=np.float32)
    for t in range(states.shape[0]):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        obs = env.env._get_observations(force_update=True)
        d = obs.get("agentview_depth")
        if d is None:
            continue
        depths[t] = d.squeeze().astype(np.float32)
    return depths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-keyword", default="between_the_plate_and_the_ramekin")
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/libero_spatial")
    p.add_argument("--max-demos", type=int, default=50)
    p.add_argument("--out-suffix", default="_depth.npz")
    args = p.parse_args()

    from geovla.data import find_libero_spatial_task
    task_file = find_libero_spatial_task(args.data_root, args.task_keyword)
    print(f"[depth-cache] task file: {task_file.name}")

    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    suite = get_benchmark_dict()["libero_spatial"]()
    task_name = task_file.stem.replace("_demo", "")
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name]
    if not matches:
        raise RuntimeError(f"no LIBERO task named {task_name}")
    bddl = suite.get_task_bddl_file_path(matches[0])

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128, camera_widths=128,
        camera_depths=True,
    )
    print(f"[depth-cache] env created")

    out_path = task_file.with_suffix("").with_suffix(args.out_suffix)
    print(f"[depth-cache] -> {out_path}")

    out_dict = {}
    t0 = time.time()
    with h5py.File(task_file, "r") as h:
        keys = sorted([k for k in h["data"].keys() if k.startswith("demo_")],
                      key=lambda s: int(s.split("_")[-1]))[:args.max_demos]
        for i, k in enumerate(keys):
            states = h["data"][k]["states"][:]
            depths = render_depth_for_demo(env, states)
            out_dict[k] = depths
            print(f"  {i+1}/{len(keys)} {k}: states={states.shape} -> "
                  f"depths={depths.shape} (elapsed {time.time()-t0:.1f}s)", flush=True)
    env.close()
    np.savez_compressed(out_path, **out_dict)
    print(f"[depth-cache] saved {len(out_dict)} demos in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
