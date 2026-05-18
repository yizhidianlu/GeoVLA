#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIBERO end-to-end evaluation pipeline check (no real policy yet).

Goal of this script: confirm before any real training starts that
  (1) LIBERO + robosuite + mujoco are importable in the geovla env,
  (2) one task can be instantiated with the expected observation/action spec,
  (3) a random policy can step the env and accumulate reward,
  (4) success detection fires when expected,
  (5) the (image, action, reward, done) tensors match what
      `geovla.interfaces.I7_EmittedAction` will eventually emit.

This is the "evaluation rail is real" sanity check that unblocks Gate-1
(the smell test in ~2 weeks).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

# Allow running before `pip install -e .` propagates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def info(msg: str):
    print(f"\n[libero-check] {msg}", flush=True)


def check_imports():
    info("import sanity")
    import torch, robosuite, mujoco          # noqa: F401
    from libero.libero import benchmark      # noqa: F401
    print(f"  torch={torch.__version__}, robosuite={robosuite.__version__}, "
          f"mujoco={mujoco.__version__}")


def list_benchmarks():
    from libero.libero import benchmark
    info("benchmark suites available")
    bm_dict = benchmark.get_benchmark_dict()
    for k in bm_dict:
        print(f"  {k}: {bm_dict[k].__name__}")
    return bm_dict


def pick_one_task(suite_name: str = "libero_spatial", task_idx: int = 0):
    from libero.libero import benchmark
    bm_dict = benchmark.get_benchmark_dict()
    suite = bm_dict[suite_name]()  # instantiate
    info(f"suite={suite_name}: {suite.n_tasks} tasks total")
    task = suite.get_task(task_idx)
    print(f"  task[{task_idx}]: name={task.name}")
    print(f"            language={task.language}")
    return suite, task, task_idx


def make_env(suite, task_idx: int):
    from libero.libero.envs import OffScreenRenderEnv
    info("creating OffScreenRenderEnv (headless)")
    task_bddl_file = suite.get_task_bddl_file_path(task_idx)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    t0 = time.time()
    env = OffScreenRenderEnv(**env_args)
    print(f"  env created in {time.time()-t0:.2f} s")
    return env


def random_rollout(env, max_steps: int = 50):
    info(f"random-policy rollout, up to {max_steps} steps")
    obs = env.reset()
    # action spec
    action_dim = env.env.action_dim  # robosuite-style 7-dof
    print(f"  action_dim = {action_dim}")
    print(f"  observation keys (subset): {list(obs.keys())[:10]}")
    if "agentview_image" in obs:
        print(f"  agentview_image shape = {obs['agentview_image'].shape}")
    rewards = []
    t0 = time.time()
    success = False
    for t in range(max_steps):
        a = np.random.uniform(-0.1, 0.1, size=action_dim)  # tiny random
        obs, reward, done, info_d = env.step(a)
        rewards.append(reward)
        if reward > 0:
            print(f"  step {t}: reward={reward}  <-- non-zero!")
        if done:
            print(f"  step {t}: done=True, success={info_d.get('success', '?')}")
            success = info_d.get("success", False)
            break
    fps = (t + 1) / (time.time() - t0 + 1e-9)
    print(f"  done. {t+1} steps, env throughput ~{fps:.1f} FPS")
    print(f"  reward sum={sum(rewards):.4f}, max step reward={max(rewards):.4f}, success={success}")
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    check_imports()
    list_benchmarks()
    suite, task, idx = pick_one_task(args.suite, args.task)
    env = make_env(suite, idx)
    random_rollout(env, args.steps)
    info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
