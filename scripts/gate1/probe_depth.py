#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe LIBERO env for depth / camera intrinsics support.

Decides whether the W2 full Gate-1 (depth-back-projected pseudo-3DGS tokens)
is feasible WITHOUT having to re-render demos: if the env hands back depth
maps + intrinsics, we can back-project at training time on-the-fly OR
re-roll demos once and cache depth alongside the existing hdf5.
"""
from __future__ import annotations
import numpy as np
import os

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark_dict()["libero_spatial"]()
    bddl = suite.get_task_bddl_file_path(0)

    # Try with depth on
    env_args = dict(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
    )
    print("[probe] creating env with camera_depths=True ...")
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    print(f"[probe] obs keys ({len(obs)}):")
    for k in sorted(obs.keys()):
        v = obs[k]
        if hasattr(v, "shape"):
            print(f"   {k:40s} shape={v.shape}  dtype={v.dtype}  range=[{float(v.min()):.4f}, {float(v.max()):.4f}]")
        else:
            print(f"   {k:40s} = {v}")

    # Look for depth keys specifically
    depth_keys = [k for k in obs if "depth" in k.lower()]
    print(f"\n[probe] depth keys found: {depth_keys}")
    if depth_keys:
        d = obs[depth_keys[0]]
        print(f"   sample depth values: min={float(d.min()):.4f}, max={float(d.max()):.4f}")

    # Look for camera intrinsics / extrinsics in env
    sim = env.env.sim
    print(f"\n[probe] mujoco sim cameras:")
    for i in range(sim.model.ncam):
        name = sim.model.camera_id2name(i)
        print(f"   cam[{i}] = {name}, fov={sim.model.cam_fovy[i]:.2f}")

    # Robosuite-style camera-to-world transform
    try:
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
        K = get_camera_intrinsic_matrix(sim, "agentview", camera_height=128, camera_width=128)
        E = get_camera_extrinsic_matrix(sim, "agentview")
        print(f"\n[probe] agentview intrinsics K (3x3):\n{K}")
        print(f"[probe] agentview extrinsics E (4x4):\n{E}")
    except Exception as e:
        print(f"\n[probe] camera intrinsics fetch failed: {e}")

    # Back-projection sanity: depth (128,128) -> 3D point cloud
    if depth_keys:
        d = obs[depth_keys[0]]
        H, W = d.shape[:2]
        try:
            K = get_camera_intrinsic_matrix(sim, "agentview", camera_height=H, camera_width=W)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            z = d.squeeze()
            # LIBERO/robosuite often returns normalised depth in [0,1] — convert if so
            if z.max() <= 1.5:
                # heuristic: normalised; expand to a plausible metric range
                z_metric = 0.1 + 1.8 * z  # 0.1m to 1.9m
                print("[probe] depth looks normalised; rescaling for back-projection demo")
            else:
                z_metric = z
            X = (u - cx) * z_metric / fx
            Y = (v - cy) * z_metric / fy
            Z = z_metric
            pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
            print(f"[probe] back-projected point cloud: {pts.shape}, "
                  f"X∈[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
                  f"Y∈[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
                  f"Z∈[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")
        except Exception as e:
            print(f"[probe] back-projection failed: {e}")

    env.close()
    print("\n[probe] DONE")


if __name__ == "__main__":
    main()
