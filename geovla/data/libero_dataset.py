# -*- coding: utf-8 -*-
"""LIBERO HDF5 dataset loader for Gate-1 smell test.

Each LIBERO demo is (T, *) trajectory:
  actions:           (T, 7)  float64
  agentview_rgb:     (T, 128, 128, 3) uint8
  eye_in_hand_rgb:   (T, 128, 128, 3) uint8
  ee_pos:            (T, 3) float64
  ee_ori:            (T, 3) float64
  joint_states:      (T, 7) float64
  gripper_states:    (T, 2) float64

For Gate-1 we sample individual (s_t, a_t) transitions across all demos in one task,
no temporal context yet.  Action chunk size = 1 (single-step regression).
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# 15-dim proprioception vector — "weakest form of explicit 3D awareness"
PROPRIO_KEYS = ("ee_pos", "ee_ori", "joint_states", "gripper_states")
PROPRIO_DIM = 3 + 3 + 7 + 2  # = 15


def _list_task_hdf5s(root: str | Path, suite: str = "libero_spatial") -> list[Path]:
    root = Path(root)
    if (root / suite).is_dir():
        # nested layout
        paths = sorted((root / suite).glob("*.hdf5"))
    else:
        paths = sorted(root.glob("*.hdf5"))
    return paths


class LiberoBCDataset(Dataset):
    """Behaviour-cloning dataset: returns (image_chw, proprio_15, action_7).

    Args:
        hdf5_path: a single LIBERO task's hdf5 file
        max_demos: cap on number of demos used (None = all)
        normalise_actions: whether to subtract train-set mean / divide by std
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        max_demos: Optional[int] = None,
        normalise_actions: bool = True,
        image_size: int = 128,
        chunk_size: int = 1,
        load_depth: bool = False,       # if True, read sibling <stem>_depth.npz
    ):
        self.path = Path(hdf5_path)
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.load_depth = load_depth
        self.depths: list[np.ndarray] = []  # filled if load_depth

        # Pre-read all demo trajectories into memory (50 demos × ~100 steps × ~150 KB image
        # ≈ 750 MB; fits comfortably in 1TB RAM).  We avoid per-getitem hdf5 reads.
        self.images: list[np.ndarray] = []     # uint8 (T,128,128,3)
        self.proprios: list[np.ndarray] = []   # float32 (T,15)
        self.actions: list[np.ndarray] = []    # float32 (T,7)
        self.demo_starts: list[int] = []       # cumulative index of first step of each demo

        running = 0
        depth_cache = None
        if load_depth:
            depth_path = self.path.parent / (self.path.stem + "_depth.npz")
            if not depth_path.exists():
                raise FileNotFoundError(
                    f"depth cache not found: {depth_path}\n"
                    f"Run: python scripts/gate1/render_depth_cache.py --task-keyword <keyword>")
            depth_cache = np.load(depth_path)
        with h5py.File(self.path, "r") as h:
            keys = sorted([k for k in h["data"].keys() if k.startswith("demo_")],
                          key=lambda s: int(s.split("_")[-1]))
            if max_demos:
                keys = keys[:max_demos]
            for k in keys:
                d = h["data"][k]
                img = d["obs"]["agentview_rgb"][:]                  # uint8 (T,128,128,3)
                prop = np.concatenate([d["obs"][n][:] for n in PROPRIO_KEYS], axis=1)
                prop = prop.astype(np.float32)
                act = d["actions"][:].astype(np.float32)            # (T,7)
                self.images.append(img)
                self.proprios.append(prop)
                self.actions.append(act)
                if load_depth:
                    if k not in depth_cache:
                        raise KeyError(f"depth cache missing {k}")
                    self.depths.append(depth_cache[k].astype(np.float32))
                self.demo_starts.append(running)
                running += img.shape[0]
        self.total = running
        if load_depth:
            # depth normalisation (z-score over all valid depths)
            all_d = np.concatenate([d.reshape(-1) for d in self.depths], axis=0)
            self.depth_mean = float(all_d.mean())
            self.depth_std = float(max(all_d.std(), 1e-3))
        else:
            self.depth_mean = 0.0
            self.depth_std = 1.0

        # action normalisation
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)
        if normalise_actions:
            stacked = np.concatenate(self.actions, axis=0)
            self.action_mean = stacked.mean(0).astype(np.float32)
            self.action_std = stacked.std(0).clip(min=1e-3).astype(np.float32)

        # proprio normalisation
        prop_stacked = np.concatenate(self.proprios, axis=0)
        self.proprio_mean = prop_stacked.mean(0).astype(np.float32)
        self.proprio_std = prop_stacked.std(0).clip(min=1e-3).astype(np.float32)

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> tuple[int, int]:
        """Given a global step idx, return (demo_idx, step_idx)."""
        for d, start in enumerate(self.demo_starts):
            if d == len(self.demo_starts) - 1:
                return d, idx - start
            if idx < self.demo_starts[d + 1]:
                return d, idx - start
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        d, t = self._locate(idx)
        img = self.images[d][t]                                       # uint8 (128,128,3)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,128,128)
        prop = self.proprios[d][t]                                    # (15,)
        prop = (prop - self.proprio_mean) / self.proprio_std
        # action chunk [chunk_size, 7] — pad with last action at episode end
        T = self.actions[d].shape[0]
        end = min(t + self.chunk_size, T)
        chunk = self.actions[d][t:end]
        if chunk.shape[0] < self.chunk_size:
            pad = np.repeat(chunk[-1:], self.chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk = (chunk - self.action_mean) / self.action_std
        # optional depth channel
        if self.load_depth:
            dep = self.depths[d][t]                                   # (128,128) float
            dep = (dep - self.depth_mean) / self.depth_std
            dep_t = torch.from_numpy(dep).unsqueeze(0).float()        # (1,128,128)
            img = torch.cat([img, dep_t], dim=0)                      # (4,128,128)
        return (img,
                torch.from_numpy(prop),
                torch.from_numpy(chunk))

    def denormalise_action(self, a: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.action_mean, device=a.device)
        s = torch.tensor(self.action_std, device=a.device)
        return a * s + m

    def normalise_proprio(self, p: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.proprio_mean, device=p.device)
        s = torch.tensor(self.proprio_std, device=p.device)
        return (p - m) / s


def find_libero_spatial_task(
    suite_root: str | Path = "/root/autodl-tmp/datasets/libero_spatial",
    task_keyword: str = "between_the_plate_and_the_ramekin",
) -> Path:
    """Pick one specific LIBERO-Spatial task by keyword."""
    root = Path(suite_root)
    candidates = list(root.glob(f"*{task_keyword}*.hdf5"))
    if not candidates:
        candidates = sorted(root.glob("*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"no hdf5 in {suite_root}")
    return candidates[0]
