#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate-1 BC training driver.

Trains either VariantA (RGB-only) or VariantB (RGB+proprio) on a single
LIBERO-Spatial task for a fixed number of optimiser steps.

Usage:
    python scripts/gate1/train.py --variant A --steps 2000 --batch 64
    python scripts/gate1/train.py --variant B --steps 2000 --batch 64

Outputs:
    runs/gate1/<variant>/{model.pt, log.json, normalisers.npz}
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import sys

# Allow running before `pip install -e .` propagates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from geovla.data import LiberoBCDataset, find_libero_spatial_task
from geovla.models import VariantA, VariantB, VariantC, trainable_param_count
# D variants lazy-import (peft + ~3GB Qwen weights)
def _build_d_rgb(**kw):
    from geovla.models.qwen_vla import VariantD_RGB
    return VariantD_RGB(**kw)
def _build_d_rgbp(**kw):
    from geovla.models.qwen_vla import VariantD_RGBProprio
    return VariantD_RGBProprio(**kw)
def _build_e_rgb(**kw):
    from geovla.models.openvla_wrap import VariantE_RGB
    return VariantE_RGB(**kw)
def _build_e_rgbp(**kw):
    from geovla.models.openvla_wrap import VariantE_RGBProprio
    return VariantE_RGBProprio(**kw)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B", "C", "D", "Dp", "E", "Ep"], required=True,
                   help="A=RGB+MLP, B=RGB+propio+MLP, C=RGB+depth+MLP, "
                        "D=RGB+Qwen+LoRA, Dp=RGB+propio+Qwen+LoRA, "
                        "E=RGB+OpenVLA-7B+LoRA, Ep=RGB+propio+OpenVLA-7B+LoRA")
    p.add_argument("--task-keyword", default="between_the_plate_and_the_ramekin")
    p.add_argument("--max-demos", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/libero_spatial")
    p.add_argument("--out-root", default="/root/autodl-tmp/GeoVLA/runs/gate1")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_root) / f"variant_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gate1] variant={args.variant} steps={args.steps} batch={args.batch}")
    print(f"[gate1] output -> {out}")

    # ------------------------------------------------------------------ data
    task_file = find_libero_spatial_task(args.data_root, args.task_keyword)
    print(f"[gate1] task file: {task_file.name}")

    load_depth = (args.variant == "C")
    ds = LiberoBCDataset(task_file, max_demos=args.max_demos,
                         chunk_size=args.chunk_size, load_depth=load_depth)
    print(f"[gate1] dataset: {len(ds)} steps from {len(ds.demo_starts)} demos, "
          f"chunk={args.chunk_size}, depth={load_depth}")

    # save normalisers so eval can mirror them
    np.savez(out / "normalisers.npz",
             action_mean=ds.action_mean, action_std=ds.action_std,
             proprio_mean=ds.proprio_mean, proprio_std=ds.proprio_std,
             depth_mean=np.float32(ds.depth_mean), depth_std=np.float32(ds.depth_std))

    val_n = int(len(ds) * args.val_frac)
    train_n = len(ds) - val_n
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True)

    # ------------------------------------------------------------------ model
    if args.variant == "A":
        model = VariantA(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "B":
        model = VariantB(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "C":
        model = VariantC(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "D":
        model = _build_d_rgb(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "Dp":
        model = _build_d_rgbp(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "E":
        model = _build_e_rgb(chunk_size=args.chunk_size).to(args.device)
    elif args.variant == "Ep":
        model = _build_e_rgbp(chunk_size=args.chunk_size).to(args.device)
    else:
        raise ValueError(args.variant)
    print(f"[gate1] model: {type(model).__name__}, trainable params = {trainable_param_count(model):,}")

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.steps)
    loss_fn = torch.nn.MSELoss()

    # ------------------------------------------------------------------ loop
    history = {"step": [], "train_loss": [], "val_loss": [], "elapsed_s": []}
    t0 = time.time()
    train_iter = iter(train_loader)
    running = 0.0
    for step in range(1, args.steps + 1):
        try:
            img, prop, act = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            img, prop, act = next(train_iter)
        img = img.to(args.device, non_blocking=True)
        prop = prop.to(args.device, non_blocking=True)
        act = act.to(args.device, non_blocking=True)

        pred = model(img, prop)
        loss = loss_fn(pred, act)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        opt.step()
        sched.step()
        running = 0.95 * running + 0.05 * loss.item() if step > 1 else loss.item()

        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps} loss={running:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for img, prop, act in val_loader:
                    img = img.to(args.device); prop = prop.to(args.device); act = act.to(args.device)
                    pred = model(img, prop)
                    vl += loss_fn(pred, act).item() * img.size(0)
                    n += img.size(0)
            vl /= max(n, 1)
            print(f"  [eval] step {step} val_loss={vl:.4f}", flush=True)
            model.train()
            history["step"].append(step)
            history["train_loss"].append(running)
            history["val_loss"].append(vl)
            history["elapsed_s"].append(time.time() - t0)

    # ------------------------------------------------------------------ save
    ckpt = {
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "args": vars(args),
        "history": history,
        "total_elapsed_s": time.time() - t0,
    }
    torch.save(ckpt, out / "model.pt")
    with open(out / "log.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "total_elapsed_s": time.time() - t0,
                   "trainable_params": trainable_param_count(model)}, f, indent=2)
    print(f"[gate1] DONE in {time.time()-t0:.1f} s ; ckpt -> {out/'model.pt'}")


if __name__ == "__main__":
    main()
