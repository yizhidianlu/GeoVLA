#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate M1 grid results across 4 tasks × 3 seeds × 2 variants.

Reads runs/M1/<task>_seed<S>/{variant_A,variant_B}/{log.json, eval.json}
Emits runs/M1/m1_summary.{json,md} with per-cell + aggregated stats.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import statistics as stat


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/GeoVLA/runs/M1")
    cells = sorted(root.glob("*_seed*"))
    if not cells:
        print("no cells found"); return

    rows = []   # per-cell rows
    for cell in cells:
        name = cell.name
        # parse task + seed
        ts = name.rsplit("_seed", 1)
        if len(ts) != 2:
            continue
        task, seed = ts
        seed = int(seed)
        row = {"task": task, "seed": seed}
        for v in ("A", "B"):
            evf = cell / f"variant_{v}" / "eval.json"
            logf = cell / f"variant_{v}" / "log.json"
            if not (evf.exists() and logf.exists()):
                row[v] = None; continue
            ev = json.loads(evf.read_text())
            lg = json.loads(logf.read_text())
            row[v] = {
                "success_rate": ev["success_rate"],
                "mean_length": ev["mean_length"],
                "final_val_loss": lg["history"]["val_loss"][-1],
                "final_train_loss": lg["history"]["train_loss"][-1],
            }
        if row.get("A") and row.get("B"):
            row["delta_success_pp"] = (row["B"]["success_rate"] - row["A"]["success_rate"]) * 100
            row["delta_val_loss_pct"] = (row["B"]["final_val_loss"] / row["A"]["final_val_loss"] - 1) * 100
        rows.append(row)

    # ---- aggregations ----
    deltas_succ = [r["delta_success_pp"] for r in rows if r.get("delta_success_pp") is not None]
    deltas_vloss = [r["delta_val_loss_pct"] for r in rows if r.get("delta_val_loss_pct") is not None]

    by_task = defaultdict(list)
    for r in rows:
        if r.get("delta_success_pp") is not None:
            by_task[r["task"]].append(r["delta_success_pp"])

    summary = {
        "n_cells": len(rows),
        "n_complete_cells": len(deltas_succ),
        "delta_success_pp": {
            "mean": stat.mean(deltas_succ) if deltas_succ else None,
            "stdev": stat.stdev(deltas_succ) if len(deltas_succ) > 1 else None,
            "median": stat.median(deltas_succ) if deltas_succ else None,
            "min": min(deltas_succ) if deltas_succ else None,
            "max": max(deltas_succ) if deltas_succ else None,
            "n_positive": sum(1 for d in deltas_succ if d > 0),
            "n_zero": sum(1 for d in deltas_succ if d == 0),
            "n_negative": sum(1 for d in deltas_succ if d < 0),
        },
        "delta_val_loss_pct": {
            "mean": stat.mean(deltas_vloss) if deltas_vloss else None,
            "stdev": stat.stdev(deltas_vloss) if len(deltas_vloss) > 1 else None,
            "median": stat.median(deltas_vloss) if deltas_vloss else None,
            "n_lower_B": sum(1 for d in deltas_vloss if d < 0),
            "n_equal_or_higher": sum(1 for d in deltas_vloss if d >= 0),
        },
        "by_task_delta_success_pp_mean": {t: stat.mean(v) for t, v in by_task.items()},
        "rows": rows,
    }

    (root / "m1_summary.json").write_text(json.dumps(summary, indent=2))

    md = [f"# M1 grid summary", "",
          f"- Cells: {summary['n_complete_cells']} / {summary['n_cells']}",
          "",
          "## Aggregate Δsuccess (B-A, pp)",
          f"- mean: **{summary['delta_success_pp']['mean']:.2f}**",
          f"- stdev: {summary['delta_success_pp']['stdev']:.2f}" if summary['delta_success_pp']['stdev'] is not None else "",
          f"- median: {summary['delta_success_pp']['median']:.2f}",
          f"- range: [{summary['delta_success_pp']['min']:.2f}, {summary['delta_success_pp']['max']:.2f}]",
          f"- n positive / zero / negative: {summary['delta_success_pp']['n_positive']} / {summary['delta_success_pp']['n_zero']} / {summary['delta_success_pp']['n_negative']}",
          "",
          "## Aggregate Δval_loss (B vs A, %)",
          f"- mean: **{summary['delta_val_loss_pct']['mean']:.2f}**",
          f"- stdev: {summary['delta_val_loss_pct']['stdev']:.2f}" if summary['delta_val_loss_pct']['stdev'] is not None else "",
          f"- median: {summary['delta_val_loss_pct']['median']:.2f}",
          f"- B lower than A: {summary['delta_val_loss_pct']['n_lower_B']} / {len(deltas_vloss)} cells",
          "",
          "## Per-task mean Δsuccess",
          ]
    for t, m in summary["by_task_delta_success_pp_mean"].items():
        md.append(f"- {t}: {m:+.2f} pp")
    md.append("")
    md.append("## Per-cell table")
    md.append("| task | seed | A success | B success | Δ pp | val A | val B | Δ val % |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("A") and r.get("B"):
            md.append(f"| {r['task']} | {r['seed']} | {r['A']['success_rate']:.2f} | {r['B']['success_rate']:.2f} | "
                      f"{r['delta_success_pp']:+.1f} | {r['A']['final_val_loss']:.3f} | {r['B']['final_val_loss']:.3f} | "
                      f"{r['delta_val_loss_pct']:+.1f} |")
    (root / "m1_summary.md").write_text("\n".join(md))
    print(f"wrote {root / 'm1_summary.json'} and m1_summary.md")
    print(f"\n=== KEY NUMBERS ===")
    print(f"  Δ success (B-A) pp:  mean {summary['delta_success_pp']['mean']:+.2f}  "
          f"stdev {summary['delta_success_pp']['stdev']:.2f if summary['delta_success_pp']['stdev'] is not None else 0}")
    print(f"  Δ val_loss (B-A) %:  mean {summary['delta_val_loss_pct']['mean']:+.2f}  "
          f"B-lower-in {summary['delta_val_loss_pct']['n_lower_B']}/{len(deltas_vloss)}")


if __name__ == "__main__":
    main()
