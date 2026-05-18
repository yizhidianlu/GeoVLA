#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read runs/gate1/variant_A/eval.json and variant_B/eval.json, render a verdict."""
import json, sys
from pathlib import Path


def _verdict(delta_pp: float) -> str:
    if delta_pp >= 3.0:
        return "PASS"
    if delta_pp <= -2.0:
        return "REJECT"
    return "MARGINAL"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/GeoVLA/runs/gate1")
    a = json.loads((root / "variant_A" / "eval.json").read_text())
    summary = {"variant_A_rgb_only": a}
    # B is optional (some runs only have A vs C)
    b_path = root / "variant_B" / "eval.json"
    c_path = root / "variant_C" / "eval.json"
    if b_path.exists():
        b = json.loads(b_path.read_text())
        delta_b = (b["success_rate"] - a["success_rate"]) * 100
        summary["variant_B_rgb_plus_proprio"] = b
        summary["delta_B_minus_A_pp"] = round(delta_b, 2)
        summary["verdict_B_vs_A"] = _verdict(delta_b)
    if c_path.exists():
        c = json.loads(c_path.read_text())
        delta_c = (c["success_rate"] - a["success_rate"]) * 100
        summary["variant_C_rgb_plus_depth"] = c
        summary["delta_C_minus_A_pp"] = round(delta_c, 2)
        summary["verdict_C_vs_A"] = _verdict(delta_c)
    # legacy keys for old reporting
    if "verdict_B_vs_A" in summary:
        summary["delta_pp"] = summary["delta_B_minus_A_pp"]
        summary["gate_verdict"] = summary["verdict_B_vs_A"]
    summary["notes"] = (
        "Variant A=RGB-only, B=RGB+15-d proprio (weak 3D proxy), "
        "C=RGB+ground-truth depth (strong 3D proxy). H1 PASS requires C to outperform A."
    )
    (root / "gate1_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
