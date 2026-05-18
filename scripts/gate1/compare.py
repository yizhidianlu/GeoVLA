#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read runs/gate1/variant_A/eval.json and variant_B/eval.json, render a verdict."""
import json, sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/GeoVLA/runs/gate1")
    a = json.loads((root / "variant_A" / "eval.json").read_text())
    b = json.loads((root / "variant_B" / "eval.json").read_text())
    delta_pp = (b["success_rate"] - a["success_rate"]) * 100
    if delta_pp >= 3.0:
        verdict = "PASS"
    elif delta_pp <= -2.0:
        verdict = "REJECT"
    else:
        verdict = "MARGINAL"
    summary = {
        "variant_A_rgb_only": a,
        "variant_B_rgb_plus_proprio": b,
        "delta_pp": round(delta_pp, 2),
        "gate_verdict": verdict,
        "notes": (
            "Gate-1 mini uses 15-d proprio (ee_pos+ee_ori+joint+gripper) as a "
            "stand-in for true 3D-aware Gaussian tokens. PASS only weakly "
            "supports H1; REJECT would be a real warning."
        ),
    }
    out = root / "gate1_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
