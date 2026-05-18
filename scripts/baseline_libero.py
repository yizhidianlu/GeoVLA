# -*- coding: utf-8 -*-
"""baseline_libero.py — placeholder for the RQ1.A0 CogACT-1.5B reimpl baseline.

Filled in during W2 per the master synthesis roadmap.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A0", "A1", "A2", "A3", "A4", "A5"], default="A0")
    p.add_argument("--suite", choices=["spatial", "object", "goal", "long"], default="spatial")
    args = p.parse_args()
    print(f"[baseline_libero] variant={args.variant} suite={args.suite}")
    print("[baseline_libero] STUB -- to be implemented in W2.")


if __name__ == "__main__":
    main()
