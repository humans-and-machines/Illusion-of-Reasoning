#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="List results by domain/model/step/temp.")
    ap.add_argument("root", nargs="?", default="results", help="Results root (default: results)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Missing results root: {root}")
        return

    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            for ckpt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for split_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
                    for temp_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                        rel = temp_dir.relative_to(root)
                        has_raw = any(temp_dir.glob("*.jsonl"))
                        has_shift = any(temp_dir.glob("*shifted.jsonl"))
                        has_recheck = any(temp_dir.glob("*recheck.jsonl"))
                        has_analysis = any(temp_dir.glob("*analysis.jsonl"))
                        print(f"{rel} | raw={has_raw} shift={has_shift} recheck={has_recheck} analysis={has_analysis}")


if __name__ == "__main__":
    main()
