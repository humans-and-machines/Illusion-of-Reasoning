#!/usr/bin/env python3
"""CLI utility to list result directories and their JSONL artifacts."""

import argparse
from pathlib import Path


def main() -> None:
    """Print a summary of available results grouped by domain/model/step/temp."""
    arg_parser = argparse.ArgumentParser(
        description="List results by domain/model/step/temp."
    )
    arg_parser.add_argument(
        "root",
        nargs="?",
        default="results",
        help="Results root (default: results)",
    )
    args = arg_parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Missing results root: {root}")
        return

    for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            for ckpt_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                for split_dir in sorted(path for path in ckpt_dir.iterdir() if path.is_dir()):
                    for temp_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                        rel = temp_dir.relative_to(root)
                        has_raw = any(temp_dir.glob("*.jsonl"))
                        has_shift = any(temp_dir.glob("*shifted.jsonl"))
                        has_recheck = any(temp_dir.glob("*recheck.jsonl"))
                        has_analysis = any(temp_dir.glob("*analysis.jsonl"))
                        summary = (
                            f"{rel} | raw={has_raw} shift={has_shift} "
                            f"recheck={has_recheck} analysis={has_analysis}"
                        )
                        print(summary)


if __name__ == "__main__":
    main()
