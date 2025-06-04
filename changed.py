#!/usr/bin/env python
import os
import re
import json
import argparse

# ← Change this to exactly match problem #5’s prompt:
TARGET_PROBLEM = (
    "5. Three people, A, B, and C, are hiking together and all want to know how far the nearest town is. "
    "A says: “At least 6 miles.” B says: “At most 5 miles.” C says: “At most 4 miles.” In fact, all three are wrong. "
    "Let $d$ represent the distance from them to the nearest town. Then $d$ belongs to the interval ( ).\n"
    "(A) $(0,4)$\n"
    "(B) $(4,5)$\n"
    "(C) $(4,6)$\n"
    "(D) $(5,6)$\n"
    "(E) $(5,+\\infty)$"
)

def load_all_checkpoints(directory):
    """
    Scan `directory` for all files matching *_step_<N>_train.jsonl,
    and return a dict: step → list of matching record_dicts.
    """
    step_pattern = re.compile(r"_step_(\d+)_train\.jsonl$")
    step_to_list = {}

    for fn in sorted(os.listdir(directory)):
        m = step_pattern.search(fn)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(directory, fn)
        records_for_step = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping invalid JSON at {fn}:{line_num}")
                    continue

                # Compare the "problem" field exactly to TARGET_PROBLEM
                if rec.get("problem", "").strip() == TARGET_PROBLEM:
                    records_for_step.append(rec)

        if records_for_step:
            step_to_list[step] = records_for_step

    return step_to_list

def print_instances(directory):
    step_to_records = load_all_checkpoints(directory)
    if not step_to_records:
        print("No instances of the target problem were found in any checkpoint files.")
        return

    print(f"Found the target problem at these steps: {sorted(step_to_records.keys())}\n")
    for step in sorted(step_to_records.keys()):
        recs = step_to_records[step]
        for idx, rec in enumerate(recs, 1):
            print(f"── Step {step} (instance #{idx}) ─────────────────────────────────────────")
            print(json.dumps(rec, indent=2, ensure_ascii=False))
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print all instances of a specific problem across all checkpoint files."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing checkpoint files named *_step_<N>_train.jsonl"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Directory not found: {args.dir}")
        exit(1)

    print_instances(args.dir)
