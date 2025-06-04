#!/usr/bin/env python
import os
import re
import json
import argparse
from collections import defaultdict

def load_all_checkpoints(directory):
    """
    Scan `directory` for all files matching *_step_<N>_train.jsonl,
    and return a dict:
      step → { (problem, gold_answer) → record_dict }
    """
    # Regex to capture the step number from filenames like "..._step_50_train.jsonl"
    step_pattern = re.compile(r"_step_(\d+)_train\.jsonl$")

    step_to_records = {}  # step (int) -> dict mapping (problem, gold) -> record

    for fn in sorted(os.listdir(directory)):
        m = step_pattern.search(fn)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(directory, fn)
        rec_map = {}

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

                problem = rec.get("problem", "")
                gold    = rec.get("gold_answer", "")
                if not problem and not gold:
                    continue

                key = (problem, gold)
                rec_map[key] = rec

        step_to_records[step] = rec_map

    return step_to_records

def find_wrong_then_right(step_to_records):
    """
    Given step_to_records: { step -> { (problem,gold) -> record } },
    return a list of keys (problem,gold) for which:
      - at the minimum step, "_correct" == False
      - at the maximum step, "_correct" == True
    """
    all_steps = sorted(step_to_records.keys())
    if not all_steps:
        return []

    first_step = all_steps[0]
    last_step  = all_steps[-1]

    wrong_then_right = []

    # Build sets of keys present in both first and last
    first_keys = set(step_to_records[first_step].keys())
    last_keys  = set(step_to_records[last_step].keys())
    common_keys = first_keys & last_keys

    for key in sorted(common_keys):
        rec_first = step_to_records[first_step][key]
        rec_last  = step_to_records[last_step][key]

        # Find the "<step>_correct" field in each record
        def get_correct_flag(rec):
            for k, v in rec.items():
                if k.endswith("_correct"):
                    return bool(v)
            return None

        corr_first = get_correct_flag(rec_first)
        corr_last  = get_correct_flag(rec_last)

        if (corr_first is False) and (corr_last is True):
            wrong_then_right.append(key)

    return wrong_then_right

def pull_intermediates(directory):
    step_to_records = load_all_checkpoints(directory)
    if not step_to_records:
        print("No checkpoint files found in directory.")
        return

    all_steps = sorted(step_to_records.keys())
    print(f"Found steps: {all_steps}\n")

    candidates = find_wrong_then_right(step_to_records)
    print(f"Total examples WRONG→RIGHT: {len(candidates)}\n")

    if not candidates:
        print("No examples went from wrong in the first step to right in the last step.")
        return

    for idx, key in enumerate(candidates, 1):
        problem, gold = key
        print(f"\n🔄 Example #{idx} changed from WRONG → RIGHT")
        print("-" * 60)
        print("Problem (gold):\n")
        print(problem.strip())
        print("\nGold answer:", gold.strip())
        print()

        # Print each intermediate record in order of ascending step
        for step in all_steps:
            rec_map = step_to_records.get(step, {})
            rec = rec_map.get(key)
            if rec is None:
                # This example did not appear at this step—skip
                continue

            # Identify the “<step>_correct” and “<step>_changed” keys for clarity
            correct_flag = None
            changed_flag = None
            for k, v in rec.items():
                if k.endswith("_correct"):
                    correct_flag = v
                elif k.endswith("_changed"):
                    changed_flag = v

            print(f"Step {step}:  correct = {correct_flag},  changed = {changed_flag}")
            # Optionally, also print the full “output” field or entire JSON blob:
            # print(json.dumps(rec, indent=2, ensure_ascii=False))
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull all intermediate records for examples that were wrong at the first step and right at the last step."
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

    pull_intermediates(args.dir)
