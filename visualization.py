#!/usr/bin/env python
import os
import json
import re
import argparse
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Parse command-line arguments
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=(
        "Compute accuracy and 'changed' rate for each checkpoint file in a directory\n"
        "(now supports both GRPO and SFT filenames)."
    )
)
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing files like "
         "Qwen2.5-1.5B-Instruct-GRPO_step_XXXX_train.jsonl "
         "or Qwen2.5-1.5B-Instruct-SFT_step_XXXX_train.jsonl",
)
args = parser.parse_args()
directory = args.directory

# ──────────────────────────────────────────────────────────────────────────────
# Pattern to match both GRPO and SFT filenames, capturing “kind” and “step”
# ──────────────────────────────────────────────────────────────────────────────
#
#   Group 1 → “GRPO” or “SFT”
#   Group 2 → (digits) for the step
#
pattern = re.compile(r"Qwen2\.5-1\.5B-Instruct-(GRPO|SFT)_step_(\d+)_train\.jsonl")

# Dictionaries to hold:
#   (kind, step) → (accuracy, changed_rate)
#
# We will keep two nested dicts, e.g.:
#   stats["GRPO"][step]    = (accuracy, changed_rate)
#   stats["SFT"][step]     = (accuracy, changed_rate)
#
stats = {
    "GRPO": {},
    "SFT": {}
}

# ──────────────────────────────────────────────────────────────────────────────
# 1) Scan and process files in the given directory
# ──────────────────────────────────────────────────────────────────────────────
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if not match:
        continue

    kind = match.group(1)           # Either "GRPO" or "SFT"
    step = int(match.group(2))      # e.g. 50, 100, 150, …

    filepath = os.path.join(directory, filename)

    correct = 0
    changed_true = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Variables to store this record's flags
            this_correct = None
            this_changed = None

            # Look for keys ending in "_correct" and "_changed"
            for k, v in record.items():
                if k.endswith("_correct"):
                    this_correct = v
                elif k.endswith("_changed"):
                    this_changed = v

            # Only count if we found a "_correct" entry
            if this_correct is not None:
                total += 1
                if this_correct is True:
                    correct += 1
                if this_changed is True:
                    changed_true += 1

    # Compute ratios if total > 0
    accuracy = (correct / total) if total > 0 else None
    changed_rate = (changed_true / total) if total > 0 else None

    stats[kind][step] = (accuracy, changed_rate)

# === Insert this snippet for debugging ===
print("⏩ Debug: Which files matched our pattern?")
for filename in sorted(os.listdir(directory)):
    m = pattern.match(filename)
    if m:
        kind, step_str = m.group(1), m.group(2)
        print(f"    → Matched {kind}, step={step_str}:  {filename}")
print("⏩ End of debug-matched files\n")
# =======================================
# ──────────────────────────────────────────────────────────────────────────────
# 2) For each “kind” (GRPO and/or SFT), prepare data and make two plots
# ──────────────────────────────────────────────────────────────────────────────
for kind in ["GRPO", "SFT"]:
    # Only proceed if we actually found any files of this kind
    if not stats[kind]:
        continue
    
    # Debug‐print the raw numbers we’re about to plot
    print(f"\n== {kind} raw stats ==== ")
    for s in sorted(stats[kind]):
        acc, chg = stats[kind][s]
        print(f"  step {s:4d} →  accuracy = {acc},  changed_rate = {chg}")
    print("=" * 30, "\n")

    # Sort steps and extract accuracy/changed
    all_steps = sorted(stats[kind].keys())
    all_accuracies = [stats[kind][s][0] for s in all_steps]
    all_changed    = [stats[kind][s][1] for s in all_steps]

    # Filter out any None entries (in case a checkpoint had zero total)
    filtered_steps = []
    filtered_accuracies = []
    filtered_changed = []
    for s, acc, ch in zip(all_steps, all_accuracies, all_changed):
        # Replace None with 0.0:
        acc_val = 0.0 if (acc is None) else acc
        chg_val = 0.0 if (ch is None) else ch
        filtered_steps.append(s)
        filtered_accuracies.append(acc_val)
        filtered_changed.append(chg_val)

    if not filtered_steps:
        print(f"[WARN] No valid data to plot for {kind}. Skipping.")
        continue

    # ──────────────────────────────────────────────────────────────────────────
    # Plot 1: Accuracy over Steps
    # ──────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_steps, filtered_accuracies, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(f"{kind} Checkpoint Accuracy over Steps")
    plt.grid(True)
    plt.tight_layout()

    accuracy_output = os.path.join(directory, f"{kind.lower()}_accuracy_plot.png")
    plt.savefig(accuracy_output)
    print(f"{kind}: Accuracy plot saved to: {accuracy_output}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Plot 2: Changed Rate over Steps
    # ──────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_steps, filtered_changed, marker='x', color='orange')
    plt.xlabel("Step")
    plt.ylabel("Changed Rate")
    plt.title(f"{kind} Checkpoint Changed Rate over Steps")
    plt.grid(True)
    plt.tight_layout()

    changed_output = os.path.join(directory, f"{kind.lower()}_changed_rate_plot.png")
    plt.savefig(changed_output)
    print(f"{kind}: Changed rate plot saved to: {changed_output}")
    plt.close()
