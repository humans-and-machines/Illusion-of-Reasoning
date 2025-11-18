#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annotate_recheck_gpt.py
────────────────────────
Recursively scans a directory for JSONL files (e.g., step0000_train.jsonl), queries GPT-4o
for re-check detection, and updates each file with:
    - gpt_recheck: bool
    - gpt_reason: str (exact phrase, or "" if none)

Also creates a `.bak` backup of each JSONL file before modification.

Usage:
    python annotate_recheck_gpt.py --input_root /n/fs/similarity/open-r1/results/od2961/Math220k/GRPO/1.5B
"""

import os, sys, json, argparse, re, shutil
from pathlib import Path
from tqdm import tqdm
from openai import AzureOpenAI, BadRequestError

# ─────────────── Azure GPT-4o Setup ───────────────
client = AzureOpenAI(
    api_key        = "1e30d0e4d7564ba984e8adff48053009",
    azure_endpoint = "https://api-ai-sandbox.princeton.edu/",
    api_version    = "2025-03-01-preview",
)
MODEL = "gpt-4o"

# ─────────────── Prompt Template ───────────────
PROMPT_TMPL = """\
You are a careful annotator for reasoning traces.

Below is a model's full chain-of-thought and answer. Does it contain a **re-check** moment?
A re-check is a sign of doubt or revision, such as "wait", "hold on", "let's verify", "on second thought", etc.

Respond in JSON:
{{
  "has_recheck": true or false,
  "recheck_phrase": "exact phrase that signals re-check, or empty string"
}}

Now annotate:
{output}
"""

def ask_gpt(output: str):
    prompt = PROMPT_TMPL.format(output=output.strip())
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        reply = resp.choices[0].message.content.strip()

        # Attempt strict JSON parse
        if reply.startswith("{") and reply.endswith("}"):
            result = json.loads(reply)
        else:
            match = re.search(r'{.*}', reply, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not find JSON in model reply: {reply}")

        print(f"GPT reply: {result}", file=sys.stderr)
        return bool(result.get("has_recheck", False)), result.get("recheck_phrase", "").strip()
    except BadRequestError as e:
        if getattr(e, "code", None) == "content_filter":
            return False, "<content_filter>"
        raise
    except Exception as e:
        print(f"⚠️ GPT call failed: {e}\nPrompt:\n{prompt}\nReply:\n{reply}", file=sys.stderr)
        return False, "<error>"

def annotate_file(path: Path):
    print(f"→ Annotating: {path}", file=sys.stderr)

    # Backup original file
    backup_path = path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        shutil.copy(path, backup_path)
        print(f"✓ Backup saved to: {backup_path}", file=sys.stderr)

    tmp_path = path.with_suffix(".tmp")

    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=path.name, unit="lines"):
            try:
                row = json.loads(line)
                if "gpt_recheck" in row and "gpt_reason" in row and row["gpt_reason"] not in ["<error>", "<content_filter>"]:
                    fout.write(line)
                    continue
                has_rc, phrase = ask_gpt(row.get("output", ""))
                row["gpt_recheck"] = has_rc
                row["gpt_reason"] = phrase
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Skipping line due to error: {e}", file=sys.stderr)
                continue

    tmp_path.replace(path)
    print(f"✓ Updated: {path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True,
                        help="Top-level directory containing checkpoint subfolders")
    args = parser.parse_args()

    root = Path(args.input_root)
    assert root.exists(), f"Directory not found: {root}"

    files = sorted(root.glob("*/step*_train.jsonl"))
    if not files:
        sys.exit("❌ No step*_train.jsonl files found.")

    for file in files:
        if "analysis" in file.parts:
            continue
        annotate_file(file)

if __name__ == "__main__":
    main()
