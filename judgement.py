"""
This Python script automates “judging” (yes/no correctness) of each model‐generated answer in your 
collection of JSONL inference files by calling the OpenAI Chat API once per answer that hasn’t yet been judged. 

"""
import argparse
import json
import os
import glob
import tempfile
from tqdm import tqdm
from openai import OpenAI

# ——— helper to ask ChatGPT (v1 API) ———————————————————————————
def judge_answer(gen_output: str, ground_truth: str) -> bool:
    prompt = f"""
Ground truth: {ground_truth}

Model output:
{gen_output}

Question: Is the model’s <answer> correct? Reply “yes” or “no” only.
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower().startswith("y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-place judge of inference JSONL files under a chosen directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing inference JSONL files (recursive search)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not set via OPENAI_API_KEY environment variable)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Chat model name to use for judgments (default: gpt-4o)."
    )
    args = parser.parse_args()

    # ——— CONFIG —————————————————————————————————————————————
    api_key_used = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_used:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY env var.")
    client = OpenAI(api_key=api_key_used)
    CHAT_MODEL = args.model

    # ——— PARAMETERS ———————————————————————————————————————————
    INPUT_ROOT = args.input_dir
    PATTERN = "**/Qwen2.5-1.5B-Instruct-SFT_step*_train.jsonl"

    # ——— MAIN ——————————————————————————————————————————————————
    for input_path in sorted(glob.glob(os.path.join(INPUT_ROOT, PATTERN), recursive=True)):
        # 1. Read existing lines
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 2. Parse all records
        records = [json.loads(line) for line in lines]

        # 3. Build a map from (problem, gold_answer) → record
        existing_map = {}
        for rec in records:
            key = json.dumps(
                {"problem": rec.get("problem"), "gold_answer": rec.get("gold_answer")},
                sort_keys=True
            )
            existing_map[key] = rec

        # 4. Update records in place
        for rec in tqdm(records, desc=f"Updating {os.path.relpath(input_path, INPUT_ROOT)}"):
            key = json.dumps(
                {"problem": rec.get("problem"), "gold_answer": rec.get("gold_answer")},
                sort_keys=True
            )
            base_rec = existing_map.get(key, {})
            rec.update(base_rec)

            rev = rec.get("step")
            if not rev:
                continue
            judgment_key = f"{rev}_correct"

            if judgment_key not in rec:
                ground = rec["gold_answer"]
                gen = rec.get("output", "")
                rec[judgment_key] = judge_answer(gen, ground)

        # 5. Write updated records back to a temp file, then replace original
        dirpath, filename = os.path.split(input_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirpath, delete=False) as tmpf:
            for rec in records:
                tmpf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            temp_path = tmpf.name

        os.replace(temp_path, input_path)
        print(f"Updated judgments in-place: {input_path}")