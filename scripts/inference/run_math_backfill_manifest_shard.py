#!/usr/bin/env python3
"""
Run math pass2 backfill for a shard of a TSV manifest.

This is intended to be launched under a single multi-GPU Slurm job via `srun`,
where each task (rank) processes a disjoint subset of the manifest lines.

Manifest format (tab-separated, 1 target per line):
  family<TAB>temp<TAB>step<TAB>jsonl_path
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Target:
    family: str
    temp: str
    step: int
    jsonl_path: str


def _parse_manifest_line(line: str) -> Optional[Target]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        return None
    family, temp, step_s, jsonl_path = parts
    family = family.strip()
    temp = temp.strip()
    jsonl_path = jsonl_path.strip()
    if not family or not temp or not step_s.strip() or not jsonl_path:
        return None
    try:
        step = int(step_s)
    except ValueError:
        return None
    return Target(family=family, temp=temp, step=step, jsonl_path=jsonl_path)


def _iter_targets(manifest_path: Path) -> Iterable[Target]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip("\n")
            if not raw.strip():
                continue
            tgt = _parse_manifest_line(raw)
            if tgt is None:
                continue
            yield tgt


def _resolve_model_paths(*, project_root: Path, family: str, step: int) -> tuple[str, str]:
    if family == "1.5B":
        base_model = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer_path = "Qwen/Qwen2.5-1.5B-Instruct"
        ckpt_root = project_root / "artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-math-v1"
    elif family == "7B":
        base_model = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"
        ckpt_root = project_root / "artifacts/models/Qwen2.5-7B-Open-R1-GRPO-math-7b"
    elif family == "Llama8B":
        base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ckpt_root = project_root / "artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2"
    else:
        raise ValueError(f"Unknown family: {family!r}")

    if step == 0:
        return base_model, tokenizer_path
    ckpt_dir = ckpt_root / f"checkpoint-{step}"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")
    return ckpt_dir.as_posix(), tokenizer_path


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to TSV manifest.")
    parser.add_argument("--rank", type=int, required=True, help="0-based rank (e.g., SLURM_PROCID).")
    parser.add_argument("--world_size", type=int, required=True, help="Total tasks (e.g., SLURM_NTASKS).")
    parser.add_argument("--project_root", default=".", help="Repo root (default: current working dir).")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--flush_every", type=int, default=None)
    parser.add_argument("--entropy_mode", default="full", choices=["full", "reconsider", "none"])
    parser.add_argument("--think_cap", type=int, default=750)
    parser.add_argument("--answer_cap", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        raise SystemExit(f"Invalid rank/world_size: rank={args.rank} world_size={args.world_size}")

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    targets = list(_iter_targets(manifest_path))
    assigned = [t for i, t in enumerate(targets) if (i % args.world_size) == args.rank]

    print(
        f"[rank {args.rank}/{args.world_size}] "
        f"manifest={manifest_path} total_targets={len(targets)} assigned={len(assigned)}"
    )
    sys.stdout.flush()

    for idx, t in enumerate(assigned):
        try:
            model_name_or_path, tokenizer_path = _resolve_model_paths(
                project_root=project_root, family=t.family, step=t.step
            )
        except Exception as exc:
            print(f"[rank {args.rank}] [skip] {t.jsonl_path}: {exc}", file=sys.stderr)
            continue

        jsonl_path = Path(t.jsonl_path)
        if not jsonl_path.is_absolute():
            jsonl_path = project_root / jsonl_path
        if not jsonl_path.is_file():
            print(f"[rank {args.rank}] [skip] missing file: {jsonl_path}", file=sys.stderr)
            continue

        cmd: list[str] = [
            sys.executable,
            "-m",
            "src.inference.cli.backfill_math_pass2",
            "--model_name_or_path",
            model_name_or_path,
            "--tokenizer_path",
            tokenizer_path,
            "--input_jsonl",
            jsonl_path.as_posix(),
            "--inplace",
            "--batch_size",
            str(args.batch_size),
            "--temperature",
            str(t.temp),
            "--top_p",
            str(args.top_p),
            "--entropy_mode",
            args.entropy_mode,
            "--think_cap",
            str(args.think_cap),
            "--answer_cap",
            str(args.answer_cap),
        ]
        if args.max_problems is not None:
            cmd += ["--max_problems", str(args.max_problems)]
        if args.max_rows is not None:
            cmd += ["--max_rows", str(args.max_rows)]
        if args.flush_every is not None:
            cmd += ["--flush_every", str(args.flush_every)]

        pretty = shlex.join(cmd)
        print(f"[rank {args.rank}] ({idx + 1}/{len(assigned)}) family={t.family} temp={t.temp} step={t.step}")
        print(f"[rank {args.rank}] cmd: {pretty}")
        sys.stdout.flush()

        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=project_root.as_posix(), check=True, env=os.environ.copy())

    print(f"[rank {args.rank}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
