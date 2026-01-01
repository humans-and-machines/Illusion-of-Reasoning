#!/usr/bin/env python3
"""
Run unified_math on missing-question chunks for a shard of a TSV manifest.

This is intended to be launched under a single multi-GPU Slurm job via `srun`,
where each task (rank) processes a disjoint subset of manifest *groups*.

Manifest format (tab-separated, 1 chunk per line):
  family<TAB>temp<TAB>step<TAB>output_dir<TAB>dataset_start<TAB>num_examples<TAB>dataset_path

Grouping/assignment:
- All chunks that target the same output_dir+step+temp+family are run on the same rank
  to avoid races that could create duplicate rows (since the resume scan is not locked).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_SECOND_PASS_PHRASE = (
    "Hold on, this reasoning might be wrong. Let's go back and check each step carefully. "
    "Actually, this approach doesn't look correct. Let's restart and work through the solution "
    "more systematically. Wait, we need to reconsider. Let's think this through step by step."
)


@dataclass(frozen=True)
class Chunk:
    family: str
    temp: str
    step: int
    output_dir: str
    dataset_start: int
    num_examples: int
    dataset_path: str


GroupKey = Tuple[str, str, int, str]


def _parse_manifest_line(line: str) -> Optional[Chunk]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 7:
        return None
    family, temp, step_s, outdir, start_s, num_s, dataset_path = (p.strip() for p in parts)
    if not (family and temp and step_s and outdir and start_s and num_s and dataset_path):
        return None
    try:
        step = int(step_s)
        dataset_start = int(start_s)
        num_examples = int(num_s)
    except ValueError:
        return None
    return Chunk(
        family=family,
        temp=temp,
        step=step,
        output_dir=outdir,
        dataset_start=dataset_start,
        num_examples=num_examples,
        dataset_path=dataset_path,
    )


def _iter_chunks(manifest_path: Path) -> Iterable[Chunk]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            chunk = _parse_manifest_line(raw)
            if chunk is None:
                continue
            yield chunk


def _resolve_model_paths(*, project_root: Path, family: str, step: int) -> Tuple[str, str]:
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


def _group_key(chunk: Chunk) -> GroupKey:
    return (chunk.family, chunk.temp, chunk.step, chunk.output_dir)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to TSV manifest.")
    parser.add_argument("--rank", type=int, required=True, help="0-based rank (e.g., SLURM_PROCID).")
    parser.add_argument("--world_size", type=int, required=True, help="Total tasks (e.g., SLURM_NTASKS).")
    parser.add_argument("--project_root", default=".", help="Repo root (default: current working dir).")

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--entropy_mode", default="full", choices=["full", "reconsider", "none"])
    parser.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--think_cap", type=int, default=750)
    parser.add_argument("--answer_cap", type=int, default=50)
    parser.add_argument("--second_pass_phrase", default=DEFAULT_SECOND_PASS_PHRASE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def _setup_outdir_caches(env: Dict[str, str], outdir: Path) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    cacheroot = outdir / "hf_cache"
    (cacheroot / "transformers").mkdir(parents=True, exist_ok=True)
    (cacheroot / "hub").mkdir(parents=True, exist_ok=True)
    (outdir / ".triton").mkdir(parents=True, exist_ok=True)
    (outdir / ".torchinductor").mkdir(parents=True, exist_ok=True)
    (outdir / ".tmp").mkdir(parents=True, exist_ok=True)

    env = dict(env)
    env["HF_HOME"] = cacheroot.as_posix()
    env["TRANSFORMERS_CACHE"] = (cacheroot / "transformers").as_posix()
    env["HF_HUB_CACHE"] = (cacheroot / "hub").as_posix()
    env["TRITON_CACHE_DIR"] = (outdir / ".triton").as_posix()
    env["TORCHINDUCTOR_CACHE_DIR"] = (outdir / ".torchinductor").as_posix()
    env["TMPDIR"] = (outdir / ".tmp").as_posix()
    return env


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        raise SystemExit(f"Invalid rank/world_size: rank={args.rank} world_size={args.world_size}")

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    chunks = list(_iter_chunks(manifest_path))
    grouped: Dict[GroupKey, List[Chunk]] = {}
    for c in chunks:
        grouped.setdefault(_group_key(c), []).append(c)

    group_items = sorted(grouped.items(), key=lambda kv: kv[0])
    assigned_groups = [item for i, item in enumerate(group_items) if (i % args.world_size) == args.rank]

    print(
        f"[rank {args.rank}/{args.world_size}] manifest={manifest_path} "
        f"groups_total={len(group_items)} groups_assigned={len(assigned_groups)} chunks_total={len(chunks)}"
    )
    sys.stdout.flush()

    for gidx, (gkey, group_chunks) in enumerate(assigned_groups):
        family, temp, step, outdir_raw = gkey
        outdir = Path(outdir_raw)
        if not outdir.is_absolute():
            outdir = project_root / outdir

        try:
            model_name_or_path, tokenizer_path = _resolve_model_paths(project_root=project_root, family=family, step=step)
        except Exception as exc:
            print(f"[rank {args.rank}] [skip group] {gkey}: {exc}", file=sys.stderr)
            continue

        print(f"[rank {args.rank}] group {gidx + 1}/{len(assigned_groups)}: family={family} temp={temp} step={step} outdir={outdir}")
        sys.stdout.flush()

        for cidx, chunk in enumerate(sorted(group_chunks, key=lambda c: (c.dataset_start, c.num_examples))):
            dataset_path = Path(chunk.dataset_path)
            if not dataset_path.is_absolute():
                dataset_path = project_root / dataset_path
            if not dataset_path.is_file():
                print(f"[rank {args.rank}] [skip] dataset not found: {dataset_path}", file=sys.stderr)
                continue

            batch_size = chunk.num_examples
            cmd: List[str] = [
                sys.executable,
                "-u",
                (project_root / "src/inference/cli/unified_math.py").as_posix(),
                "--model_name_or_path",
                model_name_or_path,
                "--output_dir",
                outdir.as_posix(),
                "--tokenizer_path",
                tokenizer_path,
                "--attn_implementation",
                args.attn_implementation,
                "--batch_size",
                str(batch_size),
                "--entropy_mode",
                args.entropy_mode,
                "--num_examples",
                str(chunk.num_examples),
                "--dataset_start",
                str(chunk.dataset_start),
                "--num_samples",
                str(args.num_samples),
                "--temperature",
                temp,
                "--top_p",
                str(args.top_p),
                "--seed",
                str(args.seed),
                "--dtype",
                args.dtype,
                "--dataset_id",
                "MATH-500",
                "--dataset_path",
                dataset_path.as_posix(),
                "--split",
                "test",
                "--two_pass",
                "--second_pass_phrase",
                args.second_pass_phrase,
                "--think_cap",
                str(args.think_cap),
                "--answer_cap",
                str(args.answer_cap),
                "--step",
                str(step),
            ]

            print(
                f"[rank {args.rank}]  chunk {cidx + 1}/{len(group_chunks)}: "
                f"start={chunk.dataset_start} num_examples={chunk.num_examples} cmd={shlex.join(cmd)}"
            )
            sys.stdout.flush()

            if args.dry_run:
                continue

            env = _setup_outdir_caches(os.environ.copy(), outdir)
            subprocess.run(cmd, cwd=project_root.as_posix(), env=env, check=True)

    print(f"[rank {args.rank}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

