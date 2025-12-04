#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2e estimator (Luccioni/Lacoste-style)
---------------------------------------
Compute rough carbon emissions for ML experiments using:
  energy_kWh = (sum over runs of GPU_HOURS * PER_GPU_kW) * PUE
  CO2e_kg    = energy_kWh * GRID_INTENSITY_kg_per_kWh

This matches the methodology popularized by:
- Lacoste et al., 2019, "Quantifying the Carbon Emissions of Machine Learning"
  and subsequent Luccioni et al. emissions calculator work.

Typical usage
-------------
# Single total (GPU-hours already multiplied by #GPUs)
python co2_estimator.py \\
  --gpu-hours 728 \\
  --per-gpu-kw 0.35 \\
  --pue 1.2 \\
  --kg-per-kwh 0.35 \\
  --round-to 5

# Multiple runs: pass (gpus,hours) per run; power is per-GPU
python co2_estimator.py \\
  --run gpus=8,hours=10 \\
  --run gpus=4,hours=36 \\
  --per-gpu-kw 0.35 \\
  --pue 1.2 \\
  --kg-per-kwh 0.35

# From a JSONL file with one object per line:
# {"gpus": 8, "hours": 10}
# {"gpus": 4, "hours": 36}
python co2_estimator.py \\
  --runs-jsonl runs.jsonl \\
  --per-gpu-kw 0.35 \\
  --pue 1.2 \\
  --kg-per-kwh 0.35

Notes
-----
- PER_GPU_kW should be an *effective* average power draw per GPU
  (board power under load + typical share of system/CPU/DRAM).
- If unsure, 0.35 kW per GPU (A100/H100-class with moderate utilization)
  is a reasonable rough value.
- PUE (Power Usage Effectiveness) defaults to 1.2 for efficient institutional DCs.
- Grid intensity varies widely by region; 0.35 kg CO2e/kWh is a conservative
  global-average-like figure.
- All results are rough, order-of-magnitude estimates — document your chosen
  parameters and assumptions.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional


GPU_TYPE_DEFAULT_kW = {
    # Approximate under-load board powers; **override** with --per-gpu-kw for your setup.
    # Values below do NOT include system overhead; if you want "effective" per-GPU power,
    # just pass --per-gpu-kw directly (e.g., 0.35).
    "H100-80GB": 0.35,
    "A100-80GB": 0.30,
    "A100-40GB": 0.25,
    "L40S": 0.35,
    "V100-16GB": 0.25,
    "T4": 0.07,
    "RTX4090": 0.45,
    "RTX3090": 0.35,
}


def parse_run_arg(run_arg: str) -> Dict[str, float]:
    """
    Parse --run 'gpus=8,hours=10' into {'gpus': 8.0, 'hours': 10.0}.
    """
    parts = run_arg.split(",")
    data = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Malformed --run '{run_arg}'. Expected key=value pairs.")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            data[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Value for {key} must be numeric, got '{value}'.") from exc
    if "gpus" not in data or "hours" not in data:
        raise ValueError(f"--run requires keys gpus and hours, got: {data}")
    return data


def load_runs_jsonl(path: str) -> List[Dict[str, float]]:
    """
    Load run specifications from a JSONL file into a list of {gpus, hours} dicts.
    """
    runs = []
    with open(path, "r", encoding="utf-8") as file_handle:
        for line_index, line in enumerate(file_handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_index} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_index} must be a JSON object with 'gpus' and 'hours'.")
            if "gpus" not in obj or "hours" not in obj:
                raise ValueError(f"Line {line_index} missing 'gpus' and/or 'hours'. Got: {obj}")
            runs.append({"gpus": float(obj["gpus"]), "hours": float(obj["hours"])})
    return runs


def estimate_energy_kwh_from_runs(runs: List[Dict[str, float]], per_gpu_kw: float, pue: float) -> float:
    """
    Energy (kWh) = [sum over runs of (gpus * hours * per_gpu_kw)] * PUE.
    """
    raw_kwh = sum(r["gpus"] * r["hours"] * per_gpu_kw for r in runs)
    return raw_kwh * pue


def estimate_energy_kwh_from_gpu_hours(gpu_hours: float, per_gpu_kw: float, pue: float) -> float:
    """
    Energy (kWh) = (total GPU-hours * per_gpu_kw) * PUE
    """
    return gpu_hours * per_gpu_kw * pue


def estimate_co2e_kg(energy_kwh: float, kg_per_kwh: float) -> float:
    """
    Compute CO2e in kilograms from energy and grid intensity.
    """
    return energy_kwh * kg_per_kwh


def format_kg(kilograms: float, rounding_increment: float = 1.0) -> str:
    """
    Format a kilogram value with optional rounding to a given increment.
    """
    if rounding_increment and rounding_increment > 0:
        kilograms = round(kilograms / rounding_increment) * rounding_increment
    if abs(kilograms - int(kilograms)) < 1e-9:
        return f"{int(kilograms)}"
    return f"{kilograms:.1f}"


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser for the CO2e estimator.
    """
    parser = argparse.ArgumentParser(description="Rough CO2e estimator for ML experiments.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--gpu-hours",
        type=float,
        help="Total GPU-hours across all GPUs (already multiplied by #GPUs).",
    )
    mode_group.add_argument(
        "--runs-jsonl",
        type=str,
        help='Path to JSONL with lines like: {"gpus":8, "hours":10}',
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Add a run as 'gpus=8,hours=10'. Can be used multiple times. "
        "Ignored if --gpu-hours is provided unless --combine is set to 'sum'.",
    )
    parser.add_argument(
        "--combine",
        choices=["sum", "prefer_gpu_hours"],
        default="prefer_gpu_hours",
        help=(
            "If both --gpu-hours and --run/--runs-jsonl are provided, choose how to combine. "
            "'prefer_gpu_hours' ignores runs; 'sum' adds them together."
        ),
    )
    parser.add_argument(
        "--per-gpu-kw",
        type=float,
        default=None,
        help=(
            "Effective average kW draw per GPU (includes typical system share). "
            "If not set, inferred from --gpu-type, else defaults to 0.35."
        ),
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=sorted(GPU_TYPE_DEFAULT_kW.keys()),
        help=(
            "Lookup a typical per-GPU kW (board power under load). Consider using --per-gpu-kw to include overhead."
        ),
    )
    parser.add_argument(
        "--pue",
        type=float,
        default=1.2,
        help="Power Usage Effectiveness (default 1.2)",
    )
    parser.add_argument(
        "--kg-per-kwh",
        type=float,
        default=0.35,
        help="Grid intensity in kg CO2e/kWh (default 0.35)",
    )
    parser.add_argument(
        "--round-to",
        type=float,
        default=5.0,
        help="Round final kg CO2e to nearest N (default 5). Use 0 to disable.",
    )
    parser.add_argument(
        "--show-range",
        action="store_true",
        help=("Also compute a sensitivity range using --pue-min/--pue-max and --kg-per-kwh-min/--kg-per-kwh-max."),
    )
    parser.add_argument("--pue-min", type=float, default=1.1)
    parser.add_argument("--pue-max", type=float, default=1.4)
    parser.add_argument("--kg-per-kwh-min", type=float, default=0.2)
    parser.add_argument("--kg-per-kwh-max", type=float, default=0.6)
    return parser


def resolve_per_gpu_kw(args) -> float:
    """
    Determine the effective per-GPU kW value from CLI arguments.
    """
    if args.per_gpu_kw is not None:
        return args.per_gpu_kw
    if args.gpu_type:
        return GPU_TYPE_DEFAULT_kW[args.gpu_type]
    return 0.35  # conservative effective default


def collect_runs(args) -> List[Dict[str, float]]:
    """
    Collect all run specifications from JSONL files and --run arguments.
    """
    runs: List[Dict[str, float]] = []
    if args.runs_jsonl:
        runs.extend(load_runs_jsonl(args.runs_jsonl))
    for run_spec in args.run:
        runs.append(parse_run_arg(run_spec))
    return runs


def compute_energy_kwh(args, per_gpu_kw: float, runs: List[Dict[str, float]]) -> Optional[float]:
    """
    Compute total energy usage in kWh from CLI arguments and resolved runs.
    Returns None when no GPU-hours information is available.
    """
    if args.gpu_hours is not None and args.combine == "prefer_gpu_hours":
        return estimate_energy_kwh_from_gpu_hours(args.gpu_hours, per_gpu_kw, args.pue)

    total_gpu_hours = 0.0
    if args.gpu_hours is not None:
        total_gpu_hours += args.gpu_hours
    if runs:
        total_gpu_hours += sum(run["gpus"] * run["hours"] for run in runs)
    if total_gpu_hours <= 0:
        return None
    return estimate_energy_kwh_from_gpu_hours(total_gpu_hours, per_gpu_kw, args.pue)


def print_summary(args, per_gpu_kw: float, energy_kwh: float, co2e_kg: float, rounded_kg: str) -> None:
    """
    Print a human-readable summary of the CO2e estimation.
    """
    print("=== CO2e Estimate (Luccioni/Lacoste-style) ===")
    print(f"per_gpu_kw (effective): {per_gpu_kw:.3f} kW")
    print(f"PUE:                    {args.pue:.2f}")
    print(f"Grid intensity:         {args.kg_per_kwh:.3f} kg/kWh")
    print(f"Energy:                 {energy_kwh:.2f} kWh")
    print(f"CO2e:                   {co2e_kg:.2f} kg  (rounded ≈ {rounded_kg} kg)")


def print_sensitivity_range(args, per_gpu_kw: float, runs: List[Dict[str, float]]) -> None:
    """
    Print a simple sensitivity analysis range over PUE and grid intensity.
    """
    if args.gpu_hours is not None:
        base_gpu_hours = args.gpu_hours
    else:
        base_gpu_hours = sum(run["gpus"] * run["hours"] for run in runs)

    energy_low = estimate_energy_kwh_from_gpu_hours(base_gpu_hours, per_gpu_kw, args.pue_min)
    energy_high = estimate_energy_kwh_from_gpu_hours(base_gpu_hours, per_gpu_kw, args.pue_max)
    low_kg = estimate_co2e_kg(energy_low, args.kg_per_kwh_min)
    high_kg = estimate_co2e_kg(energy_high, args.kg_per_kwh_max)
    print(
        f"Range (PUE {args.pue_min}-{args.pue_max}, "
        f"kg/kWh {args.kg_per_kwh_min}-{args.kg_per_kwh_max}): "
        f"{low_kg:.0f}–{high_kg:.0f} kg"
    )


def main(argv=None) -> int:
    """
    Entry point for the CO2e estimator CLI.

    Parses command-line arguments, computes total energy usage and emissions,
    and prints a human-readable summary.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    per_gpu_kw = resolve_per_gpu_kw(args)
    runs = collect_runs(args)
    energy_kwh = compute_energy_kwh(args, per_gpu_kw, runs)
    if energy_kwh is None:
        print(
            "No GPU-hours found. Provide --gpu-hours or at least one --run/--runs-jsonl.",
            file=sys.stderr,
        )
        return 2

    co2e_kg = estimate_co2e_kg(energy_kwh, args.kg_per_kwh)
    rounded_str = format_kg(co2e_kg, rounding_increment=args.round_to)
    print_summary(args, per_gpu_kw, energy_kwh, co2e_kg, rounded_str)

    if args.show_range:
        print_sensitivity_range(args, per_gpu_kw, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
