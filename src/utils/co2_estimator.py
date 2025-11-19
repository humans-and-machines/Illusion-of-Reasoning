#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
CO2e estimator (Luccioni/Lacoste-style)
--------------------------------------
Compute rough carbon emissions for ML experiments using:
  energy_kWh = (sum over runs of GPU_HOURS * PER_GPU_kW) * PUE
  CO2e_kg    = energy_kWh * GRID_INTENSITY_kg_per_kWh

This matches the methodology popularized by:
- Lacoste et al., 2019, "Quantifying the Carbon Emissions of Machine Learning"
  and subsequent Luccioni et al. emissions calculator work.

Typical usage
-------------
# Single total (GPU-hours already multiplied by #GPUs)
python co2_estimator.py \
  --gpu-hours 728 \  --per-gpu-kw 0.35 \  --pue 1.2 \  --kg-per-kwh 0.35 \  --round-to 5

# Multiple runs: pass (gpus,hours) per run; power is per-GPU
python co2_estimator.py \  --run gpus=8,hours=10 \  --run gpus=4,hours=36 \  --per-gpu-kw 0.35 \  --pue 1.2 \  --kg-per-kwh 0.35

# From a JSONL file with one object per line:
# {"gpus": 8, "hours": 10}
# {"gpus": 4, "hours": 36}
python co2_estimator.py --runs-jsonl runs.jsonl --per-gpu-kw 0.35 --pue 1.2 --kg-per-kwh 0.35

Notes
-----
- PER_GPU_kW should be an *effective* average power draw per GPU (board power under load + typical share of system/CPU/DRAM).
  If unsure, 0.35 kW per GPU (A100/H100-class with moderate utilization) is a reasonable rough value.
- PUE (Power Usage Effectiveness) defaults to 1.2 for efficient institutional DCs.
- Grid intensity varies widely by region; 0.35 kg CO2e/kWh is a conservative global-average-like figure.
- All results are rough, order-of-magnitude estimates — document your chosen parameters and assumptions.
'''
import argparse, json, sys
from typing import List, Dict

GPU_TYPE_DEFAULT_kW = {
    # Approximate under-load board powers; **override** with --per-gpu-kw for your setup.
    # Values below do NOT include system overhead; if you want "effective" per-GPU power,
    # just pass --per-gpu-kw directly (e.g., 0.35).
    "H100-80GB": 0.35,
    "A100-80GB": 0.30,
    "A100-40GB": 0.25,
    "L40S":      0.35,
    "V100-16GB": 0.25,
    "T4":        0.07,
    "RTX4090":   0.45,
    "RTX3090":   0.35,
}

def parse_run_arg(run_arg: str) -> Dict[str, float]:
    '''
    Parse --run 'gpus=8,hours=10' into {'gpus': 8.0, 'hours': 10.0}.
    '''
    parts = run_arg.split(",")
    data = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Malformed --run '{run_arg}'. Expected key=value pairs.")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            data[k] = float(v)
        except ValueError:
            raise ValueError(f"Value for {k} must be numeric, got '{v}'.")
    if "gpus" not in data or "hours" not in data:
        raise ValueError(f"--run requires keys gpus and hours, got: {data}")
    return data

def load_runs_jsonl(path: str) -> List[Dict[str, float]]:
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} must be a JSON object with 'gpus' and 'hours'.")
            if "gpus" not in obj or "hours" not in obj:
                raise ValueError(f"Line {i} missing 'gpus' and/or 'hours'. Got: {obj}")
            runs.append({"gpus": float(obj["gpus"]), "hours": float(obj["hours"])})
    return runs

def estimate_energy_kwh_from_runs(runs: List[Dict[str, float]], per_gpu_kW: float, pue: float) -> float:
    '''
    Energy (kWh) = [sum over runs of (gpus * hours * per_gpu_kW)] * PUE
    '''
    raw_kwh = sum(r["gpus"] * r["hours"] * per_gpu_kW for r in runs)
    return raw_kwh * pue

def estimate_energy_kwh_from_gpu_hours(gpu_hours: float, per_gpu_kW: float, pue: float) -> float:
    '''
    Energy (kWh) = (total GPU-hours * per_gpu_kW) * PUE
    '''
    return gpu_hours * per_gpu_kW * pue

def estimate_co2e_kg(energy_kwh: float, kg_per_kwh: float) -> float:
    return energy_kwh * kg_per_kwh

def format_kg(x: float, round_to: float = 1.0) -> str:
    if round_to and round_to > 0:
        x = round(x / round_to) * round_to
    if abs(x - int(x)) < 1e-9:
        return f"{int(x)}"
    return f"{x:.1f}"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rough CO2e estimator for ML experiments.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gpu-hours", type=float, help="Total GPU-hours across all GPUs (already multiplied by #GPUs).")
    g.add_argument("--runs-jsonl", type=str, help="Path to JSONL with lines like: {\"gpus\":8, \"hours\":10}")
    ap.add_argument("--run", action="append", default=[],
                    help="Add a run as 'gpus=8,hours=10'. Can be used multiple times. "
                         "Ignored if --gpu-hours is provided unless --combine is set to 'sum'.")
    ap.add_argument("--combine", choices=["sum","prefer_gpu_hours"], default="prefer_gpu_hours",
                    help="If both --gpu-hours and --run/--runs-jsonl are provided, choose how to combine. "
                         "'prefer_gpu_hours' ignores runs; 'sum' adds them together.")
    ap.add_argument("--per-gpu-kw", type=float, default=None,
                    help="Effective average kW draw per GPU (includes typical system share). "
                         "If not set, inferred from --gpu-type, else defaults to 0.35.")
    ap.add_argument("--gpu-type", type=str, choices=sorted(GPU_TYPE_DEFAULT_kW.keys()),
                    help="Lookup a typical per-GPU kW (board power under load). Consider using --per-gpu-kw to include overhead.")
    ap.add_argument("--pue", type=float, default=1.2, help="Power Usage Effectiveness (default 1.2)")
    ap.add_argument("--kg-per-kwh", type=float, default=0.35, help="Grid intensity in kg CO2e/kWh (default 0.35)")
    ap.add_argument("--round-to", type=float, default=5.0, help="Round final kg CO2e to nearest N (default 5). Use 0 to disable.")
    ap.add_argument("--show-range", action="store_true",
                    help="Also compute a sensitivity range using --pue-min/--pue-max and --kg-per-kwh-min/--kg-per-kwh-max.")
    ap.add_argument("--pue-min", type=float, default=1.1)
    ap.add_argument("--pue-max", type=float, default=1.4)
    ap.add_argument("--kg-per-kwh-min", type=float, default=0.2)
    ap.add_argument("--kg-per-kwh-max", type=float, default=0.6)
    args = ap.parse_args(argv)

    # Determine per-GPU kW
    if args.per_gpu_kw is not None:
        per_gpu_kw = args.per_gpu_kw
    elif args.gpu_type:
        per_gpu_kw = GPU_TYPE_DEFAULT_kW[args.gpu_type]
    else:
        per_gpu_kw = 0.35  # conservative effective default

    # Gather runs
    runs: List[Dict[str,float]] = []
    if args.runs_jsonl:
        runs.extend(load_runs_jsonl(args.runs_jsonl))
    for ra in args.run:
        runs.append(parse_run_arg(ra))

    # Compute energy
    energy_kwh = 0.0
    if args.gpu_hours is not None and args.combine == "prefer_gpu_hours":
        energy_kwh = estimate_energy_kwh_from_gpu_hours(args.gpu_hours, per_gpu_kw, args.pue)
    else:
        total_gpu_hours = 0.0
        if args.gpu_hours is not None:
            total_gpu_hours += args.gpu_hours
        if runs:
            total_gpu_hours += sum(r["gpus"] * r["hours"] for r in runs)
        if total_gpu_hours <= 0:
            print("No GPU-hours found. Provide --gpu-hours or at least one --run/--runs-jsonl.", file=sys.stderr)
            return 2
        energy_kwh = estimate_energy_kwh_from_gpu_hours(total_gpu_hours, per_gpu_kw, args.pue)

    co2e_kg = estimate_co2e_kg(energy_kwh, args.kg_per_kwh)

    rounded_str = format_kg(co2e_kg, round_to=args.round_to)
    print("=== CO2e Estimate (Luccioni/Lacoste-style) ===")
    print(f"per_gpu_kw (effective): {per_gpu_kw:.3f} kW")
    print(f"PUE:                    {args.pue:.2f}")
    print(f"Grid intensity:         {args.kg_per_kwh:.3f} kg/kWh")
    print(f"Energy:                 {energy_kwh:.2f} kWh")
    print(f"CO2e:                   {co2e_kg:.2f} kg  (rounded ≈ {rounded_str} kg)")

    if args.show_range:
        base_gpu_hours = args.gpu_hours if args.gpu_hours is not None else sum(r['gpus']*r['hours'] for r in runs)
        e_lo = estimate_energy_kwh_from_gpu_hours(base_gpu_hours, per_gpu_kw, args.pue_min)
        e_hi = estimate_energy_kwh_from_gpu_hours(base_gpu_hours, per_gpu_kw, args.pue_max)
        lo = estimate_co2e_kg(e_lo, args.kg_per_kwh_min)
        hi = estimate_co2e_kg(e_hi, args.kg_per_kwh_max)
        print(f"Range (PUE {args.pue_min}-{args.pue_max}, kg/kWh {args.kg_per_kwh_min}-{args.kg_per_kwh_max}): "
              f"{lo:.0f}–{hi:.0f} kg")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
