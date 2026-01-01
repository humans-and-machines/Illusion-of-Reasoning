# Analysis package overview (`src/analysis`)

The `src/analysis` package contains all of the **post‑hoc analysis and figure
generation code** used in the paper. It sits on top of:

- `src/inference` – which produces step‑level JSONL logs, and
- `src/annotate` – which adds GPT‑based “shift in reasoning” labels.

Analysis is organized explicitly around the paper’s three research questions:

- **RQ1 – Do reasoning shifts raise accuracy?**
- **RQ2 – How do training stage and temperature affect shifts?**
- **RQ3 – How does uncertainty interact with forced reconsideration?**

This document explains the main entrypoints and how they map onto the paper’s
figures and tables.

---

## High‑level workflow

 Typical end‑to‑end pipeline (per domain / model):

1. **Run inference** (math / crossword / car‑park)  
   - e.g. via `scripts/inference/*.slurm`, which call into `src/inference`.
2. **Annotate shifts (optional but recommended)**  
   - `annotate-shifts <results_root> --split test ...`  
     or `python -m src.annotate.cli.shift_cli <results_root> --split test ...`
   - Quick audit across the standard multi-model grid (per-checkpoint % shifts):
     `python tools/master_analysis.py shift-prevalence`
   - Quick audit of accuracy (shifted vs non-shifted):
     `python tools/master_analysis.py shift-accuracy`
   - Run both audits:
     `python tools/master_analysis.py all`
   - Emit Table 2 (LaTeX) from `shift-accuracy` output:
     `python tools/master_analysis.py table2`
   - Emit pooled logistic-regression sentence (`correct ~ shift`):
     `python tools/master_analysis.py pooled-logit`
   - Emit external-model table + sentence (DeepSeek-R1 / GPT-4o):
     `python tools/master_analysis.py external-models`
   - Emit the full RQ1 section LaTeX snippet:
     `python tools/master_analysis.py rq1-section`
   - Emit the full RQ2 section LaTeX snippet:
     `python tools/master_analysis.py rq2-section`
3. **Run core RQ analyses (CSV + GLMs)**  
   - `python -m src.analysis.rq1_analysis ...`
   - `python -m src.analysis.rq2_analysis ...`
   - `python -m src.analysis.rq3_analysis ...`
4. **Generate figures / tables for the paper**  
   - `python -m src.analysis.figure_1 ...` (Aha ratios by step)  
   - `python -m src.analysis.figure_2_uncertainty ...` (uncertainty → correctness)  
   - `python -m src.analysis.final_plot_uncertainty_gate ...` (uncertainty‑gated reconsideration)  
   - `python -m src.analysis.table_1 ...` (Aha conditionals)

All analysis code expects a `results_root` containing training‑step folders
with `*.jsonl` logs (e.g. `artifacts/results/GRPO-1.5B-math-temp-0.05-3`).

---

## Core RQ entrypoints

These are the **canonical entrypoints** that orchestrate the heavier H1/H2/H3
scripts and keep RQ‑specific defaults in one place.

### RQ1 – Do shifts raise accuracy? (`rq1_analysis.py`)

- **Purpose**
  - Fit H1 Binomial GLMs and compute per‑variant accuracy deltas:
    `correct ~ C(problem) + step_std + aha`.
  - Optionally print a simple correctness×shift 2×2 table.
- **Implementation**
  - Wraps `src/analysis/h1_analysis.py` (H1 GLM + LaTeX/PDF summaries).
  - Uses `src/analysis/shift_summary.py` for the lightweight 2×2 table.
- **Key outputs (under `<results_root>/rq1/` by default)**
  - `h1_glm/h1_glm_ame_summary.csv` – per‑variant AMEs, coefficients, p‑values.
  - `h1_glm/h1_group_accuracy.csv` – accuracy for Aha=0/1 by variant.
  - `h1_glm/h1_group_accuracy_delta.csv` – accuracy deltas by variant.
  - `h1_glm/h1_group_accuracy_by_step.csv` – per‑step Aha vs accuracy.
- **Canonical command (Math example)**

```bash
python -m src.analysis.rq1_analysis \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --dataset_name MATH-500 \
  --model_name Qwen2.5-1.5B
```

### RQ2 – Training stage & temperature (`rq2_analysis.py`)

- **Purpose**
  - Training‑stage view: per‑step GLMs and uncertainty buckets (H2).
  - Temperature view: per‑temperature effects and GLMs when multiple temps are
    available.
- **Implementation**
  - Wraps `src/analysis/h2_analysis.py` for H2 GLMs and uncertainty buckets.
  - Wraps `src/analysis/temperature_effects.py` for temperature‑wise deltas.
- **Key outputs (under `<results_root>/rq2/` by default)**
  - `h2_analysis/h2_pass1_samples.csv` – per‑sample pass‑1 rows with Aha + entropy.
  - `h2_analysis/h2_step_regression.csv` – step‑wise GLM coefficients and AMEs.
  - `h2_analysis/h2_aha_vs_uncertainty_buckets.csv` – “all three Aha definitions vs uncertainty” buckets.
  - `h2_analysis/h2_all3_pass1_samples.csv` – shared sample‑level table used by uncertainty figures.
  - `temperature_effects/*` (optional) – per‑temperature summaries.
- **Canonical command**

```bash
python -m src.analysis.rq2_analysis \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --out_dir artifacts/results/GRPO-1.5B-math-temp-0.05-3/rq2
```

Optionally, to include the temperature view:

```bash
python -m src.analysis.rq2_analysis \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --temp_root artifacts/results/GRPO-1.5B-math-all-temps \
  --low_alias 0.0
```

### RQ3 – Uncertainty & intervention (`rq3_analysis.py`)

- **Purpose**
  - H3 GLMs: whether second‑pass / reconsideration helps more under high
    uncertainty (PASS‑1 entropy).
  - Optional flat CSV export with one row per `(sample, cue_variant)` for
    custom plotting in pandas/R.
- **Implementation**
  - Wraps `src/analysis/h3_analysis.py` for pooled and bucket GLMs.
  - Uses `src/analysis/export_cue_variants.py` for the cue‑variant CSV.
- **Key outputs (under `<results_root>/rq3/` by default)**
  - `h3_analysis/h3_pairs.csv` – wide PASS‑1 / PASS‑2 pairs.
  - `h3_analysis/h3_long.csv` – long `(pair, phase)` table with buckets.
  - `h3_analysis/bucket_effects.csv` – per‑bucket AMEs of phase toggle.
  - `cue_variants[_split].csv` (optional) – flat cue‑variant table.
- **Canonical command**

```bash
python -m src.analysis.rq3_analysis \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --uncertainty_field entropy_answer \
  --num_buckets 4 \
  --export_cues
```

---

## Mapping to figures & tables

This section summarizes which scripts correspond to the main paper outputs and
how they relate to the RQ pipelines above.

### Figure 1 – Aha ratios by training step (problem‑level)

- **Script**
  - `src/analysis/figure_1.py`
- **What it does**
  - Builds **problem‑level** Aha ratios over training steps, for:
    - Cue‑phrase (Words) Aha
    - GPT‑detected shifts
    - Formal shifts (δ‑based)
  - Plots per‑domain overlays (Math / Crossword) with:
    - Dots + bootstrap CIs
    - Weighted least‑squares trend lines
    - Green highlights where the average gain‑at‑shift is positive.
- **Inputs**
  - Directly reads raw `*.jsonl` inference logs under per‑domain roots.
  - Uses the same Aha definitions as H1/H2, but re‑implemented locally to
    support domain‑aware gating and δ‑sweeps.
- **Recommended sequence**

```bash
# 1) (optional) run RQ2 to materialize H2 training-stage outputs
python -m src.analysis.rq2_analysis <results_root> --split test

# 2) build the 3-up Aha ratios figure
python -m src.analysis.figure_1 \
  --root_crossword artifacts/results/.../crossword_root \
  --root_math      artifacts/results/.../math_root \
  --split test \
  --dataset_name "Crossword + MATH-500" \
  --model_name "Qwen2.5-1.5B"
```

### Figure 2 – Uncertainty → correctness (counts, densities, accuracy, regression)

- **Script**
  - `src/analysis/figure_2_uncertainty.py`
- **What it does**
  - Produces a **suite of panels** for uncertainty vs correctness:
    1. Count histograms of correct examples (All, Words, GPT, Formal).
    2. Overlaid uncertainty densities for correct examples by Aha type.
    3. 2×2 panels of densities (correct vs incorrect) for each Aha type.
    4. Per‑bin accuracy with Wilson CIs (overlay of Aha types).
    5. GLM‑based curves of accuracy vs perplexity bucket with Aha×bucket
       interaction (Words / GPT / Formal).
- **Inputs**
  - Prefers `rq2`/H2 outputs when available:
    - Reads `h2_analysis/h2_all3_pass1_samples.csv` if present (written by
      `h2_analysis.py`).
  - Falls back to parsing raw `*.jsonl` logs if H2 outputs are absent.
- **Recommended sequence**

```bash
# 1) run RQ2 / H2 for the target run
python -m src.analysis.rq2_analysis \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test

# 2) generate all uncertainty panels
python -m src.analysis.figure_2_uncertainty \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --dataset_name MATH-500 \
  --model_name "Qwen2.5-1.5B" \
  --run_rq2
```

### Uncertainty‑gated reconsideration figure (final plot)

- **Script**
  - `src/analysis/final_plot_uncertainty_gate.py`
- **What it does**
  - Builds a two‑panel “uncertainty‑gated reconsideration” figure:
    - Panel A: shift prevalence vs entropy (PASS‑1).
    - Panel B: pass‑2 correctness (triggered reconsiderations only) vs
      entropy, stratified by baseline PASS‑1 success.
- **Inputs**
  - Reads `*.jsonl` logs (potentially across domains/datasets).
  - For car‑park runs, can use the soft reward to define success.
- **Recommended sequence**

```bash
# 1) (optional) run core RQ3 H3 analysis
python -m src.analysis.rq3_analysis \
  <results_root> \
  --split test \
  --uncertainty_field entropy_answer \
  --num_buckets 4

# 2) build the uncertainty-gated reconsideration figure
python -m src.analysis.final_plot_uncertainty_gate \
  --scan_root <results_root> \
  --split test \
  --bins 0 1 2 3 4 \
  --dataset_name MATH-500 \
  --model_name "Qwen2.5-1.5B" \
  --make_plot
```

### Table 1 – Aha conditionals (domain‑level)

- **Script**
  - `src/analysis/table_1.py`
- **What it does**
  - Computes domain‑level conditionals:
    - `P(success | shift)` and `P(success | no shift)` for each domain.
    - Supports Crosswords/Math (accuracy) and Car‑park (soft reward threshold).
  - Robust to small schema variations (e.g. shift fields moved to top level).
- **Inputs**
  - Directly scans `*.jsonl` logs under per‑domain roots.
  - Uses the same shift/cue semantics as H1/H2 (Words vs GPT; optional broad
    GPT mode and gating tweaks).
- **Example command**

```bash
python -m src.analysis.table_1 \
  --root_math      artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --root_crossword artifacts/results/GRPO-1.5B-xword-temp-0.05-3 \
  --split test \
  --dataset_name "Math + Crossword" \
  --model_name "Qwen2.5-1.5B"
```

---

## Other analysis scripts

Beyond the core RQ pipelines and headline figures, `src/analysis` contains
additional scripts that build diagnostics, extended plots, and robustness
checks:

- **H‑series / uncertainty scripts**
  - `h2_temp_aha_eval.py`, `h3_uncertainty_buckets.py`, `h4_analysis.py`
  - `uncertainty_bucket_effects.py`, `entropy_bin_regression.py`
- **Graphs and heatmaps**
  - `graph_1.py`, `graph_2.py`, `graph_3.py`, `graph_3_stacked.py`, `graph_4.py`
  - `heatmap_1.py`, `temp_graph.py`, `math_plots.py`, `flips.py`
- **Shared helpers**
  - `io.py`, `labels.py`, `metrics.py`, `plotting.py`, `entropy.py`, `utils.py`
  - These are used internally by the H1–H4 scripts and figures.

Where possible, new analysis and figure code should:

- Reuse the RQ entrypoints (`rq1_analysis`, `rq2_analysis`, `rq3_analysis`)
  and their CSV outputs instead of re‑parsing JSONL; and
- Import label/entropy helpers from `labels.py` / `metrics.py` / `entropy.py`
  rather than duplicating extraction logic in standalone scripts.
