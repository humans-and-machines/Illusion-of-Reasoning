#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines

"""
Uncertainty buckets: Aha vs No-Aha (accuracy, odds, effects)
Qwen-7B & Llama-8B only; all temperatures & steps

Adds (kept from 1.5B script):
- Stratified CMH test (domain×entropy-bin strata) on P(correct | shift)
- GLM LRT "ANOVA" with stratum fixed effects
- Domain-wise two-proportion tests and pooled test
- Tidy CSV + concise console summary

NEW for 7B/8B:
- Flexible discovery: finds directories whose names include
  (qwen & 7b) or (llama & 8b), plus domain (math/xword/carpark) and temp tokens.
- Per-model global regressions:
    (a) Training stage:   correct ~ C(problem_id) + step_std + shift
    (b) Temperature:      correct ~ C(problem_id) + C(temp) + shift
  -> Produces N, shift share, p(correct|S=1), Δpp, AME, p-value.

Multi-metric runner:
- Single metric: use --metric entropy_answer|entropy|entropy_think
- Multiple: use --metrics entropy_answer entropy entropy_think answer_plus_think
- Default (no --metric/--metrics): runs all four.

Binning:
- fixed edges via --fixed_bins "0,0.5,1,2,inf"
- or equal-count bins via --equal_n_bins --bins K (global/domain scope)

Only steps <= 1000 are loaded by default (hard cap).
"""

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tools.sm_exceptions import PerfectSeparationError

plt.switch_backend("Agg")

try:
    # Preferred when analysis is installed as a package
    from src.analysis.common.model_utils import (
        MODEL_LABELS,
        detect_domain,
        detect_model_key,
        detect_temperature,
    )
    from src.analysis.labels import AHA_KEYS_CANONICAL, AHA_KEYS_BROAD, aha_gpt_for_rec
    from src.analysis.io import iter_records_from_file
    from src.analysis.metrics import wilson_ci
    from src.analysis.utils import coerce_bool, coerce_float, get_problem_id, step_within_bounds
except ImportError:  # pragma: no cover - script fallback
    # Fallback for running this file directly: add project src root and import
    import sys as _sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.append(_ROOT)
    from analysis.common.model_utils import (  # type: ignore
        MODEL_LABELS,
        detect_domain,
        detect_model_key,
        detect_temperature,
    )
    from analysis.labels import AHA_KEYS_CANONICAL, AHA_KEYS_BROAD, aha_gpt_for_rec  # type: ignore
    from analysis.io import iter_records_from_file  # type: ignore
    from analysis.utils import (  # type: ignore
        coerce_bool,
        coerce_float,
        get_problem_id,
        step_within_bounds,
    )
    from analysis.metrics import wilson_ci  # type: ignore

@dataclass
class RootDiscoveryConfig:
    """Configuration for discovering model/domain roots for 7B/8B models."""

    temps: List[float]
    low_alias: float
    want_models: List[str]
    want_domains: List[str]
    verbose: bool


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap settings for AME estimation."""

    draws: int
    seed: int


@dataclass(frozen=True)
class ShiftBucketCounts:
    """Summary counts for shift vs. no-shift outcomes."""

    total: int
    shift: int
    no_shift: int
    correct_shift: int
    correct_no_shift: int


def _print_discovered_roots(mapping: Dict[str, Dict[float, Dict[str, str]]]) -> None:
    """Pretty-print discovered roots grouped by model and temperature."""
    print("[info] discovered roots (7B/8B):")
    for model_key, temps_by_domain in mapping.items():
        label = MODEL_LABELS.get(model_key, model_key)
        print(f"  [{label}]")
        for temp_value, domains_for_temp in sorted(temps_by_domain.items()):
            print(f"    T={temp_value}:")
            for domain_name, root_path in domains_for_temp.items():
                print(f"      {domain_name:9s} -> {root_path}")


def discover_roots_7b8b(
    scan_root: str,
    config: RootDiscoveryConfig,
) -> Dict[str, Dict[float, Dict[str, str]]]:
    """
    Returns: mapping_by_model[model_key][T][domain] = path
    Prefers the longest path string if duplicates exist.
    """
    temps_set = {float(value) for value in config.temps}
    want_models = [model_name.lower() for model_name in config.want_models]
    want_domains = set(config.want_domains)

    mapping: Dict[str, Dict[float, Dict[str, str]]] = {}

    for entry in os.listdir(scan_root):
        full = os.path.join(scan_root, entry)
        if not os.path.isdir(full):
            continue
        low = entry.lower()

        model_key = detect_model_key(low)
        if model_key is None or model_key not in want_models:
            continue

        dom = detect_domain(low)
        if dom is None or dom not in want_domains:
            continue

        temp_value = detect_temperature(low, config.low_alias)
        if temp_value is None:
            continue
        target_temp = float(temp_value)
        if target_temp not in temps_set:
            continue

        mapping.setdefault(model_key, {})
        mapping[model_key].setdefault(target_temp, {})
        prev = mapping[model_key][target_temp].get(dom)
        if prev is None or len(full) > len(prev):
            mapping[model_key][target_temp][dom] = full

    if config.verbose:
        _print_discovered_roots(mapping)
    return mapping

def get_pid(rec: Dict[str, Any]) -> Optional[str]:
    """
    Domain-agnostic problem identifier, wrapper around shared get_problem_id
    to preserve the previous behavior (namespacing by domain happens later).
    """
    return get_problem_id(rec)

# ==========================
# Load pass-1 rows (all T)
# ==========================

class RowLoadConfig:
    """
    Configuration controlling how pass-1 rows are loaded.

    To keep the number of instance attributes small for style checks, most
    optional settings are stored in a private options mapping and exposed
    via properties.
    """

    def __init__(
        self,
        *,
        entropy_source: str,
        allow_fallback: bool,
        **options: Any,
    ) -> None:
        self.entropy_source = entropy_source
        self.allow_fallback = allow_fallback
        self.min_step = options.get("min_step")
        self.max_step = options.get("max_step")
        self._options: Dict[str, Any] = {
            "carpark_op": options.get("carpark_op", "ge"),
            "carpark_threshold": options.get("carpark_threshold", 0.1),
            "gpt_mode": options.get("gpt_mode", "canonical"),
            "verbose": options.get("verbose", False),
            "combined_mode": options.get("combined_mode", "sum"),
            "temp": options.get("temp"),
            "model_label": options.get("model_label"),
        }

    @property
    def carpark_op(self) -> str:
        """Return the comparison operator used for Carpark scoring."""
        return str(self._options["carpark_op"])

    @property
    def carpark_threshold(self) -> float:
        """Return the numeric Carpark correctness threshold."""
        return float(self._options["carpark_threshold"])

    @property
    def gpt_mode(self) -> str:
        """Return which GPT judge subset to use when reading rows."""
        return str(self._options["gpt_mode"])

    @property
    def verbose(self) -> bool:
        """Return whether verbose logging is enabled."""
        return bool(self._options["verbose"])

    @property
    def combined_mode(self) -> str:
        """Return the mode for combining entropy signals."""
        return str(self._options["combined_mode"])

    @property
    def temp(self) -> Optional[float]:
        """Return the sampling temperature override, if any."""
        raw = self._options["temp"]
        return None if raw is None else float(raw)

    @property
    def model_label(self) -> Optional[str]:
        """Return the human-readable model label, if provided."""
        raw = self._options["model_label"]
        return None if raw is None else str(raw)


def _extract_step_from_path(file_path: str) -> Optional[int]:
    """
    Infer a training step integer from a path containing tokens like ``step250``.
    """
    step_value: Optional[int] = None
    for token in os.path.normpath(file_path).split(os.sep):
        if token.lower().startswith("step"):
            digits = re.sub("[^0-9]", "", token)
            if digits:
                try:
                    step_value = int(digits)
                except (TypeError, ValueError):
                    step_value = None
            break
    return step_value


def _compute_correct_flag_for_carpark(
    pass1_data: Dict[str, Any],
    cfg: RowLoadConfig,
) -> Optional[int]:
    """Compute correctness for Carpark-style soft_reward records."""
    soft_reward = coerce_float(pass1_data.get("soft_reward"))
    if soft_reward is None:
        return None
    if cfg.carpark_op == "gt":
        return int(soft_reward > cfg.carpark_threshold)
    if cfg.carpark_op == "ge":
        return int(soft_reward >= cfg.carpark_threshold)
    if cfg.carpark_op == "eq":
        return int(soft_reward == cfg.carpark_threshold)
    return int(soft_reward >= cfg.carpark_threshold)


def _compute_correct_flag_generic(pass1_data: Dict[str, Any]) -> Optional[int]:
    """Compute correctness for non-Carpark domains."""
    correct_bool = coerce_bool(pass1_data.get("is_correct_pred"))
    if correct_bool is None:
        return None
    return int(correct_bool)


def _compute_correct_flag_for_domain(
    domain: str,
    pass1_data: Dict[str, Any],
    cfg: RowLoadConfig,
) -> Optional[int]:
    """Dispatch correctness computation based on domain."""
    if domain == "Carpark":
        return _compute_correct_flag_for_carpark(pass1_data, cfg)
    return _compute_correct_flag_generic(pass1_data)


def _compute_shift_flag_for_domain(
    pass1_data: Dict[str, Any],
    record: Dict[str, Any],
    cfg: RowLoadConfig,
    domain: str,
) -> int:
    """Compute the GPT-based shift flag (domain-aware)."""
    gpt_keys = (
        AHA_KEYS_CANONICAL
        if cfg.gpt_mode == "canonical"
        else AHA_KEYS_BROAD
    )
    return aha_gpt_for_rec(
        pass1_data,
        record,
        gpt_subset_native=True,
        gpt_keys=gpt_keys,
        domain=domain,
    )


_SIMPLE_ENTROPY_KEYS = {"entropy_answer", "entropy", "entropy_think"}


def _entropy_from_answer_plus_think(
    pass1_data: Dict[str, Any],
    combined_mode: str,
    allow_fallback: bool,
) -> Optional[float]:
    """Combine answer/think entropies, with optional fallback to ``entropy``."""
    entropy_answer = coerce_float(pass1_data.get("entropy_answer"))
    entropy_think = coerce_float(pass1_data.get("entropy_think"))
    if entropy_answer is not None and entropy_think is not None:
        if combined_mode == "sum":
            return entropy_answer + entropy_think
        return (entropy_answer + entropy_think) / 2.0
    if allow_fallback:
        return coerce_float(pass1_data.get("entropy"))
    return None


def _fallback_entropy_from_simple_sources(
    pass1_data: Dict[str, Any],
) -> Optional[float]:
    """Fallback over simple entropy sources in a fixed order."""
    for entropy_key in ("entropy_answer", "entropy", "entropy_think"):
        fallback_value = coerce_float(pass1_data.get(entropy_key))
        if fallback_value is not None:
            return fallback_value
    return None


def _entropy_from_pass1(
    pass1_data: Dict[str, Any],
    *,
    entropy_source: str,
    allow_fallback: bool,
    combined_mode: str,
) -> Optional[float]:
    """
    Extract the requested entropy metric from a pass-1 mapping, with fallbacks.
    """
    if entropy_source == "answer_plus_think":
        entropy_value = _entropy_from_answer_plus_think(
            pass1_data,
            combined_mode=combined_mode,
            allow_fallback=allow_fallback,
        )
    elif entropy_source in _SIMPLE_ENTROPY_KEYS:
        entropy_value = coerce_float(pass1_data.get(entropy_source))
    else:
        entropy_value = None

    if entropy_value is None and allow_fallback:
        entropy_value = _fallback_entropy_from_simple_sources(pass1_data)
    return entropy_value


def _rows_for_file(
    file_path: str,
    domain: str,
    cfg: RowLoadConfig,
) -> List[Dict[str, Any]]:
    """Build analysis rows for a single JSONL file, or return [] if filtered."""
    step_value = _extract_step_from_path(file_path)
    if step_value is None:
        return []
    if not step_within_bounds(step_value, cfg.min_step, cfg.max_step):
        return []

    rows_for_file: List[Dict[str, Any]] = []
    for record in iter_records_from_file(file_path):
        pass1_data = record.get("pass1") or {}
        if not isinstance(pass1_data, dict):
            pass1_data = {}

        problem_id_raw = get_pid(record)
        if problem_id_raw is None:
            continue
        problem_id = f"{domain}::{problem_id_raw}"

        entropy_value = _entropy_from_pass1(
            pass1_data,
            entropy_source=cfg.entropy_source,
            allow_fallback=cfg.allow_fallback,
            combined_mode=cfg.combined_mode,
        )
        if entropy_value is None:
            continue

        correct_flag = _compute_correct_flag_for_domain(
            domain,
            pass1_data,
            cfg,
        )
        if correct_flag is None:
            continue

        shift_flag = _compute_shift_flag_for_domain(
            pass1_data,
            record,
            cfg,
            domain,
        )

        rows_for_file.append(
            {
                "model": cfg.model_label or "UNKNOWN",
                "domain": domain,
                "temp": cfg.temp,
                "step": int(step_value),
                "problem_id": problem_id,
                "ent": entropy_value,
                "correct": correct_flag,
                "shift": shift_flag,
            },
        )
    return rows_for_file


def load_rows(
    dir_path: str,
    split: str,
    domain: str,
    cfg: RowLoadConfig,
) -> pd.DataFrame:
    """
    Load per-problem rows from JSONL result files for a given domain and split.

    The loader applies step filters, computes entropy from the requested source,
    derives correctness (with a domain-specific Carpark rule), and attaches
    GPT-based Aha shift indicators and metadata for downstream analyses.
    """
    files = scan_step_jsonls(dir_path, split, cfg.verbose)
    rows: List[Dict[str, Any]] = []

    for file_path in files:
        rows.extend(_rows_for_file(file_path, domain, cfg))

    dataframe = pd.DataFrame(rows)
    if cfg.verbose:
        print(
            f"[info] loaded rows for {cfg.model_label}/{domain} (steps",
            end="",
        )
        if cfg.max_step is not None:
            print(f" ≤ {cfg.max_step}", end="")
        print(f"): {len(dataframe)}")
    return dataframe

def scan_step_jsonls(dir_path: str, split: str, verbose: bool) -> List[str]:
    """
    Recursively find step*/.../*.jsonl files under ``dir_path``, filtered by ``split``.
    """
    files: List[str] = []
    all_json_paths: List[str] = []
    for dirpath, _, filenames in os.walk(dir_path):
        if not os.path.basename(dirpath).lower().startswith("step"):
            continue
        for filename in filenames:
            if not filename.lower().endswith(".jsonl"):
                continue
            full = os.path.join(dirpath, filename)
            all_json_paths.append(full)
            if split:
                if split.lower() in filename.lower():
                    files.append(full)
            else:
                files.append(full)
    if split and not files and verbose:
        print(
            f"[warn] no files matched split='{split}' in {dir_path}; "
            f"using ALL ({len(all_json_paths)})",
        )
        files = all_json_paths
    files.sort()
    if verbose:
        print(f"[info] {dir_path}: {len(files)} JSONLs")
    return files

# =================
# Binning helpers
# =================

def _parse_fixed_edges(edges_spec: str) -> np.ndarray:
    """
    Parse a comma-separated string of bin edges into a numeric array.
    """
    tokens = [token.strip() for token in edges_spec.split(",") if token.strip()]
    values: List[float] = []
    for token in tokens:
        if token.lower() in {"inf", "+inf", "infinity"}:
            values.append(np.inf)
        else:
            values.append(float(token))
    edges_array = np.asarray(values, dtype=float)
    if edges_array.size < 2 or np.any(np.diff(edges_array) <= 0):
        raise ValueError(
            "fixed_bins must be strictly increasing with ≥2 edges, "
            f"got: {edges_array}",
        )
    return edges_array

def build_edges(
    data_frame: pd.DataFrame,
    bins: int,
    method: str,
    scope: str,
) -> Dict[str, np.ndarray]:
    """
    Build per-domain entropy bin edges given a method and scope.

    ``scope="global"`` reuses one set of edges for every domain; otherwise,
    the edges are computed per domain using either uniform spacing or quantiles.
    """
    edges_by_dom: Dict[str, np.ndarray] = {}
    if scope == "global":
        ent_values = data_frame["ent"].to_numpy()
        if method == "uniform":
            edges = np.linspace(
                np.nanmin(ent_values),
                np.nanmax(ent_values),
                bins + 1,
            )
        else:
            edges = np.quantile(
                ent_values,
                np.linspace(0, 1, bins + 1),
            )
        for domain_name in data_frame["domain"].unique():
            edges_by_dom[domain_name] = edges
    else:
        for domain_name, sub in data_frame.groupby("domain", sort=False):
            ent_values = sub["ent"].to_numpy()
            if method == "uniform":
                edges = np.linspace(
                    np.nanmin(ent_values),
                    np.nanmax(ent_values),
                    bins + 1,
                )
            else:
                edges = np.quantile(
                    ent_values,
                    np.linspace(0, 1, bins + 1),
                )
            edges_by_dom[domain_name] = edges
    return edges_by_dom


def assign_bins(
    data_frame: pd.DataFrame,
    edges_by_dom: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Assign entropy values to integer bins given per-domain bin edges.
    """
    data_frame = data_frame.copy()
    data_frame["bin"] = -1
    centers_by_dom: Dict[str, np.ndarray] = {}
    for domain_name, edges in edges_by_dom.items():
        centers_by_dom[domain_name] = 0.5 * (edges[:-1] + edges[1:])
        mask = data_frame["domain"] == domain_name
        data_frame.loc[mask, "bin"] = np.digitize(
            data_frame.loc[mask, "ent"],
            edges,
            right=True,
        ) - 1
        data_frame.loc[mask, "bin"] = data_frame.loc[mask, "bin"].clip(
            lower=0,
            upper=len(edges) - 2,
        )
    return data_frame, centers_by_dom


def assign_bins_fixed(
    data_frame: pd.DataFrame,
    edges_by_dom: Dict[str, np.ndarray],
    last_center_offset: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Assign bins for fixed, explicitly provided edges (per domain).

    The final open-ended bin is centered using ``last_center_offset`` when the
    top edge is infinite.
    """
    data_frame = data_frame.copy()
    data_frame["bin"] = -1
    centers_by_dom: Dict[str, np.ndarray] = {}
    for domain_name, edges in edges_by_dom.items():
        mask = data_frame["domain"] == domain_name
        ent_values = data_frame.loc[mask, "ent"].to_numpy()
        idx = np.digitize(ent_values, edges, right=False) - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        data_frame.loc[mask, "bin"] = idx

        centers = 0.5 * (edges[:-1] + edges[1:])
        if np.isinf(edges[-1]):
            centers[-1] = edges[-2] + float(last_center_offset)
        centers_by_dom[domain_name] = centers
    return data_frame, centers_by_dom

def _assign_equal_bins_1d(num_rows: int, num_bins: int) -> np.ndarray:
    """Split ``num_rows`` items into ``num_bins`` nearly equal contiguous groups."""
    idx = np.arange(num_rows)
    parts = np.array_split(idx, num_bins)
    out = np.empty(num_rows, dtype=int)
    for bin_index, part in enumerate(parts):
        out[part] = bin_index
    return out

def assign_bins_equal_count(
    data_frame: pd.DataFrame,
    bins: int,
    scope: str,
    tie_break: str = "stable",
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Assign equal-count entropy bins, either globally or per-domain.
    """
    rng = np.random.default_rng(seed)
    data_frame = data_frame.copy()
    data_frame["bin"] = -1

    def _sort_keys(frame: pd.DataFrame) -> pd.DataFrame:
        if tie_break == "random":
            jitter = rng.uniform(-1e-12, 1e-12, size=len(frame))
            return (
                frame.assign(_j=jitter)
                .sort_values(
                    ["ent", "_j", "domain", "problem_id", "step"],
                    kind="mergesort",
                )
                .drop(columns="_j")
            )
        return frame.sort_values(
            ["ent", "domain", "problem_id", "step"],
            kind="mergesort",
        )

    centers_by_dom: Dict[str, np.ndarray] = {}

    if scope == "global":
        sorted_frame = _sort_keys(data_frame)
        bin_vector = _assign_equal_bins_1d(len(sorted_frame), bins)
        data_frame.loc[sorted_frame.index, "bin"] = bin_vector
        for domain_name, sub_frame in data_frame.groupby("domain", sort=False):
            med = sub_frame.groupby("bin", sort=False)["ent"].median()
            centers = med.reindex(range(bins)).to_numpy()
            if np.any(pd.isna(centers)):
                centers = np.where(
                    pd.isna(centers),
                    np.nanmedian(data_frame["ent"].to_numpy()),
                    centers,
                )
            centers_by_dom[domain_name] = centers
    else:
        for domain_name, sub_frame in data_frame.groupby("domain", sort=False):
            sorted_frame = _sort_keys(sub_frame)
            bin_vector = _assign_equal_bins_1d(len(sorted_frame), bins)
            data_frame.loc[sorted_frame.index, "bin"] = bin_vector
            centers = (
                pd.Series(bin_vector, index=sorted_frame.index)
                .to_frame("bin")
                .join(sub_frame["ent"])
                .groupby("bin")["ent"]
                .median()
            )
            centers_by_dom[domain_name] = (
                centers.reindex(range(bins))
                .fillna(np.nanmedian(sub_frame["ent"]))
                .to_numpy()
            )

    return data_frame, centers_by_dom

# =========================
# Stats, GLM, CI helpers
# =========================


def _inv_logit(logits: np.ndarray) -> np.ndarray:
    """Inverse-logit transform used for AME calculations."""
    return 1.0 / (1.0 + np.exp(-logits))


def newcombe_diff_ci(
    k_shift: int,
    n_shift: int,
    k_control: int,
    n_control: int,
) -> Tuple[float, float, float]:
    """
    Newcombe interval for the difference in two proportions, in percentage points.
    """
    if n_shift == 0 or n_control == 0:
        return (np.nan, np.nan, np.nan)
    lower_shift, upper_shift = wilson_ci(k_shift, n_shift)
    lower_control, upper_control = wilson_ci(k_control, n_control)
    diff_pp = (k_shift / n_shift - k_control / n_control) * 100.0
    lower_pp = (lower_shift - upper_control) * 100.0
    upper_pp = (upper_shift - lower_control) * 100.0
    return (diff_pp, lower_pp, upper_pp)


def _newcombe_ame_stats(
    counts: Dict[int, int],
    corrects: Dict[int, int],
) -> Tuple[float, float, float, float]:
    diff_pp, lower_pp, upper_pp = newcombe_diff_ci(
        corrects[1],
        counts[1],
        corrects[0],
        counts[0],
    )
    return (diff_pp / 100.0, lower_pp / 100.0, upper_pp / 100.0, np.nan)


def _shift_group_stats(sub: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], int]:
    counts = {}
    corrects = {}
    for shift_value in (0, 1):
        mask = sub["shift"] == shift_value
        counts[shift_value] = int(mask.sum())
        corrects[shift_value] = (
            int(sub.loc[mask, "correct"].sum()) if counts[shift_value] > 0 else 0
        )
    grouped = sub.groupby("shift")["correct"]
    return counts, corrects, int(grouped.nunique().min())


def _count_shift_outcomes(sub: pd.DataFrame) -> ShiftBucketCounts:
    """Return totals and correct counts split by shift indicator."""
    total = int(len(sub))
    shift_mask = sub["shift"] == 1
    shift = int(shift_mask.sum())
    no_shift = total - shift
    correct_shift = int(sub.loc[shift_mask, "correct"].sum()) if shift else 0
    no_shift_mask = sub["shift"] == 0
    correct_no_shift = int(sub.loc[no_shift_mask, "correct"].sum()) if no_shift else 0
    return ShiftBucketCounts(
        total=total,
        shift=shift,
        no_shift=no_shift,
        correct_shift=correct_shift,
        correct_no_shift=correct_no_shift,
    )


def _fit_shift_glm(sub: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.glm(
            "correct ~ C(problem_id) + shift",
            data=sub,
            family=sm.families.Binomial(),
        )
        return model.fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["problem_id"]},
            maxiter=200,
        )


def _compute_ame_from_design(
    design_matrix: np.ndarray,
    shift_index: int,
    params: np.ndarray,
    cov: np.ndarray,
    bootstrap: BootstrapConfig,
) -> Tuple[float, float, float]:
    x_shift_1 = design_matrix.copy()
    x_shift_1[:, shift_index] = 1.0
    x_shift_0 = design_matrix.copy()
    x_shift_0[:, shift_index] = 0.0
    ame = float(
        np.mean(_inv_logit(x_shift_1 @ params) - _inv_logit(x_shift_0 @ params)),
    )
    rng = np.random.default_rng(bootstrap.seed)
    draws = rng.multivariate_normal(mean=params, cov=cov, size=bootstrap.draws)
    ame_draws = [
        float(
            np.mean(_inv_logit(x_shift_1 @ beta) - _inv_logit(x_shift_0 @ beta)),
        )
        for beta in draws
    ]
    lower, upper = np.percentile(ame_draws, [2.5, 97.5])
    return ame, float(lower), float(upper)


def _ame_from_res(
    res,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    names = list(res.model.exog_names)
    shift_index = (
        names.index("shift")
        if "shift" in names
        else max(index for index, name in enumerate(names) if "shift" in name)
    )
    params = res.params.to_numpy()
    cov = res.cov_params().to_numpy()
    ame, lower, upper = _compute_ame_from_design(
        design_matrix=res.model.exog,
        shift_index=shift_index,
        params=params,
        cov=cov,
        bootstrap=BootstrapConfig(draws=n_boot, seed=seed),
    )
    p_val = float(res.pvalues.get("shift", np.nan))
    return ame, lower, upper, p_val


def glm_ame_bucket(
    sub: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Estimate AME(shift) and its bootstrap CI within a single entropy bucket.
    """
    if sub["correct"].nunique() < 2 or sub["shift"].nunique() < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    counts, corrects, min_unique = _shift_group_stats(sub)
    if min_unique == 1:
        return _newcombe_ame_stats(counts, corrects)

    try:
        res = _fit_shift_glm(sub)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return _newcombe_ame_stats(counts, corrects)

    return _ame_from_res(res, n_boot, seed)

# ---------- NEW: Global regressions (per model) ----------

def _ame_from_fitted_glm(
    res,
    _dataset: pd.DataFrame,
    shift_var: str = "shift",
    n_boot: int = 300,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Compute the average marginal effect of a binary ``shift_var`` from a fitted GLM.

    Uses a simple difference-in-predicted-probabilities approach with
    multivariate-normal draws over the parameter covariance for bootstrap CIs.
    """
    names = list(res.model.exog_names)
    if shift_var in names:
        shift_index = names.index(shift_var)
    else:
        shift_index = max(index for index, name in enumerate(names) if shift_var in name)

    params = res.params.to_numpy()
    cov = res.cov_params().to_numpy()
    ame, lower, upper = _compute_ame_from_design(
        design_matrix=res.model.exog,
        shift_index=shift_index,
        params=params,
        cov=cov,
        bootstrap=BootstrapConfig(draws=n_boot, seed=seed),
    )
    p_val = float(res.pvalues.get(shift_var, np.nan))
    return ame, lower, upper, p_val


def _basic_regression_stats(data_frame: pd.DataFrame) -> Dict[str, float]:
    n_total = int(len(data_frame))
    num_shift = int((data_frame["shift"] == 1).sum())
    num_no_shift = n_total - num_shift
    num_correct_shift = (
        int(data_frame.loc[data_frame["shift"] == 1, "correct"].sum())
        if num_shift > 0
        else 0
    )
    num_correct_no_shift = (
        int(data_frame.loc[data_frame["shift"] == 0, "correct"].sum())
        if num_no_shift > 0
        else 0
    )
    share_shift = (num_shift / n_total) if n_total else np.nan
    acc_shift = (num_correct_shift / num_shift) if num_shift else np.nan
    delta_pp = (
        (num_correct_shift / num_shift - num_correct_no_shift / num_no_shift) * 100.0
        if (num_shift and num_no_shift)
        else np.nan
    )
    return {
        "N": n_total,
        "share_shift": share_shift,
        "acc_shift": acc_shift,
        "delta_pp": delta_pp,
    }


def _fit_glm_clustered(data_frame: pd.DataFrame, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_model = smf.glm(
            formula,
            data=data_frame,
            family=sm.families.Binomial(),
        )
        return glm_model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data_frame["problem_id"]},
            maxiter=200,
        )


def _regression_row(
    kind: str,
    stats: Dict[str, float],
    model_label: str,
    data_frame: pd.DataFrame,
    fitted,
) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "model": model_label,
        "N": stats["N"],
        "share_shift": stats["share_shift"],
        "acc_shift": stats["acc_shift"],
        "delta_pp": stats["delta_pp"],
        "AME": np.nan,
        "p_value": np.nan,
    }
    if fitted is not None:
        ame, _, _, p_val = _ame_from_fitted_glm(fitted, data_frame)
        row["AME"] = ame
        row["p_value"] = p_val
    return row


def _attempt_regression(
    data_frame: pd.DataFrame,
    stats: Dict[str, float],
    model_label: str,
    kind: str,
    formula: str,
) -> Dict[str, Any]:
    try:
        fitted = _fit_glm_clustered(data_frame, formula)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        fitted = None
    return _regression_row(kind, stats, model_label, data_frame, fitted)


def _build_strata_tables(
    df_binned: pd.DataFrame,
) -> Tuple[List[np.ndarray], pd.Series]:
    """Return CMH contingency tables and a mask of strata with both shift labels."""
    tables: List[np.ndarray] = []
    keep_mask = pd.Series(False, index=df_binned.index)
    for _, sub in df_binned.groupby(
        ["domain", "bin"],
        sort=False,
    ):
        shift1_count = int((sub["shift"] == 1).sum())
        shift0_count = int((sub["shift"] == 0).sum())
        if shift1_count == 0 or shift0_count == 0:
            continue
        shift1_correct = int(sub.loc[sub["shift"] == 1, "correct"].sum())
        shift0_correct = int(sub.loc[sub["shift"] == 0, "correct"].sum())
        tables.append(
            np.array(
                [
                    [shift1_correct, shift1_count - shift1_correct],
                    [shift0_correct, shift0_count - shift0_correct],
                ],
                dtype=int,
            ),
        )
        keep_mask.loc[sub.index] = True
    return tables, keep_mask


def _compute_cmh_summary(
    tables: List[np.ndarray],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute CMH statistics and a corresponding row dict."""
    if not tables:
        return None, None
    stratified_table = StratifiedTable(tables)
    cmh = stratified_table.test_null_odds()
    cmh_stat = float(cmh.statistic)
    cmh_p = float(cmh.pvalue)
    or_pooled = float(stratified_table.oddsratio_pooled)
    or_lo, or_hi = [
        float(value) for value in stratified_table.oddsratio_pooled_confint()
    ]
    summary = {
        "test": "CMH",
        "statistic": cmh_stat,
        "pvalue": cmh_p,
        "pooled_or": or_pooled,
        "pooled_or_lo": or_lo,
        "pooled_or_hi": or_hi,
        "n_strata": len(tables),
    }
    row = {
        "test": "CMH",
        "domain": "ALL",
        "strata": len(tables),
        "stat": cmh_stat,
        "p": cmh_p,
        "effect": "pooled OR",
        "est": or_pooled,
        "lo": or_lo,
        "hi": or_hi,
    }
    return summary, row


def _likelihood_ratio_stats(full_model, reduced_model) -> Tuple[float, int, float]:
    """Return chi-square statistic, df, and p-value for nested GLMs."""
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    lr_df = len(full_model.params) - len(reduced_model.params)
    return float(lr_stat), int(lr_df), float(chi2.sf(lr_stat, lr_df))


def _shift_odds_ratio(
    glm_result,
    param_name: str = "shift",
) -> Tuple[float, float, float]:
    """Compute an odds-ratio and CI for ``param_name`` if present."""
    if param_name not in glm_result.params.index:
        return (np.nan, np.nan, np.nan)
    coef = float(glm_result.params[param_name])
    se_value = float(glm_result.bse[param_name])
    delta = 1.96 * se_value
    return (
        float(np.exp(coef)),
        float(np.exp(coef - delta)),
        float(np.exp(coef + delta)),
    )


def _glm_lrt_and_anova_rows(
    sub_all: pd.DataFrame,
    stratum_count: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fit GLM shift models with stratum FE and build output rows."""
    rows: List[Dict[str, Any]] = []
    if (
        sub_all.empty
        or sub_all["shift"].nunique() < 2
        or sub_all["correct"].nunique() < 2
    ):
        return None, rows

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_stratum_only = smf.glm(
            "correct ~ C(stratum)",
            data=sub_all,
            family=sm.families.Binomial(),
        ).fit()
        full_glm = smf.glm(
            "correct ~ C(stratum) + shift",
            data=sub_all,
            family=sm.families.Binomial(),
        ).fit()
    lr_stat, lr_df, lr_p = _likelihood_ratio_stats(full_glm, glm_stratum_only)
    or_hat, or_lo, or_hi = _shift_odds_ratio(full_glm)
    glm_summary = {
        "test": "GLM_LRT",
        "statistic": float(lr_stat),
        "df": lr_df,
        "pvalue": lr_p,
        "shift_or": or_hat,
        "shift_or_lo": or_lo,
        "shift_or_hi": or_hi,
    }
    rows.append({
        "test": "GLM_LRT",
        "domain": "ALL",
        "strata": stratum_count,
        "stat": float(lr_stat),
        "p": lr_p,
        "effect": "shift OR (Wald CI)",
        "est": or_hat,
        "lo": or_lo,
        "hi": or_hi,
    })

    rows.append(_anova_row_shift_vs_strata(sub_all, stratum_count))
    return glm_summary, rows


def _anova_row_shift_vs_strata(sub_all: pd.DataFrame, stratum_count: int) -> Dict[str, Any]:
    """Compare models with vs. without stratum FE to build a reporting row."""
    base_row = {
        "test": "ANOVA_strata_vs_shift",
        "domain": "ALL",
        "strata": stratum_count,
        "est": np.nan,
        "lo": np.nan,
        "hi": np.nan,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm_shift_only = smf.glm(
                "correct ~ C(shift)",
                data=sub_all,
                family=sm.families.Binomial(),
            ).fit()
            glm_stratum_shift = smf.glm(
                "correct ~ C(stratum) + C(shift)",
                data=sub_all,
                family=sm.families.Binomial(),
            ).fit()
        lr_stat = 2.0 * (glm_stratum_shift.llf - glm_shift_only.llf)
        lr_df = len(glm_stratum_shift.params) - len(glm_shift_only.params)
        lr_p = float(chi2.sf(lr_stat, lr_df))
        return {
            **base_row,
            "stat": float(lr_stat),
            "p": lr_p,
            "effect": f"add C(stratum) vs C(shift) (Δdf={lr_df})",
        }
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return {
            **base_row,
            "stat": np.nan,
            "p": np.nan,
            "effect": "add C(stratum) vs C(shift)",
        }


def _domain_ztest_rows(df_binned: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute domain-specific two-proportion z-test rows."""
    domain_rows: List[Dict[str, Any]] = []
    for dom, sub in df_binned.groupby("domain", sort=False):
        shift1_count = int((sub["shift"] == 1).sum())
        shift0_count = int((sub["shift"] == 0).sum())
        if shift1_count == 0 or shift0_count == 0:
            domain_rows.append(
                {
                    "test": "Z_2prop",
                    "domain": dom,
                    "strata": sub["bin"].nunique(),
                    "stat": np.nan,
                    "p": np.nan,
                    "effect": "p1 - p0 (pp)",
                    "est": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                },
            )
            continue
        shift1_correct = int(sub.loc[sub["shift"] == 1, "correct"].sum())
        shift0_correct = int(sub.loc[sub["shift"] == 0, "correct"].sum())
        z_stat, p_val = proportions_ztest(
            [shift1_correct, shift0_correct],
            [shift1_count, shift0_count],
            alternative="two-sided",
            prop_var=False,
        )
        diff_pp, lo_pp, hi_pp = newcombe_diff_ci(
            shift1_correct,
            shift1_count,
            shift0_correct,
            shift0_count,
        )
        domain_rows.append(
            {
                "test": "Z_2prop",
                "domain": dom,
                "strata": sub["bin"].nunique(),
                "stat": float(z_stat),
                "p": float(p_val),
                "effect": "p1 - p0 (pp)",
                "est": diff_pp,
                "lo": lo_pp,
                "hi": hi_pp,
            },
        )
    return domain_rows


def _pooled_ztest_row(df_binned: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Compute the pooled two-proportion z-test row if counts are available."""
    pooled = df_binned.copy()
    shift1_count = int((pooled["shift"] == 1).sum())
    shift0_count = int((pooled["shift"] == 0).sum())
    if shift1_count == 0 or shift0_count == 0:
        return None
    shift1_correct = int(pooled.loc[pooled["shift"] == 1, "correct"].sum())
    shift0_correct = int(pooled.loc[pooled["shift"] == 0, "correct"].sum())
    z_stat, p_val = proportions_ztest(
        [shift1_correct, shift0_correct],
        [shift1_count, shift0_count],
        alternative="two-sided",
        prop_var=False,
    )
    diff_pp, lo_pp, hi_pp = newcombe_diff_ci(
        shift1_correct,
        shift1_count,
        shift0_correct,
        shift0_count,
    )
    return {
        "test": "Z_2prop",
        "domain": "ALL",
        "strata": df_binned["bin"].nunique() * df_binned["domain"].nunique(),
        "stat": float(z_stat),
        "p": float(p_val),
        "effect": "p1 - p0 (pp)",
        "est": diff_pp,
        "lo": lo_pp,
        "hi": hi_pp,
    }


def run_model_regressions(
    data_frame: pd.DataFrame,
    out_dir: str,
    model_label: str,
    slug_base: str,
) -> pd.DataFrame:
    """
    Runs:
      (a) correct ~ C(problem_id) + step_std + shift
      (b) correct ~ C(problem_id) + C(temp_cat) + shift
    Returns tidy df; also writes CSV.
    """
    if data_frame.empty:
        return pd.DataFrame()

    data_frame = data_frame.copy()
    data_frame["step_std"] = (
        (data_frame["step"] - data_frame["step"].mean())
        / max(1e-9, data_frame["step"].std(ddof=0))
    )
    data_frame["temp_cat"] = data_frame["temp"].astype(str)

    stats = _basic_regression_stats(data_frame)
    rows = [
        _attempt_regression(
            data_frame=data_frame,
            stats=stats,
            model_label=model_label,
            kind="training_stage",
            formula="correct ~ C(problem_id) + step_std + shift",
        ),
        _attempt_regression(
            data_frame=data_frame,
            stats=stats,
            model_label=model_label,
            kind="temperature_controls",
            formula="correct ~ C(problem_id) + C(temp_cat) + shift",
        ),
    ]

    out = pd.DataFrame(rows)
    out_csv = os.path.join(
        out_dir,
        f"global_regressions__{slug_base}__{model_label.replace(' ', '_')}.csv",
    )
    out.to_csv(out_csv, index=False)
    print("[global-reg] saved:", out_csv)
    return out

# ---------- ANOVA/CMH + domain z-tests ----------

def run_anova_and_cmh(df_binned: pd.DataFrame, out_dir: str, slug: str) -> pd.DataFrame:
    """
    Summarize shift correctness effects via CMH, GLM LRT, and domain-wise tests.
    """
    rows: List[Dict[str, Any]] = []
    tables, keep_mask = _build_strata_tables(df_binned)
    sub_all = df_binned.loc[keep_mask].copy()
    sub_all["stratum"] = sub_all["domain"].astype(str) + ":" + sub_all["bin"].astype(str)
    stratum_count = len(tables)

    cmh_summary, cmh_row = _compute_cmh_summary(tables)
    if cmh_row is not None:
        rows.append(cmh_row)

    glm_summary, glm_rows = _glm_lrt_and_anova_rows(sub_all, stratum_count)
    rows.extend(glm_rows)

    rows.extend(_domain_ztest_rows(df_binned))
    pooled_row = _pooled_ztest_row(df_binned)
    if pooled_row is not None:
        rows.append(pooled_row)

    out = pd.DataFrame(
        rows,
        columns=["test", "domain", "strata", "stat", "p", "effect", "est", "lo", "hi"],
    )
    out_path = os.path.join(out_dir, f"anova_cmh_summary__{slug}.csv")
    out.to_csv(out_path, index=False)
    print("\n[ANOVA/CMH] saved:", out_path)
    _log_anova_details(out, cmh_summary, glm_summary)
    return out


def _log_anova_details(
    result_df: pd.DataFrame,
    cmh_summary: Optional[Dict[str, Any]],
    glm_summary: Optional[Dict[str, Any]],
) -> None:
    """Log CMH/GLM/ANOVA summaries to stdout."""
    if cmh_summary is not None:
        print(
            f"  CMH: chi2={cmh_summary['statistic']:.3f}, "
            f"p={cmh_summary['pvalue']:.3g}, pooled OR={cmh_summary['pooled_or']:.3f} "
            f"[{cmh_summary['pooled_or_lo']:.3f}, {cmh_summary['pooled_or_hi']:.3f}]  "
            f"(n_strata={cmh_summary['n_strata']})",
        )
    if glm_summary is not None:
        print(
            f"  GLM LRT: chi2={glm_summary['statistic']:.3f} (df={glm_summary['df']}), "
            f"p={glm_summary['pvalue']:.3g}; "
            f"shift OR={glm_summary['shift_or']:.3f} "
            f"[{glm_summary['shift_or_lo']:.3f}, {glm_summary['shift_or_hi']:.3f}]",
        )
    mask = (result_df["test"] == "ANOVA_strata_vs_shift") & (result_df["domain"] == "ALL")
    if not result_df.loc[mask].empty:
        entry = result_df.loc[mask].iloc[0]
        print(
            "  ANOVA (stratum FE vs shift-only): "
            f"chi2={entry['stat']:.3f}, p={entry['p']:.3g}; {entry['effect']}",
        )


def print_anova_quick(anova_df: pd.DataFrame):
    """Print a condensed textual summary of ANOVA/CMH results."""
    if anova_df.empty:
        print("[ANOVA/CMH] No strata with both S=0 and S=1; skipping.")
        return
    print("\n====== Quick ANOVA/CMH Summary ======")
    cmh = anova_df[
        (anova_df["test"] == "CMH")
        & (anova_df["domain"] == "ALL")
    ]
    if not cmh.empty:
        cmh_row = cmh.iloc[0]
        print(
            "CMH (pooled across strata): "
            f"chi2={cmh_row['stat']:.3f}, p={cmh_row['p']:.3g}; "
            f"{cmh_row['effect']}={cmh_row['est']:.3f} "
            f"[{cmh_row['lo']:.3f}, {cmh_row['hi']:.3f}]",
        )

    glm = anova_df[
        (anova_df["test"] == "GLM_LRT")
        & (anova_df["domain"] == "ALL")
    ]
    if not glm.empty:
        glm_row = glm.iloc[0]
        print(
            "GLM LRT (with stratum FE): "
            f"chi2={glm_row['stat']:.3f}, p={glm_row['p']:.3g}; "
            f"{glm_row['effect']}={glm_row['est']:.3f} "
            f"[{glm_row['lo']:.3f}, {glm_row['hi']:.3f}]",
        )

    # NEW quick line for the true ANOVA
    anova = anova_df[
        (anova_df["test"] == "ANOVA_strata_vs_shift")
        & (anova_df["domain"] == "ALL")
    ]
    if not anova.empty:
        anova_row = anova.iloc[0]
        print(
            "ANOVA (stratum FE vs shift-only): "
            f"chi2={anova_row['stat']:.3f}, p={anova_row['p']:.3g}; "
            f"{anova_row['effect']}",
        )

    for dom in ["Crossword", "Math", "Carpark", "ALL"]:
        domain_rows = anova_df[
            (anova_df["test"] == "Z_2prop")
            & (anova_df["domain"] == dom)
        ]
        if not domain_rows.empty:
            summary_row = domain_rows.iloc[0]
            print(
                f"{dom:9s}  Δpp = {summary_row['est']:.2f} "
                f"[{summary_row['lo']:.2f}, {summary_row['hi']:.2f}],  "
                f"z={summary_row['stat']:.3f}, p={summary_row['p']:.3g}",
            )
    print("=====================================\n")


# =================
# Plotting helpers
# =================

def _safe_yerr(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Build a non-negative y-error array for Matplotlib from low/high bounds.
    """
    low_clipped = np.nan_to_num(np.maximum(0, low))
    high_clipped = np.nan_to_num(np.maximum(0, high))
    return np.vstack([low_clipped, high_clipped])

# =========
#   Runner
# =========


def _build_binned_dataframe(
    input_df: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """
    Apply entropy binning to the combined rows for one model/metric.

    Returns ``(df_binned, centers_by_dom, edges_global)`` where ``edges_global``
    is a representative global edge array used for combined plots.
    """
    if args.equal_n_bins:
        df_binned, centers_by_dom = assign_bins_equal_count(
            input_df,
            bins=args.bins,
            scope=args.bin_scope,
            tie_break=args.tie_break,
            seed=args.random_seed,
        )
        edges_global = np.arange(args.bins + 1, dtype=float)
        return df_binned, centers_by_dom, edges_global

    if args.fixed_bins:
        fixed_edges = _parse_fixed_edges(args.fixed_bins)
        edges_by_dom = {dom: fixed_edges for dom in input_df["domain"].unique()}
        df_binned, centers_by_dom = assign_bins_fixed(
            input_df,
            edges_by_dom,
            last_center_offset=args.last_bin_center_offset,
        )
        edges_global = fixed_edges
        return df_binned, centers_by_dom, edges_global

    edges_by_dom = build_edges(
        input_df,
        args.bins,
        args.binning,
        args.bin_scope,
    )
    df_binned, centers_by_dom = assign_bins(input_df, edges_by_dom)
    if args.bin_scope == "global":
        edges_global = list(edges_by_dom.values())[0]
    else:
        x_vals = input_df["ent"].to_numpy()
        if args.binning == "uniform":
            edges_global = np.linspace(
                np.nanmin(x_vals),
                np.nanmax(x_vals),
                args.bins + 1,
            )
        else:
            edges_global = np.quantile(
                x_vals,
                np.linspace(0, 1, args.bins + 1),
            )
    return df_binned, centers_by_dom, edges_global


def _compute_aha_bucket_rows(
    df_binned: pd.DataFrame,
    model_label: str,
) -> pd.DataFrame:
    """
    Compute per-domain, per-bin Aha vs. no-Aha accuracy and odds statistics.
    """
    def _prob_to_odds(probability: float) -> float:
        clipped = float(np.clip(probability, 1e-9, 1 - 1e-9))
        return clipped / (1 - clipped)

    def _row_for_group(
        domain: str,
        bin_index: int,
        group_label: str,
        subset_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        num_samples = int(len(subset_df))
        num_correct = int(subset_df["correct"].sum())
        if num_samples > 0:
            accuracy = num_correct / num_samples
            lower_ci, upper_ci = wilson_ci(num_correct, num_samples)
            odds = (num_correct + 0.5) / (
                (num_samples - num_correct) + 0.5
            )
            odds_lower = _prob_to_odds(lower_ci)
            odds_upper = _prob_to_odds(upper_ci)
        else:
            accuracy = np.nan
            lower_ci = np.nan
            upper_ci = np.nan
            odds = np.nan
            odds_lower = np.nan
            odds_upper = np.nan

        return {
            "model": model_label,
            "domain": domain,
            "bin": int(bin_index),
            "grp": group_label,
            "n": num_samples,
            "k": num_correct,
            "acc": accuracy,
            "lo": lower_ci,
            "hi": upper_ci,
            "odds": odds,
            "odds_lo": odds_lower,
            "odds_hi": odds_upper,
        }

    aha_rows: List[Dict[str, Any]] = []
    for (domain, bin_index), subset in df_binned.groupby(["domain", "bin"], sort=False):
        for group_label, mask in [
            ("no", subset["shift"] == 0),
            ("yes", subset["shift"] == 1),
        ]:
            group_df = subset[mask]
            aha_rows.append(
                _row_for_group(
                    domain=domain,
                    bin_index=int(bin_index),
                    group_label=group_label,
                    subset_df=group_df,
                ),
            )

    return pd.DataFrame(aha_rows)


def _compute_effect_bucket_rows(
    df_binned: pd.DataFrame,
    model_label: str,
    num_bootstrap: int,
) -> pd.DataFrame:
    """
    Compute per-bin raw and model-based Aha effects for each domain.
    """
    effect_rows: List[Dict[str, Any]] = []
    for (dom, bin_index), sub in df_binned.groupby(["domain", "bin"], sort=False):
        counts = _count_shift_outcomes(sub)
        raw_pp, raw_lo, raw_hi = newcombe_diff_ci(
            counts.correct_shift,
            counts.shift,
            counts.correct_no_shift,
            counts.no_shift,
        )
        ame, ame_lower, ame_upper, p_val = glm_ame_bucket(
            sub[["problem_id", "correct", "shift"]],
            n_boot=num_bootstrap,
        )
        effect_rows.append(
            {
                "model": model_label,
                "domain": dom,
                "bin": int(bin_index),
                "n": counts.total,
                "n_shift": counts.shift,
                "n_noshift": counts.no_shift,
                "raw_pp": raw_pp,
                "raw_lo": raw_lo,
                "raw_hi": raw_hi,
                "ame_pp": (ame * 100.0 if np.isfinite(ame) else np.nan),
                "ame_lo_pp": (
                    ame_lower * 100.0 if np.isfinite(ame_lower) else np.nan
                ),
                "ame_hi_pp": (
                    ame_upper * 100.0 if np.isfinite(ame_upper) else np.nan
                ),
                "p_shift": p_val,
            },
        )
    return pd.DataFrame(effect_rows)


def _confidence_bounds_for_counts(
    correct_counts: np.ndarray,
    total_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson CIs for counts arrays (NaN when total==0)."""
    lower_ci = np.empty_like(correct_counts, dtype=float)
    upper_ci = np.empty_like(correct_counts, dtype=float)
    for index, (correct_i, total_i) in enumerate(zip(correct_counts, total_counts)):
        if total_i > 0:
            lower_i, upper_i = wilson_ci(int(correct_i), int(total_i))
        else:
            lower_i = upper_i = np.nan
        lower_ci[index] = lower_i
        upper_ci[index] = upper_i
    return lower_ci, upper_ci


def _aggregate_group_frame(
    aha_df: pd.DataFrame,
    group_label: str,
    bin_indices: np.ndarray,
) -> pd.DataFrame:
    """Aggregate accuracy/CIs for a single grp label across bins."""
    grouped = (
        aha_df[aha_df["grp"] == group_label]
        .groupby("bin", as_index=False)
        .agg({"n": "sum", "k": "sum"})
        .set_index("bin")
        .reindex(bin_indices, fill_value=0)
    )
    totals = grouped["n"].to_numpy(dtype=float)
    correct = grouped["k"].to_numpy(dtype=float)
    accuracy = np.divide(
        correct,
        totals,
        out=np.full_like(totals, np.nan),
        where=totals > 0,
    )
    lower, upper = _confidence_bounds_for_counts(correct, totals)
    return pd.DataFrame(
        {
            "bin": bin_indices,
            "grp": group_label,
            "n": totals,
            "k": correct,
            "acc": accuracy,
            "lo": lower,
            "hi": upper,
        },
    )


def aggregate_combined(
    aha_df: pd.DataFrame,
    edges_global: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Aggregate per-domain Aha rows into a global combined-by-bin DataFrame.
    """
    bin_indices = np.arange(len(edges_global) - 1)
    frames = [
        _aggregate_group_frame(aha_df, label, bin_indices).reset_index(drop=True)
        for label in ("no", "yes")
    ]
    combined_df = pd.concat(frames, ignore_index=True)
    centers = 0.5 * (edges_global[:-1] + edges_global[1:])
    return combined_df, centers

@dataclass(frozen=True)
class RunForModelConfig:
    """Configuration for running all bucket-level analyses for a single model."""

    metric: str
    args: Any
    roots_for_model: Dict[float, Dict[str, str]]
    model_label: str
    combined_mode: str
    min_step: Optional[int]
    max_step: Optional[int]


@dataclass(frozen=True)
class BucketOutputPaths:
    """File paths for bucket-level CSV outputs."""

    aha_csv: str
    odds_csv: str
    effects_csv: str
    combined_csv: str


def _load_frames_for_model(
    cfg: RunForModelConfig,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for temp_value in sorted(cfg.roots_for_model):
        domain_map = cfg.roots_for_model[temp_value]
        for domain in cfg.args.domains:
            path = domain_map.get(domain)
            if path is None:
                continue
            row_cfg = RowLoadConfig(
                entropy_source=cfg.metric,
                allow_fallback=cfg.args.allow_metric_fallback,
                carpark_op=cfg.args.carpark_success_op,
                carpark_threshold=cfg.args.carpark_soft_threshold,
                gpt_mode=cfg.args.gpt_mode,
                verbose=cfg.args.verbose,
                combined_mode=cfg.combined_mode,
                min_step=cfg.min_step,
                max_step=cfg.max_step,
                temp=temp_value,
                model_label=cfg.model_label,
            )
            rows_df = load_rows(
                path,
                cfg.args.split,
                domain,
                row_cfg,
            )
            if not rows_df.empty:
                frames.append(rows_df)
    return frames


def _build_bucket_slug(cfg: RunForModelConfig) -> str:
    """Return a descriptive slug for CSV/figure outputs."""
    metric_key = (
        cfg.metric
        if cfg.metric != "answer_plus_think"
        else f"answerPlusThink_{cfg.combined_mode}"
    )
    bound_slug = f"stepMax{cfg.max_step}" if cfg.max_step is not None else "stepAll"
    bin_slug = (
        "equalN"
        if cfg.args.equal_n_bins
        else (
            "fixedbins"
            if cfg.args.fixed_bins
            else f"{cfg.args.binning}_{cfg.args.bin_scope}_{cfg.args.bins}bins"
        )
    )
    return f"{cfg.model_label.replace(' ', '_')}_allT_{metric_key}_{bin_slug}_{bound_slug}"


def _write_bucket_outputs(
    aha_df: pd.DataFrame,
    eff_df: pd.DataFrame,
    aha_combined: pd.DataFrame,
    out_dir: str,
    slug: str,
) -> BucketOutputPaths:
    """Write bucket CSVs and return the written paths."""
    paths = BucketOutputPaths(
        aha_csv=os.path.join(out_dir, f"aha_acc_buckets__{slug}.csv"),
        odds_csv=os.path.join(out_dir, f"aha_odds_buckets__{slug}.csv"),
        effects_csv=os.path.join(out_dir, f"effects_buckets__{slug}.csv"),
        combined_csv=os.path.join(out_dir, f"aha_acc_buckets__{slug}_COMBINED.csv"),
    )
    aha_df.to_csv(paths.aha_csv, index=False)
    aha_df.to_csv(paths.odds_csv, index=False)
    eff_df.to_csv(paths.effects_csv, index=False)
    aha_combined.to_csv(paths.combined_csv, index=False)
    return paths


def run_for_model(cfg: RunForModelConfig):
    """Collect all bucket tables and regressions for the provided model/metric."""
    frames = _load_frames_for_model(cfg)
    if not frames:
        print(f"[warn] no rows for {cfg.model_label} / metric={cfg.metric}. Skipping.")
        return

    samples_df = pd.concat(frames, ignore_index=True)
    df_binned, _, edges_global = _build_binned_dataframe(samples_df, cfg.args)
    out_dir = cfg.args.out_dir or os.path.join(cfg.args.scan_root, "temperature_effects")
    os.makedirs(out_dir, exist_ok=True)
    aha_df = _compute_aha_bucket_rows(df_binned, cfg.model_label)
    eff_df = _compute_effect_bucket_rows(df_binned, cfg.model_label, cfg.args.n_boot)
    aha_combined, _ = aggregate_combined(aha_df, edges_global)
    slug = _build_bucket_slug(cfg)
    bucket_paths = _write_bucket_outputs(aha_df, eff_df, aha_combined, out_dir, slug)

    # --- ANOVA/CMH ---
    if not cfg.args.no_anova:
        print_anova_quick(
            run_anova_and_cmh(df_binned=df_binned, out_dir=out_dir, slug=slug)
        )

    # --- Global regressions (per model) ---
    run_model_regressions(
        samples_df,
        out_dir=out_dir,
        model_label=cfg.model_label,
        slug_base=slug,
    )

    max_step_label = cfg.max_step if cfg.max_step is not None else "∞"
    print(f"\n[{cfg.model_label} | metric={cfg.metric}, steps ≤ {max_step_label}] Saved:")
    print("  Aha vs No-Aha accuracy (per-domain) CSV :", bucket_paths.aha_csv)
    print("  Aha vs No-Aha odds     (per-domain) CSV :", bucket_paths.odds_csv)
    print("  Aha vs No-Aha accuracy (COMBINED)  CSV :", bucket_paths.combined_csv)
    print("  Effects per bucket CSV               :", bucket_paths.effects_csv)
    reg_csv = os.path.join(out_dir, f"global_regressions__{slug}.csv").replace("//", "/")
    print("  Global regressions CSV               :", reg_csv)

def main():
    """CLI entry-point for temperature-bucket uncertainty analyses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=str, default="results")
    parser.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.05, 0.3, 0.7])
    parser.add_argument("--low_alias", type=float, default=0.3)
    parser.add_argument("--split", type=str, default="test")

    # Models/domains
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen7b", "llama8b"],
        choices=["qwen7b", "llama8b"],
        help="Which models to include.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Math"],
        choices=["Math", "Crossword", "Carpark"],
        help="Domains to include (default Math).",
    )

    # Metrics
    parser.add_argument(
        "--metric",
        choices=["entropy_answer", "entropy", "entropy_think", "answer_plus_think"],
        default=None,
        help="Single metric to run (overridden by --metrics).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["entropy_answer", "entropy", "entropy_think", "answer_plus_think"],
        help="Run these metrics in one go; overrides --metric.",
    )
    parser.add_argument(
        "--combined_mode",
        choices=["sum", "mean"],
        default="sum",
        help="For answer_plus_think: sum or mean of (answer, think).",
    )

    # Step bounds (max is hard-capped to 1000)
    parser.add_argument(
        "--min_step",
        type=int,
        default=None,
        help="Lower bound on step (inclusive).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=1000,
        help="Upper bound on step (inclusive; hard cap = 1000).",
    )

    # Fixed bins override
    parser.add_argument(
        "--fixed_bins",
        type=str,
        default=None,
        help=(
            "Comma-separated entropy edges, e.g. '0,0.5,1,2,inf'. "
            "Overrides --bins/--binning/--bin_scope."
        ),
    )
    parser.add_argument(
        "--last_bin_center_offset",
        type=float,
        default=0.5,
        help="Center offset used to place the open-ended last bin (if top edge is inf).",
    )

    # Equal-count bins (rank-based)
    parser.add_argument(
        "--equal_n_bins",
        action="store_true",
        help=(
            "Force equal-count bins (e.g., quartiles if --bins 4). "
            "Ignores --fixed_bins and numeric quantile edges."
        ),
    )
    parser.add_argument(
        "--tie_break",
        choices=["stable", "random"],
        default="stable",
        help="How to break ties when many entropies are identical.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed used when --tie_break=random",
    )

    parser.add_argument("--allow_metric_fallback", action="store_true")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--binning",
        choices=["uniform", "quantile"],
        default="uniform",
    )
    parser.add_argument(
        "--bin_scope",
        choices=["global", "domain"],
        default="global",
    )
    parser.add_argument(
        "--carpark_success_op",
        choices=["gt", "ge", "eq"],
        default="ge",
    )
    parser.add_argument("--carpark_soft_threshold", type=float, default=0.1)
    parser.add_argument(
        "--gpt_mode",
        choices=["canonical", "broad"],
        default="canonical",
    )
    parser.add_argument("--n_boot", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no_anova",
        action="store_true",
        help="Skip CMH/GLM and two-proportion tests.",
    )

    args = parser.parse_args()

    hard_max_step = 1000
    eff_max = hard_max_step if args.max_step is None else min(args.max_step, hard_max_step)
    if args.max_step is None or args.max_step > hard_max_step:
        print(f"[info] Capping max_step to {eff_max} (hard cap = {hard_max_step}).")

    discovery_config = RootDiscoveryConfig(
        temps=args.temps,
        low_alias=args.low_alias,
        want_models=args.models,
        want_domains=args.domains,
        verbose=args.verbose,
    )
    mapping = discover_roots_7b8b(args.scan_root, discovery_config)
    if not mapping:
        raise SystemExit("[error] no 7B/8B temp dirs found under scan_root.")

    metrics_to_run = (
        args.metrics
        if args.metrics
        else (
            [args.metric]
            if args.metric
            else [
                "entropy_answer",
                "entropy",
                "entropy_think",
                "answer_plus_think",
            ]
        )
    )

    for model_key in args.models:
        model_label = MODEL_LABELS.get(model_key, model_key)
        roots_for_model = mapping.get(model_key, {})
        if not roots_for_model:
            print(f"[warn] No runs found for {model_label}.")
            continue
        for metric in metrics_to_run:
            config = RunForModelConfig(
                metric=metric,
                args=args,
                roots_for_model=roots_for_model,
                model_label=model_label,
                combined_mode=args.combined_mode,
                min_step=args.min_step,
                max_step=eff_max,
            )
            run_for_model(config)

if __name__ == "__main__":
    main()
