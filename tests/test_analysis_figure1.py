import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.core.figure_1_data import (
    bootstrap_problem_ratio,
    build_positive_delta_flags,
    build_problem_step,
    fit_trend_wls,
    load_pass1_samples_multi,
    mark_formal_pairs,
    parse_float_list,
)
from src.analysis.core.figure_1_export import (
    ExportDestinations,
    FormalExportConfig,
    FormalThresholds,
    GptFilterConfig,
    export_formal_aha_json_with_text,
)


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def test_load_pass1_samples_multi_builds_rows_from_pass1(tmp_path):
    file_path = tmp_path / "step0005.jsonl"
    _write_jsonl(
        file_path,
        [
            {
                # Step omitted to ensure filename hint is used.
                "problem": "p1",
                "pass1": {
                    "is_correct_pred": 1,
                    "has_reconsider_cue": 1,
                    "change_way_of_thinking": 1,
                },
            },
            {
                # Missing pass1 payload should be ignored.
                "problem": "p2",
            },
        ],
    )
    df = load_pass1_samples_multi(
        {"Math": [str(file_path)]},
        gpt_keys=["change_way_of_thinking"],
        gpt_subset_native=False,
    )
    assert df.shape[0] == 1
    row = df.iloc[0].to_dict()
    assert row["domain"] == "Math"
    assert row["step"] == 5  # pulled from filename
    assert row["problem"] == "p1"
    assert row["aha_native"] == 1
    assert row["aha_gpt"] == 1
    assert row["correct"] == 1


def test_build_problem_step_aggregates_rates():
    df = pd.DataFrame(
        [
            {"domain": "Math", "step": 1, "problem": "p1", "aha_native": 0, "aha_gpt": 1, "correct": 1},
            {"domain": "Math", "step": 1, "problem": "p1", "aha_native": 1, "aha_gpt": 0, "correct": 0},
            {"domain": "Math", "step": 2, "problem": "p2", "aha_native": 1, "aha_gpt": 1, "correct": 1},
        ]
    )
    problem_step = build_problem_step(df)
    first = problem_step.loc[problem_step["problem"] == "p1"].iloc[0]
    assert first["n_samples"] == 2
    assert first["freq_correct"] == pytest.approx(0.5)
    assert first["aha_any_gpt"] == 1
    assert first["aha_rate_gpt"] == pytest.approx(0.5)
    assert first["aha_any_native"] == 1
    assert first["p_correct_given_shift"] == pytest.approx(1.0)


def test_mark_formal_pairs_flags_shift_rows():
    data = pd.DataFrame(
        [
            {
                "domain": "Math",
                "problem": "p1",
                "step": 1,
                "freq_correct": 0.1,
                "aha_rate_gpt": 0.1,
                "aha_any_gpt": 0,
                "p_correct_given_shift": np.nan,
                "n_samples": 2,
            },
            {
                "domain": "Math",
                "problem": "p1",
                "step": 2,
                "freq_correct": 0.1,
                "aha_rate_gpt": 0.1,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.5,
                "n_samples": 1,
            },
        ]
    )
    with pytest.raises(ValueError):
        mark_formal_pairs(pd.DataFrame([{"step": 1}]))
    flagged = mark_formal_pairs(data, min_prior_steps=1)
    assert flagged.loc[flagged["step"] == 1, "aha_formal"].iloc[0] == 0
    assert flagged.loc[flagged["step"] == 2, "aha_formal"].iloc[0] == 1


def test_bootstrap_problem_ratio_handles_missing_bootstraps():
    df = pd.DataFrame(
        [
            {"step": 1, "aha_any_gpt": 1},
            {"step": 1, "aha_any_gpt": 0},
            {"step": 2, "aha_any_gpt": 1},
        ]
    )
    ratios = bootstrap_problem_ratio(df, "aha_any_gpt", num_bootstrap=0, seed=123)
    first_step = ratios.loc[ratios["step"] == 1].iloc[0]
    assert first_step["k"] == 1 and first_step["n"] == 2
    assert np.isnan(first_step["lo"]) and np.isnan(first_step["hi"])
    second_step = ratios.loc[ratios["step"] == 2].iloc[0]
    assert second_step["ratio"] == pytest.approx(1.0)


def test_fit_trend_wls_returns_weighted_slope():
    ratio_df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "ratio": [1.0, 2.0, 3.0],
            "n": [1.0, 1.0, 1.0],
        }
    )
    slope, intercept, slope_per_1k, delta_range, r_squared, x_fit, y_fit = fit_trend_wls(ratio_df)
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0)
    assert slope_per_1k == pytest.approx(1000.0)
    assert delta_range == pytest.approx(2.0)
    assert r_squared == pytest.approx(1.0)
    assert len(x_fit) == len(y_fit) == 200


def test_build_positive_delta_flags_by_domain():
    df = pd.DataFrame(
        [
            {
                "domain": "Math",
                "step": 1,
                "p_correct_given_shift": 0.5,
                "freq_correct": 0.2,
                "aha_any_gpt": 1,
            },
            {
                "domain": "Math",
                "step": 2,
                "p_correct_given_shift": 0.25,
                "freq_correct": 0.2,
                "aha_any_gpt": 1,
            },
        ]
    )
    flags = build_positive_delta_flags(df)
    assert flags["Math"][1] is True
    assert flags["Math"][2] is False


def test_parse_float_list_parses_commas_and_spaces():
    assert parse_float_list("0.1, 0.2 ,0.3") == [0.1, 0.2, 0.3]
    assert parse_float_list("   ") == []


def _export_config(tmp_path) -> FormalExportConfig:
    return FormalExportConfig(
        dataset="DS",
        model="Model",
        thresholds=FormalThresholds(delta1=0.1, delta2=0.2, delta3=None, min_prior_steps=1),
        gpt_filter=GptFilterConfig(keys=["change_way_of_thinking"], subset_native=False),
        destinations=ExportDestinations(out_dir=str(tmp_path), slug="demo"),
        max_chars=10,
    )


def test_export_formal_aha_json_with_text_writes_events(tmp_path):
    out_dir = tmp_path / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    problem_step_df = pd.DataFrame(
        [
            {
                "domain": "Math",
                "problem": "p1",
                "step": 5,
                "aha_formal": 1,
                "n_samples": 4,
                "freq_correct": 0.2,
                "aha_rate_gpt": 0.5,
                "p_correct_given_shift": 0.6,
            },
            {
                # No matching record in files → placeholder event.
                "domain": "Math",
                "problem": "missing",
                "step": 7,
                "aha_formal": 1,
                "n_samples": 1,
                "freq_correct": 0.1,
                "aha_rate_gpt": 0.0,
                "p_correct_given_shift": np.nan,
            },
        ]
    )

    record_payload = {
        "problem": "p1",
        "pass1": {
            "change_way_of_thinking": 1,
            "has_reconsider_cue": 1,
            "final_answer": "<answer>The solution is longwinded text</answer>",
        },
    }
    file_path = out_dir / "step0005.jsonl"
    _write_jsonl(file_path, [record_payload])

    json_path, jsonl_path, count = export_formal_aha_json_with_text(
        problem_step_df,
        {"Math": [str(file_path)]},
        _export_config(out_dir),
    )
    assert count == 2
    events = json.loads(Path(json_path).read_text())
    assert Path(jsonl_path).exists()
    assert {event["problem"] for event in events} == {"p1", "missing"}
    p1_event = next(event for event in events if event["problem"] == "p1")
    assert p1_event["answer"].endswith("…[truncated]")
    assert p1_event["delta_gain_at_shift"] == pytest.approx(0.4)
    assert p1_event["thresholds"]["delta1"] == pytest.approx(0.1)
    placeholder = next(event for event in events if event["problem"] == "missing")
    assert placeholder["answer"] is None
