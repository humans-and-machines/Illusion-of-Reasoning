import os

import pandas as pd
import pytest

import src.analysis.core as analysis_core


def test_compute_correct_and_shift_carpark_missing_reward_returns_none():
    # Carpark domain: soft_reward missing -> carpark_success_fn returns None -> None outcome.
    result = analysis_core.compute_correct_and_shift(
        "Carpark",
        pass1_data={"soft_reward": None},
        record={},
        gpt_subset_native=False,
        gpt_keys=[],
        carpark_success_fn=lambda reward: None,
    )
    assert result is None


def test_mark_formal_pairs_with_gain_requires_columns():
    with pytest.raises(ValueError):
        analysis_core.mark_formal_pairs_with_gain(
            pd.DataFrame([{"problem": "p1", "step": 1}]),
            thresholds=analysis_core.FormalThresholds(delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=None),
        )


def test_rank_math_path_scores_qwen_and_low_temp():
    base, length = analysis_core._rank_math_path("run-qwen-low-temp")
    assert base > 0  # qwen and low_temp both contribute
    longer_score, longer_len = analysis_core._rank_math_path("run-llama-low_temp-1.5b")
    # 1.5b and llama should give higher score than qwen+low_temp-only path despite similar length.
    assert longer_score > base
    assert longer_len >= length


def test_discover_roots_by_temp_filters_temp_and_domain(tmp_path):
    # Build directories that should all be filtered out for different reasons.
    (tmp_path / "no_temp_dir").mkdir()
    (tmp_path / "temp-0.5-math").mkdir()
    (tmp_path / "temp-0.1-unknown").mkdir()

    mapping = analysis_core.discover_roots_by_temp(
        scan_root=str(tmp_path),
        temps=[0.1],
        low_alias=0.1,
        skip_substrings=set(),
    )
    # All dirs filtered: no temp token, temp not requested, and domain not recognized.
    assert mapping == {}


def test_discover_roots_by_temp_skips_unlisted_temp(monkeypatch):
    # Force parse_temp_from_dir to produce a temp not in the requested list.
    monkeypatch.setattr(analysis_core, "parse_temp_from_dir", lambda dirname, low_alias: 0.2)
    monkeypatch.setattr(analysis_core, "_classify_domain_from_dir", lambda dirname: "Math")
    log_called = {"flag": False}
    monkeypatch.setattr(analysis_core, "_log_discovered_roots", lambda mapping: log_called.__setitem__("flag", True))

    def fake_walk(root):
        yield ("root_dir", ["temp-0.2-math"], [])

    monkeypatch.setattr(os, "walk", fake_walk)

    mapping = analysis_core.discover_roots_by_temp(
        scan_root="root_dir",
        temps=[0.1],  # request only 0.1, should skip 0.2
        low_alias=0.1,
        skip_substrings=set(),
    )

    assert mapping == {}  # temp not in temps_set triggers the skip branch
    assert log_called["flag"] is False  # no logging when mapping is empty
