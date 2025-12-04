import numpy as np
import pandas as pd

import src.analysis.h3_uncertainty.reasking as r


def test_wilson_ci_basic_and_empty():
    lo, hi = r.wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    lo_empty, hi_empty = r.wilson_ci(0, 0)
    assert np.isnan(lo_empty) and np.isnan(hi_empty)
    np_erfcinv = getattr(r.np, "erfcinv", None)
    try:
        r.np.erfcinv = lambda a: 0.5  # simple stub
        lo_custom, hi_custom = r.wilson_ci(1, 2, alpha=0.1)
        assert 0.0 <= lo_custom <= hi_custom <= 1.0
    finally:
        if np_erfcinv is not None:
            r.np.erfcinv = np_erfcinv
    assert 0.0 <= lo_custom <= hi_custom <= 1.0


def _build_sample_frames():
    # Two pairs across two problems.
    pass1 = pd.DataFrame(
        [
            {
                "pass_id": 1,
                "pair_id": 1,
                "problem": "p1",
                "step": 1,
                "prompt_key": "a1",
                "correct": 0,
                "perplexity_bucket": "low",
            },
            {
                "pass_id": 1,
                "pair_id": 2,
                "problem": "p2",
                "step": 1,
                "prompt_key": "b1",
                "correct": 1,
                "perplexity_bucket": "high",
            },
        ]
    )
    pass2 = pd.DataFrame(
        [
            {
                "pass_id": 2,
                "pair_id": 1,
                "problem": "p1",
                "step": 1,
                "prompt_key": "a2",
                "correct": 1,
                "forced_insight": 1,
            },
            {
                "pass_id": 2,
                "pair_id": 2,
                "problem": "p2",
                "step": 1,
                "prompt_key": "b2",
                "correct": 1,
                "forced_insight": 0,
            },
        ]
    )
    df_all = pd.concat([pass1, pass2], ignore_index=True)
    return df_all


def test_compute_reasking_tables_and_forced_split():
    pairs_df, probs_df, cond_df, cond_forced = r.compute_reasking_tables(_build_sample_frames())
    assert len(pairs_df) == 2
    assert set(pairs_df["delta"]) == {0, 1}
    # Problem-level summaries capture any/mean deltas.
    p1_row = probs_df[probs_df["problem"] == "p1"].iloc[0]
    assert p1_row["delta_any"] == 1 and p1_row["delta_mean"] == 1
    # Conditional summaries split by pass1 correctness
    assert set(cond_df["condition"]) == {"pass1_any_correct==0", "pass1_any_correct==1"}
    # Forced insight table should have two entries (forced=1 with pass1 wrong, forced=0 with pass1 right)
    assert set(cond_forced["forced_insight"]) == {0, 1}


def test_compute_reasking_tables_no_pass2():
    # All rows are pass1 → expect empties.
    df = _build_sample_frames()
    df.loc[df["pass_id"] == 2, "pass_id"] = 1
    pairs_df, probs_df, cond_df, cond_forced = r.compute_reasking_tables(df)
    for frame in (pairs_df, probs_df, cond_df, cond_forced):
        assert frame.empty


def test_split_reasking_by_aha_flags():
    pairs_df, probs_df, _, _ = r.compute_reasking_tables(_build_sample_frames())
    pass1_flags = pd.DataFrame(
        [
            {"pair_id": 1, "aha_words": 1, "aha_gpt": 0, "aha_formal": 0},
            {"pair_id": 2, "aha_words": 0, "aha_gpt": 1, "aha_formal": 1},
        ]
    )
    prompt_by_aha, problem_by_aha = r.split_reasking_by_aha(pairs_df, probs_df, pass1_flags)
    assert set(prompt_by_aha["aha_variant"]) == {"words", "gpt", "formal"}
    assert set(problem_by_aha["aha_variant"]) == {"words", "gpt", "formal"}


def test_split_reasking_by_aha_empty_and_partial():
    pairs_df, probs_df, _, _ = r.compute_reasking_tables(_build_sample_frames())
    empty_flags = pd.DataFrame(columns=["pair_id", "aha_words", "aha_gpt", "aha_formal"])
    p_empty, prob_empty = r.split_reasking_by_aha(pairs_df, probs_df, empty_flags)
    assert p_empty.empty and prob_empty.empty

    flags = pd.DataFrame([{"pair_id": 1, "aha_words": 0, "aha_gpt": 1, "aha_formal": 0}])
    probs_missing = probs_df[probs_df["problem"] != "p1"].copy()
    prompt_by_aha, problem_by_aha = r.split_reasking_by_aha(pairs_df, probs_missing, flags)
    assert set(prompt_by_aha["aha_variant"]) == {"gpt"}
    assert problem_by_aha.empty


def test_question_and_prompt_level_accuracy_with_ci():
    # Build a small pairs table with grouping buckets.
    pairs_df, _, _, _ = r.compute_reasking_tables(_build_sample_frames())
    pairs_df["bucket"] = ["lo", "hi"]
    q_df = r.question_level_any_with_ci(pairs_df, group_keys=["bucket"])
    p_df = r.prompt_level_acc_with_ci(pairs_df, group_keys=["bucket"])
    assert set(q_df["bucket"]) == {"hi", "lo"}
    assert set(p_df["bucket"]) == {"hi", "lo"}
    # Ensure Wilson CI columns are present
    for col in ("any_pass1_lo", "any_pass2_hi"):
        assert col in q_df.columns
    for col in ("correct1_lo", "correct2_hi"):
        assert col in p_df.columns
    # Delta should reflect pass2 - pass1
    assert set(p_df["delta"]) == {0.0, 1.0}


def test_pairwise_accuracy_delta_with_all_group():
    data = pd.DataFrame(
        {
            "problem": ["p1", "p2"],
            "any_pass1": [1, 0],
            "any_pass2": [1, 1],
        }
    )
    agg, keys = r._ensure_groupcol(data, [])
    assert keys == ["__all__"]
    row = r._pairwise_accuracy_delta(agg, keys, "all", r.QUESTION_GROUP_SPEC)
    assert row["n_problems"] == 2
    assert row["delta"] == 0.5


def test_forced_condition_table_missing_and_empty():
    df = pd.DataFrame({"pair_id": [1], "problem": ["p"], "correct1": [1], "correct2": [1]})
    assert r._build_forced_condition_table(df, lambda *a, **k: {}).empty

    pass1 = pd.DataFrame([{"pass_id": 1, "pair_id": 1, "problem": "p1", "step": 1, "prompt_key": "a", "correct": 1}])
    pass2 = pd.DataFrame([{"pass_id": 2, "pair_id": 2, "problem": "p2", "step": 1, "prompt_key": "b", "correct": 0}])
    df_all = pd.concat([pass1, pass2], ignore_index=True)
    pairs_df, probs_df, cond_df, cond_forced = r.compute_reasking_tables(df_all)
    assert pairs_df.empty and probs_df.empty and cond_df.empty and cond_forced.empty
