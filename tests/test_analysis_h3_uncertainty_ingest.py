#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.analysis.h3_uncertainty.ingest as ingest


def test_aha_words_and_gpt_from_mode(monkeypatch):
    rec = {"has_reconsider_cue": "1", "reconsider_markers": ["injected_cue"]}
    assert ingest._aha_words(rec) == 0  # injected cue forces 0

    # GPT mode uses supplied keys
    monkeypatch.setattr(ingest, "gpt_keys_for_mode", lambda mode: ["k1", "k2"])
    assert ingest._aha_gpt_from_mode({"k1": True}, {}, "canonical") == 1
    assert ingest._aha_gpt_from_mode({}, {"k2": "1"}, "canonical") == 1
    assert ingest._aha_gpt_from_mode({}, {}, "canonical") == 0


def test_aha_gpt_broad_and_string_any_key(monkeypatch):
    monkeypatch.setattr(ingest, "gpt_keys_for_mode", lambda mode: ["broad"])
    assert ingest._aha_gpt_broad({"broad": 1}, {}) == 1
    # _any_key_contains converts string numeric values
    record = {"foo_entropy_bar": "3.5"}
    val = ingest._any_key_contains(record, ["entropy"], ["foo"])
    assert val == pytest.approx(3.5)


def test_number_and_perplexity_extractors(monkeypatch):
    record = {"a": "1.5", "perplexity": 2.0, "entropy": 0.3, "overall_entropy_nats": 0.1}
    assert ingest._first_num(record, ["missing", "a"]) == 1.5
    assert ingest._any_key_contains(record, ["perplexity"], [""]) == 2.0

    # entropy measure
    val_ent = ingest.extract_uncertainty_or_ppx(record, unc_field="overall", measure="entropy")
    assert val_ent == pytest.approx(0.1)
    # perplexity measure uses explicit perplexity when present
    val_ppx = ingest.extract_uncertainty_or_ppx(record, unc_field="overall", measure="perplexity")
    assert val_ppx == pytest.approx(2.0)

    # None record short-circuits
    assert ingest.extract_uncertainty_or_ppx(None, unc_field="overall", measure="entropy") is None

    # think branch uses think_entropy_nats
    think_record = {"think_entropy_nats": 0.7}
    val_think = ingest.extract_uncertainty_or_ppx(think_record, unc_field="think", measure="entropy")
    assert val_think == pytest.approx(0.7)


def test_entropy_and_perplexity_value_fallbacks():
    value_map = {"entropy_something": 0.5}
    assert ingest._entropy_value(value_map, ppx=None, ent_n=None, ent=None) == 0.5
    assert ingest._perplexity_value(value_map, ppx=None, ent_n=None, ent=None) == pytest.approx(np.exp(0.5))
    assert ingest._perplexity_value({}, ppx=None, ent_n=1.0, ent=None) == pytest.approx(np.exp(1.0))
    assert ingest._perplexity_value({}, ppx=None, ent_n=None, ent=0.2) == pytest.approx(np.exp(0.2))
    assert ingest._entropy_value({}, ppx=4.0, ent_n=None, ent=None) == pytest.approx(np.log(4.0))


def test_prompt_key_and_forced_detection():
    rec = {"prompt_variant": "forced_insight"}
    assert ingest._detect_forced_insight({"reconsider_markers": []}, rec) == 1
    assert ingest._detect_forced_insight({"reconsider_markers": ["injected_cue"]}, {}) == 1

    pass_dict = {"prompt_id": "pid123"}
    assert ingest._extract_prompt_key(pass_dict, {}) == "pid123"
    assert ingest._extract_prompt_key({}, {"prompt": "text"}) == "textlen4"
    assert ingest._extract_prompt_key({}, {}) == "unknown"


def test_build_pass_row_and_iter(monkeypatch):
    payload = {"is_correct_pred": True, "has_reconsider_cue": True, "entropy_answer": 0.2}
    aha_ctx = ingest.AhaContext(gpt_fn=lambda p, r: 1, gate_by_words=True, include_forced_flag=True)
    unc_ctx = ingest.UncertaintyContext(field="answer", measure="entropy")
    ctx = ingest.PassRowContext(pair_id=1, pass_id=2, problem="prob", step=3, aha=aha_ctx, uncertainty=unc_ctx)
    row = ingest._build_pass_row(payload, {"prompt_variant": "forced_insight"}, ctx)
    assert row["aha_gpt"] == 1
    assert row["aha_words"] == 1
    assert row["unc"] == pytest.approx(0.2)
    assert row["forced_insight"] == 1

    # correctness missing -> None
    assert ingest._build_pass_row({}, {}, ctx) is None

    # iterator yields both pass1 and pass2 payloads when present
    builder_ctx = ingest.RowBuilderContext(
        pass1_aha=aha_ctx,
        pass2_aha=aha_ctx,
        uncertainty=unc_ctx,
    )
    payloads = list(ingest._iter_pass_payloads({"pass2": {}}, {"is_correct_pred": 1}, builder_ctx))
    assert len(payloads) == 2 and payloads[0][0] == 1 and payloads[1][0] == 2


def test_rows_for_record_and_load_rows(monkeypatch):
    # Monkeypatch extract_pass1_and_step and resolve_problem_identifier
    monkeypatch.setattr(ingest, "extract_pass1_and_step", lambda rec, step: (rec.get("pass1"), rec.get("step", step)))
    monkeypatch.setattr(ingest, "resolve_problem_identifier", lambda rec, fallback: rec.get("problem", fallback))
    monkeypatch.setattr(ingest, "gpt_keys_for_mode", lambda mode: ["k"])

    # Build a fake iter_pass1_records
    def fake_iter_pass1_records(files):
        yield (
            "path",
            0,
            {
                "problem": "p",
                "step": 1,
                "pass1": {"is_correct_pred": 1, "entropy": 0.5},
                "pass2": {"is_correct_pred": 0, "entropy": 0.6},
            },
        )

    monkeypatch.setattr(ingest, "iter_pass1_records", fake_iter_pass1_records)

    rows = ingest._rows_for_record(
        {"problem": "p", "step": 1, "pass1": {"is_correct_pred": 1, "entropy": 0.5}, "pass2": {"is_correct_pred": 0}},
        step_from_name=1,
        pair_id=0,
        builder_ctx=ingest.RowBuilderContext(
            pass1_aha=ingest.AhaContext(lambda p, r: 0, gate_by_words=False, include_forced_flag=False),
            pass2_aha=ingest.AhaContext(lambda p, r: 0, gate_by_words=False, include_forced_flag=True),
            uncertainty=ingest.UncertaintyContext(field="overall", measure="entropy"),
        ),
    )
    assert rows and rows[0]["problem"] == "p"

    df = ingest.load_rows(
        ["f1"], gpt_mode="canonical", gate_gpt_by_words=False, unc_field="overall", measure="entropy"
    )
    assert not df.empty
    assert set(df["pass_id"]) == {1, 2}

    # missing pass1/step yields empty list
    empty_rows = ingest._rows_for_record(
        {"problem": "p", "pass1": None},
        step_from_name=None,
        pair_id=0,
        builder_ctx=ingest.RowBuilderContext(
            pass1_aha=ingest.AhaContext(lambda p, r: 0, gate_by_words=False, include_forced_flag=False),
            pass2_aha=ingest.AhaContext(lambda p, r: 0, gate_by_words=False, include_forced_flag=False),
            uncertainty=ingest.UncertaintyContext(field="overall", measure="entropy"),
        ),
    )
    assert empty_rows == []

    # empty rows triggers SystemExit
    def fake_iter_pass1_records_empty(files):
        if False:
            yield

    monkeypatch.setattr(ingest, "iter_pass1_records", fake_iter_pass1_records_empty)
    with pytest.raises(SystemExit):
        ingest.load_rows(["f1"])


def test_perplexity_bucket_helpers():
    df = pd.DataFrame({"unc": [0.1, 0.2, 0.3]})
    bucketed = ingest.add_perplexity_buckets(df, n_buckets=2, method="quantile")
    assert "perplexity_bucket" in bucketed.columns
    assert bucketed["perplexity_bucket"].dtype == object

    bucketed_fixed = ingest.add_perplexity_buckets(df, n_buckets=2, method="fixed", custom_edges=[0.0, 0.2, 0.4])
    assert bucketed_fixed["perplexity_bucket"].dtype == object

    with pytest.raises(SystemExit):
        ingest._prepare_bucket_frame(pd.DataFrame())
    with pytest.raises(ValueError):
        ingest._assign_fixed_perplexity_buckets(df, custom_edges=None)
