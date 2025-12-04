import argparse
import runpy
import sys
from types import ModuleType

import pandas as pd
import pytest

import src.analysis.h3_uncertainty_buckets as h3


class ArgsStub(argparse.Namespace):
    """Simple args holder with sensible defaults."""

    def __init__(self, **kwargs):
        defaults = dict(
            delta1=0.1,
            delta2=0.1,
            min_prior_steps=1,
            n_buckets=3,
            bucket_method="quantile",
            bucket_edges=None,
            gpt_mode="canonical",
            no_gate_gpt_by_words=False,
            unc_field="answer",
            measure="entropy",
            strict_interaction_only=False,
            cluster_by="problem",
            dataset_name="DS",
            model_name="Model",
            split=None,
            results_root="root",
            out_dir=None,
            split_reasking_by_aha=False,
            make_plots=False,
            also_pdf_plots=False,
            make_pdf=False,
            font_family="Times",
            font_size=12,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


def test_parse_bucket_edges_success_and_error():
    assert h3.parse_bucket_edges("0.1, 0.5,1") == [0.1, 0.5, 1.0]
    assert h3.parse_bucket_edges(None) is None
    with pytest.raises(SystemExit):
        h3.parse_bucket_edges("bad,1")


def test_build_arg_parser_defaults():
    parser = h3.build_arg_parser()
    args = parser.parse_args(["root_dir"])
    assert args.results_root == "root_dir"
    assert args.unc_field == "answer"
    assert args.bucket_method == "quantile"
    assert args.font_family == "Times New Roman"


def test_prepare_pass1_dataframe_adds_formal_and_buckets(monkeypatch):
    samples_df = pd.DataFrame(
        {
            "pass_id": [1, 2],
            "pair_id": [1, 1],
            "problem": ["p1", "p1"],
            "step": [1, 1],
            "aha_gpt": [1, 0],
        }
    )

    monkeypatch.setattr(h3, "build_problem_step_from_samples", lambda df: df[["problem", "step"]])
    monkeypatch.setattr(h3, "make_formal_thresholds", lambda *a, **k: "thresholds")
    monkeypatch.setattr(
        h3,
        "mark_formal_pairs_with_gain",
        lambda df, thresholds: df.assign(aha_formal_pair=1),
    )

    def fake_add_perplexity_buckets(df, n_buckets, method, custom_edges=None):
        assert n_buckets == 3 and method == "quantile"
        return df.assign(perplexity_bucket=[0])

    monkeypatch.setattr(h3, "add_perplexity_buckets", fake_add_perplexity_buckets)

    out_df = h3.prepare_pass1_dataframe(samples_df, ArgsStub())
    assert "aha_formal" in out_df.columns
    assert out_df["aha_formal"].iloc[0] == 1
    assert "perplexity_bucket" in out_df.columns

    # Fixed buckets parses custom edges
    parsed_edges = {}
    monkeypatch.setattr(
        h3,
        "add_perplexity_buckets",
        lambda df, n_buckets, method, custom_edges=None: parsed_edges.setdefault("edges", custom_edges) or df,
    )
    args = ArgsStub(bucket_method="fixed", bucket_edges="0.1,0.2", n_buckets=2)
    h3.prepare_pass1_dataframe(samples_df, args)
    assert parsed_edges["edges"] == [0.1, 0.2]


def test_run_glm_variants_writes_outputs(monkeypatch, tmp_path):
    pass1_df = pd.DataFrame(
        {
            "perplexity_bucket": [0, 1],
            "aha_words": [0, 1],
            "aha_gpt": [0, 1],
            "aha_formal": [0, 1],
            "correct": [1, 0],
            "problem": ["p1", "p1"],
            "step": [1, 2],
        }
    )

    def fake_fit(data_frame, aha_col, strict_interaction_only, cluster_by, out_txt):
        return (
            {
                "N": len(data_frame),
                "acc_overall": 0.5,
                "bucket_rows": [
                    {"bucket": 0, "N": 1, "share_aha": 0.0, "AME_bucket": 0.1},
                    {"bucket": 1, "N": 1, "share_aha": 1.0, "AME_bucket": 0.2},
                ],
            },
            "result",
        )

    monkeypatch.setattr(h3, "fit_glm_bucket_interaction", fake_fit)
    monkeypatch.setattr(
        h3,
        "bucket_group_accuracy",
        lambda subset, aha_col: subset[["perplexity_bucket", aha_col]]
        .assign(n=1, k=subset[aha_col])
        .rename(columns={aha_col: "aha"})
        .assign(accuracy=1.0),
    )

    margins_df = h3.run_glm_variants(pass1_df, ArgsStub(out_dir=str(tmp_path)), out_dir=str(tmp_path))
    assert not margins_df.empty
    assert (tmp_path / "h3_glm_bucket_margins.csv").exists()
    assert (tmp_path / "h3_bucket_group_accuracy.csv").exists()


def test_run_glm_variants_skips_empty_subsets(monkeypatch, tmp_path):
    pass1_df = pd.DataFrame(
        {
            "perplexity_bucket": [0],
            "aha_words": [pd.NA],
            "aha_gpt": [1],
            "aha_formal": [pd.NA],
        }
    )
    monkeypatch.setattr(
        h3,
        "fit_glm_bucket_interaction",
        lambda *a, **k: (
            {
                "bucket_rows": [
                    {"bucket": 0, "N": 1, "share_aha": 0.0, "AME_bucket": 0.0},
                ],
            },
            None,
        ),
    )
    monkeypatch.setattr(
        h3,
        "bucket_group_accuracy",
        lambda subset, aha_col: subset[["perplexity_bucket", aha_col]].rename(columns={aha_col: "aha"}),
    )
    margins_df = h3.run_glm_variants(pass1_df, ArgsStub(out_dir=str(tmp_path)), out_dir=str(tmp_path))
    assert not margins_df.empty
    assert (tmp_path / "h3_glm_bucket_margins.csv").exists()
    # Words subset should be skipped, GPT present
    assert (tmp_path / "h3_bucket_group_accuracy__gpt.csv").exists()


def test_run_glm_variants_returns_empty_template(tmp_path):
    """Exercise the empty margin template branch."""
    pass1_df = pd.DataFrame(
        {
            "perplexity_bucket": [0],
            "aha_words": [pd.NA],
            "aha_gpt": [pd.NA],
            "aha_formal": [pd.NA],
        }
    )

    margins_df = h3.run_glm_variants(pass1_df, ArgsStub(out_dir=str(tmp_path)), out_dir=str(tmp_path))
    assert margins_df.empty
    assert list(margins_df.columns) == [
        "dataset",
        "model",
        "variant",
        "perplexity_bucket",
        "N",
        "share_aha",
        "AME_bucket",
        "glm_summary_path",
    ]
    assert (tmp_path / "h3_glm_bucket_margins.csv").exists()


def test_write_dataframe_if_any_and_empty_helpers(tmp_path):
    df = pd.DataFrame({"a": [1]})
    dest = tmp_path / "out.csv"
    h3._write_dataframe_if_any(df, str(dest))
    assert dest.exists()

    dest2 = tmp_path / "skip.csv"
    h3._write_dataframe_if_any(pd.DataFrame(), str(dest2))
    assert not dest2.exists()

    empty_prompt = h3._empty_prompt_tables()
    assert set(empty_prompt.keys()) == {"p_overall", "p_by_step", "p_by_bucket"}
    assert all(frame.empty for frame in empty_prompt.values())

    empty_q = h3._empty_question_tables()
    assert set(empty_q.keys()) == {"q_overall", "q_by_step", "q_by_bucket"}
    assert all(frame.empty for frame in empty_q.values())


def test_prepare_prompt_and_question_tables(monkeypatch, tmp_path):
    pairs_df = pd.DataFrame(
        {
            "step": [1, 2],
            "perplexity_bucket": [0, 1],
            "correct": [1, 0],
        }
    )
    monkeypatch.setattr(h3, "prompt_level_acc_with_ci", lambda df, by: df.assign(n=1, k=1, accuracy=1.0))
    monkeypatch.setattr(h3, "question_level_any_with_ci", lambda df, by: df.assign(n=1, k=1, accuracy=1.0))

    prompt_tables = h3._prepare_prompt_tables(pairs_df, str(tmp_path))
    assert (tmp_path / "h3_prompt_level_overall.csv").exists()
    assert not prompt_tables["p_by_bucket"].empty

    question_tables = h3._prepare_question_tables(pairs_df, str(tmp_path))
    assert (tmp_path / "h3_question_level_overall.csv").exists()
    assert not question_tables["q_by_bucket"].empty

    # No bucket column branches
    pairs_no_bucket = pd.DataFrame({"step": [1], "correct": [1]})
    prompt_tables2 = h3._prepare_prompt_tables(pairs_no_bucket, str(tmp_path))
    assert prompt_tables2["p_by_bucket"].empty
    question_tables2 = h3._prepare_question_tables(pairs_no_bucket, str(tmp_path))
    assert question_tables2["q_by_bucket"].empty

    # Empty input returns placeholder tables
    empty_prompt = h3._prepare_prompt_tables(pd.DataFrame(), str(tmp_path))
    assert all(df.empty for df in empty_prompt.values())
    empty_q = h3._prepare_question_tables(pd.DataFrame(), str(tmp_path))
    assert all(df.empty for df in empty_q.values())


def test_write_split_by_aha(monkeypatch, tmp_path):
    pairs_df = pd.DataFrame({"pair_id": [1], "perplexity_bucket": [0]})
    probs_df = pd.DataFrame({"problem": ["p1"]})
    pass1_df = pd.DataFrame({"pair_id": [1]})
    monkeypatch.setattr(
        h3,
        "split_reasking_by_aha",
        lambda *a, **k: (
            pd.DataFrame({"a": [1]}),
            pd.DataFrame({"b": [2]}),
        ),
    )
    h3._write_split_by_aha(pairs_df, probs_df, pass1_df, str(tmp_path))
    assert (tmp_path / "h3_pass2_prompt_level_by_aha.csv").exists()
    assert (tmp_path / "h3_pass2_question_level_by_aha.csv").exists()


def test_write_split_by_aha_no_data(tmp_path):
    h3._write_split_by_aha(
        pairs_df=pd.DataFrame(),
        probs_df=pd.DataFrame({"problem": []}),
        pass1_df=pd.DataFrame(),
        out_dir=str(tmp_path),
    )
    assert not list(tmp_path.iterdir())


def test_run_reasking_analysis_flow(monkeypatch, tmp_path):
    samples_df = pd.DataFrame(
        {
            "pair_id": [1, 1],
            "problem": ["p1", "p1"],
            "step": [1, 1],
            "pass_id": [1, 2],
        }
    )
    pass1_df = pd.DataFrame(
        {
            "pair_id": [1],
            "perplexity_bucket": [0],
            "aha_words": [0],
            "aha_gpt": [0],
            "aha_formal": [0],
        }
    )
    pairs_df = pd.DataFrame({"pair_id": [1], "perplexity_bucket": [0]})
    probs_df = pd.DataFrame({"problem": ["p1"]})
    cond_df = pd.DataFrame({"a": [1]})
    cond_forced_df = pd.DataFrame({"b": [2]})
    monkeypatch.setattr(
        h3,
        "compute_reasking_tables",
        lambda *a, **k: (pairs_df, probs_df, cond_df, cond_forced_df),
    )
    monkeypatch.setattr(
        h3,
        "prompt_level_acc_with_ci",
        lambda df, by: pd.DataFrame({"k": [1]}),
    )
    monkeypatch.setattr(
        h3,
        "question_level_any_with_ci",
        lambda df, by: pd.DataFrame({"k": [1]}),
    )
    monkeypatch.setattr(
        h3,
        "split_reasking_by_aha",
        lambda *a, **k: (pd.DataFrame({"k": [1]}), pd.DataFrame({"k": [2]})),
    )

    args = ArgsStub(split_reasking_by_aha=True)
    results = h3.run_reasking_analysis(samples_df, pass1_df, args, str(tmp_path))
    # Core outputs exist
    assert (tmp_path / "h3_pass2_prompt_level.csv").exists()
    assert (tmp_path / "h3_pass2_question_level.csv").exists()
    assert results["pairs_df"].equals(pairs_df)
    assert "p_overall" in results and "q_overall" in results


def test_maybe_plot_results_calls_plotters(monkeypatch, tmp_path):
    calls = []

    def _mark(name):
        return lambda *a, **k: calls.append(name)

    monkeypatch.setattr(h3, "plot_question_overall_ci", _mark("q_overall"))
    monkeypatch.setattr(h3, "plot_question_by_step_ci", _mark("q_step"))
    monkeypatch.setattr(h3, "plot_question_by_bucket_ci", _mark("q_bucket"))
    monkeypatch.setattr(h3, "plot_prompt_overall_ci", _mark("p_overall"))
    monkeypatch.setattr(h3, "plot_prompt_by_step_ci", _mark("p_step"))
    monkeypatch.setattr(h3, "plot_prompt_by_bucket_ci", _mark("p_bucket"))
    monkeypatch.setattr(h3, "plot_prompt_level_deltas", _mark("deltas"))

    args = ArgsStub(make_plots=True, also_pdf_plots=False)
    df_nonempty = pd.DataFrame({"x": [1]})
    results = {
        "pairs_df": df_nonempty,
        "q_overall": df_nonempty,
        "q_by_step": df_nonempty,
        "q_by_bucket": df_nonempty,
        "p_overall": df_nonempty,
        "p_by_step": df_nonempty,
        "p_by_bucket": df_nonempty,
    }
    h3.maybe_plot_results(args, str(tmp_path), results)
    assert set(calls) == {"q_overall", "q_step", "q_bucket", "p_overall", "p_step", "p_bucket", "deltas"}


def test_maybe_plot_results_no_plots(monkeypatch, tmp_path):
    def fail(*a, **k):
        pytest.fail("plot function should not be called when make_plots is False")

    monkeypatch.setattr(h3, "plot_question_overall_ci", fail)
    args = ArgsStub(make_plots=False)
    h3.maybe_plot_results(
        args,
        str(tmp_path),
        {
            "pairs_df": pd.DataFrame(),
            "q_overall": pd.DataFrame(),
            "q_by_step": pd.DataFrame(),
            "q_by_bucket": pd.DataFrame(),
            "p_overall": pd.DataFrame(),
            "p_by_step": pd.DataFrame(),
            "p_by_bucket": pd.DataFrame(),
        },
    )


def test_summarize_outputs_prints_existing(tmp_path, capsys):
    filenames = ["h3_glm_bucket_margins.csv", "h3_plot_prompt_overall.png"]
    for name in filenames:
        (tmp_path / name).touch()
    h3.summarize_outputs(str(tmp_path))
    captured = capsys.readouterr()
    assert "h3_glm_bucket_margins.csv" in captured.out
    assert "h3_plot_prompt_overall.png" in captured.out


def test_run_pipeline_exits_on_missing_files(monkeypatch):
    monkeypatch.setattr(h3, "scan_jsonl_files", lambda root, split: [])
    args = ArgsStub(results_root="root", out_dir=None)
    with pytest.raises(SystemExit):
        h3.run_pipeline(args)


def test_run_pipeline_requires_results_root():
    args = ArgsStub(results_root=None)
    with pytest.raises(SystemExit, match="results_root is required."):
        h3.run_pipeline(args)


def test_run_pipeline_happy_path(monkeypatch, tmp_path):
    # Stub everything to avoid heavy work but exercise wiring.
    monkeypatch.setattr(h3, "scan_jsonl_files", lambda root, split: ["file.jsonl"])
    monkeypatch.setattr(
        h3,
        "load_rows",
        lambda files, gpt_mode, gate_gpt_by_words, unc_field, measure: pd.DataFrame(
            {"pass_id": [1], "pair_id": [1], "problem": ["p1"], "step": [1], "aha_gpt": [0]}
        ),
    )
    monkeypatch.setattr(
        h3,
        "prepare_pass1_dataframe",
        lambda samples_df, args: samples_df.assign(perplexity_bucket=0, aha_words=0, aha_formal=0),
    )
    monkeypatch.setattr(
        h3,
        "run_glm_variants",
        lambda pass1_df, args, out_dir: pd.DataFrame({"dummy": [1]}),
    )
    monkeypatch.setattr(
        h3,
        "run_reasking_analysis",
        lambda samples_df, pass1_df, args, out_dir: {
            "cond_df": pd.DataFrame({"a": [1]}),
            "q_overall": pd.DataFrame({"x": [1]}),
            "q_by_step": pd.DataFrame({"x": [1]}),
            "q_by_bucket": pd.DataFrame({"x": [1]}),
            "p_overall": pd.DataFrame({"x": [1]}),
            "p_by_step": pd.DataFrame({"x": [1]}),
            "p_by_bucket": pd.DataFrame({"x": [1]}),
        },
    )
    monkeypatch.setattr(h3, "write_a4_summary_pdf", lambda *a, **k: None)
    monkeypatch.setattr(h3, "write_answers_txt", lambda *a, **k: None)
    monkeypatch.setattr(h3, "maybe_plot_results", lambda *a, **k: None)
    monkeypatch.setattr(h3, "summarize_outputs", lambda *a, **k: None)

    args = ArgsStub(results_root="root", out_dir=str(tmp_path), make_pdf=True)
    h3.run_pipeline(args)
    # Outputs should live under provided out_dir
    assert (tmp_path).exists()


def test_run_pipeline_requires_pass1(monkeypatch):
    monkeypatch.setattr(h3, "scan_jsonl_files", lambda root, split: ["f.jsonl"])
    monkeypatch.setattr(
        h3,
        "load_rows",
        lambda *a, **k: pd.DataFrame({"pass_id": [2], "pair_id": [1], "problem": ["p"], "step": [1], "aha_gpt": [0]}),
    )
    args = ArgsStub(results_root="root", out_dir=None)
    with pytest.raises(SystemExit):
        h3.run_pipeline(args)


def test_main_invokes_pipeline(monkeypatch):
    called = {}
    args = ArgsStub(results_root="root", out_dir="out")
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(h3, "build_arg_parser", lambda: parser)
    monkeypatch.setattr(h3, "run_pipeline", lambda got_args: called.setdefault("args", got_args))
    h3.main()
    assert called["args"] is args


def test_module_entrypoint_runs(monkeypatch, tmp_path):
    args = ArgsStub(results_root="root", out_dir=str(tmp_path))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)

    # Ensure scan_jsonl_files is available and returns empty so pipeline exits early.
    stub_io = ModuleType("src.analysis.io")
    stub_io.scan_jsonl_files = lambda root, split: []
    monkeypatch.setitem(sys.modules, "src.analysis.io", stub_io)

    with pytest.raises(SystemExit):
        sys.modules.pop("src.analysis.h3_uncertainty_buckets", None)
        runpy.run_module("src.analysis.h3_uncertainty_buckets", run_name="__main__")
