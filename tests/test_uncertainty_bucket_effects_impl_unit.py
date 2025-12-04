import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _import_with_stubs(monkeypatch):
    # Stub statsmodels modules to avoid dependency.
    sm_module = types.ModuleType("statsmodels.api")
    sm_module.families = types.SimpleNamespace(Binomial=lambda: None)
    smf_module = types.ModuleType("statsmodels.formula.api")
    gen_links = types.ModuleType("statsmodels.genmod.families.links")
    gen_links.Logit = lambda: None
    gen_fam = types.ModuleType("statsmodels.genmod.families")
    gen_fam.links = gen_links
    sm_exc = types.ModuleType("statsmodels.tools.sm_exceptions")
    sm_exc.PerfectSeparationError = ValueError

    base = types.ModuleType("statsmodels")
    base.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "statsmodels", base)
    monkeypatch.setitem(sys.modules, "statsmodels.api", sm_module)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", types.ModuleType("statsmodels.formula"))
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", smf_module)
    monkeypatch.setitem(sys.modules, "statsmodels.genmod", types.ModuleType("statsmodels.genmod"))
    monkeypatch.setitem(sys.modules, "statsmodels.genmod.families", gen_fam)
    monkeypatch.setitem(sys.modules, "statsmodels.genmod.families.links", gen_links)
    monkeypatch.setitem(sys.modules, "statsmodels.tools", types.ModuleType("statsmodels.tools"))
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", sm_exc)
    strat_tables = types.ModuleType("statsmodels.stats.contingency_tables")
    strat_tables.StratifiedTable = lambda *a, **k: None
    stats_pkg = types.ModuleType("statsmodels.stats")
    stats_pkg.__path__ = []  # mark as package
    proportion_mod = types.ModuleType("statsmodels.stats.proportion")
    proportion_mod.proportions_ztest = lambda *a, **k: (0.0, 0.5)
    monkeypatch.setitem(sys.modules, "statsmodels.stats", stats_pkg)
    monkeypatch.setitem(sys.modules, "statsmodels.stats.contingency_tables", strat_tables)
    monkeypatch.setitem(sys.modules, "statsmodels.stats.proportion", proportion_mod)
    return importlib.import_module("src.analysis.uncertainty_bucket_effects_impl")


def test_extract_step_and_rows(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    assert ub._extract_step_from_path("/a/step123/run.jsonl") == 123
    cfg = ub.RowLoadConfig(
        entropy_source="entropy", allow_fallback=True, min_step=5, max_step=10, temp=None, model_label=None
    )
    monkeypatch.setattr(
        ub,
        "iter_records_from_file",
        lambda path: [{"pass1": {"entropy": 0.5, "is_correct_pred": True}, "problem": "p1"}],
    )
    monkeypatch.setattr(ub, "step_within_bounds", lambda step, min_step, max_step: min_step <= step <= max_step)
    rows = ub._rows_for_file("/root/step008/a.jsonl", "Math", cfg)
    assert rows and rows[0]["problem_id"] == "Math::p1"


def test_correct_flags_and_entropy(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    cfg = ub.RowLoadConfig(
        entropy_source="entropy",
        allow_fallback=True,
        carpark_threshold=0.5,
        carpark_op="gt",
        temp=None,
        model_label=None,
    )
    assert ub._compute_correct_flag_for_domain("Carpark", {"soft_reward": 0.4}, cfg) == 0
    cfg2 = ub.RowLoadConfig(
        entropy_source="entropy",
        allow_fallback=True,
        carpark_threshold=0.5,
        carpark_op="ge",
        temp=None,
        model_label=None,
    )
    assert ub._compute_correct_flag_for_domain("Math", {"is_correct_pred": True}, cfg2) == 1
    p1 = {"entropy_answer": 0.2, "entropy_think": 0.3}
    val = ub._entropy_from_pass1(p1, entropy_source="answer_plus_think", allow_fallback=False, combined_mode="sum")
    assert val == 0.5
    val_fb = ub._entropy_from_pass1(
        {"entropy": 0.7}, entropy_source="unknown", allow_fallback=True, combined_mode="sum"
    )
    assert val_fb == 0.7


def test_basic_regression_and_glm_paths(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    df = pd.DataFrame({"shift": [0, 1, 1], "correct": [1, 1, 0]})
    stats = ub._basic_regression_stats(df)
    assert stats["N"] == 3
    # Force glm_ame_bucket to take Newcombe path
    monkeypatch.setattr(ub, "_shift_group_stats", lambda sub: ({0: 2, 1: 2}, {0: 1, 1: 2}, 1))
    ame, lo, hi, p = ub.glm_ame_bucket(df)
    assert np.isfinite(ame)


def test_print_discovered_roots(monkeypatch, capsys):
    ub = _import_with_stubs(monkeypatch)
    mapping = {"m": {0.1: {"Math": "/root"}}}
    ub._print_discovered_roots(mapping)
    out = capsys.readouterr().out
    assert "discovered roots" in out and "Math" in out


def test_discover_roots_skips_unknown_domain(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    monkeypatch.setattr(ub.os, "listdir", lambda root: ["dir1"])
    monkeypatch.setattr(ub.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(ub, "detect_model_key", lambda name: "m")
    monkeypatch.setattr(ub, "detect_domain", lambda name: None)  # triggers skip at line 157
    monkeypatch.setattr(ub, "detect_temperature", lambda name, low_alias: 0.1)
    cfg = ub.RootDiscoveryConfig(temps=[0.1], low_alias=None, want_models=["m"], want_domains=["math"], verbose=False)
    mapping = ub.discover_roots_7b8b("root", cfg)
    assert mapping == {}


def test_scan_step_jsonls_filters_and_warns(tmp_path, capsys):
    root = tmp_path / "step001"
    root.mkdir()
    (root / "file.jsonl").write_text("x")
    (root / "ignore.txt").write_text("y")
    from src.analysis import uncertainty_bucket_effects_impl as ub  # real module import ok here

    files = ub.scan_step_jsonls(str(tmp_path), split="val", verbose=True)
    out = capsys.readouterr().out
    assert str(root / "file.jsonl") in files
    assert "no files matched split" in out
    assert not any("ignore.txt" in f for f in files)


def test_glm_ame_bucket_falls_back_on_fit_error(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    monkeypatch.setattr(ub, "_shift_group_stats", lambda sub: ({0: 2, 1: 2}, {0: 1, 1: 1}, 2))
    monkeypatch.setattr(ub, "_fit_shift_glm", lambda sub: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(ub, "_newcombe_ame_stats", lambda counts, corrects: (1.0, 0.0, 2.0, 0.5))
    df = pd.DataFrame({"shift": [0, 1], "correct": [1, 0]})
    ame = ub.glm_ame_bucket(df)
    assert ame == (1.0, 0.0, 2.0, 0.5)


def test_glm_ame_bucket_calls_fit_and_ame(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    called = {}

    class Res:
        def __init__(self):
            self.model = types.SimpleNamespace(exog=np.array([[1, 0], [1, 1]]), exog_names=["intercept", "shift"])
            self.params = types.SimpleNamespace(to_numpy=lambda: np.array([0.0, 1.0]))

        def cov_params(self):
            return types.SimpleNamespace(to_numpy=lambda: np.eye(2))

        @property
        def pvalues(self):
            return {"shift": 0.1}

    def fake_fit(sub):
        called["fit"] = True
        return Res()

    monkeypatch.setattr(ub, "_shift_group_stats", lambda sub: ({0: 2, 1: 2}, {0: 1, 1: 1}, 2))
    monkeypatch.setattr(ub, "_fit_shift_glm", fake_fit)

    def fake_compute(design_matrix, shift_index, params, cov, bootstrap):
        called["design"] = (design_matrix.shape, shift_index, params.tolist(), cov.tolist())
        return 0.1, 0.0, 0.2

    monkeypatch.setattr(ub, "_compute_ame_from_design", fake_compute)

    df = pd.DataFrame({"shift": [0, 1, 1], "correct": [0, 1, 1]})
    ame, lo, hi, p = ub.glm_ame_bucket(df)
    assert called["fit"] and "design" in called
    assert np.isfinite(ame) and np.isfinite(lo) and np.isfinite(hi)
    assert p == 0.1


def test_main_capped_max_and_runs(monkeypatch, capsys, tmp_path):
    ub = _import_with_stubs(monkeypatch)
    args_ns = types.SimpleNamespace(
        scan_root="root",
        temps=[0.1],
        low_alias=None,
        models=["m"],
        domains=["math"],
        metrics=None,
        metric=None,
        combined_mode="sum",
        min_step=0,
        max_step=None,
        split=None,
        seed=0,
        entropy_mode="entropy",
        batch_size=1,
        num_samples=1,
        think_cap=1,
        answer_cap=1,
        temperature=0.0,
        top_p=0.9,
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        carpark_success_op="ge",
        carpark_soft_threshold=0.1,
        gpt_mode="canonical",
        n_boot=10,
        out_dir=str(tmp_path),
        verbose=True,
        allow_metric_fallback=False,
        binning="uniform",
        bin_scope="global",
        bins=2,
        tie_break="none",
        samples_per_bin=1,
        no_anova=False,
        train_split=None,
        metric_aggregate=None,
    )
    monkeypatch.setattr(ub.argparse.ArgumentParser, "parse_args", lambda self=None, argv=None: args_ns)
    monkeypatch.setattr(ub, "discover_roots_7b8b", lambda scan_root, config: {"m": {0.1: {"math": "p"}}})
    seen = {}
    monkeypatch.setattr(ub, "run_for_model", lambda cfg: seen.setdefault("cfg", cfg))

    ub.main()
    out = capsys.readouterr().out
    assert "Capping max_step" in out
    assert seen["cfg"].max_step == 1000


def test_main_exits_when_no_roots(monkeypatch):
    ub = _import_with_stubs(monkeypatch)
    args_ns = types.SimpleNamespace(
        scan_root="root",
        temps=[0.1],
        low_alias=None,
        models=["m"],
        domains=["math"],
        metrics=None,
        metric=None,
        combined_mode="sum",
        min_step=0,
        max_step=5,
        split=None,
        seed=0,
        entropy_mode="entropy",
        batch_size=1,
        num_samples=1,
        think_cap=1,
        answer_cap=1,
        temperature=0.0,
        top_p=0.9,
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        carpark_success_op="ge",
        carpark_soft_threshold=0.1,
        gpt_mode="canonical",
        n_boot=10,
        out_dir=None,
        verbose=False,
        allow_metric_fallback=False,
        binning="uniform",
        bin_scope="global",
        bins=2,
        tie_break="none",
        samples_per_bin=1,
        no_anova=False,
        train_split=None,
        metric_aggregate=None,
    )
    monkeypatch.setattr(ub.argparse.ArgumentParser, "parse_args", lambda self=None, argv=None: args_ns)
    monkeypatch.setattr(ub, "discover_roots_7b8b", lambda scan_root, config: {})
    with pytest.raises(SystemExit):
        ub.main()


def test_main_warns_when_model_missing(monkeypatch, capsys):
    ub = _import_with_stubs(monkeypatch)
    args_ns = types.SimpleNamespace(
        scan_root="root",
        temps=[0.1],
        low_alias=None,
        models=["m1"],
        domains=["math"],
        metrics=None,
        metric=None,
        combined_mode="sum",
        min_step=0,
        max_step=5,
        split=None,
        seed=0,
        entropy_mode="entropy",
        batch_size=1,
        num_samples=1,
        think_cap=1,
        answer_cap=1,
        temperature=0.0,
        top_p=0.9,
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        carpark_success_op="ge",
        carpark_soft_threshold=0.1,
        gpt_mode="canonical",
        n_boot=10,
        out_dir=None,
        verbose=False,
        allow_metric_fallback=False,
        binning="uniform",
        bin_scope="global",
        bins=2,
        tie_break="none",
        samples_per_bin=1,
        no_anova=False,
        train_split=None,
        metric_aggregate=None,
    )
    monkeypatch.setattr(ub.argparse.ArgumentParser, "parse_args", lambda self=None, argv=None: args_ns)
    # mapping missing model m1
    monkeypatch.setattr(ub, "discover_roots_7b8b", lambda scan_root, config: {"other": {"temps": {}}})
    ub.main()
    out = capsys.readouterr().out
    assert "No runs found for" in out
