import builtins
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

import src.analysis.cue_delta_accuracy as cue_mod


def test_build_table_and_rows_skip_non_int_quartiles():
    rows = [
        {
            "domain": "Math",
            "entropy_quartile": 1,
            "cue_variant": "baseline",
            "intervention_correct": True,
            "baseline_correct": True,
        },
        {
            "domain": "Math",
            "entropy_quartile": 1,
            "cue_variant": "cueA",
            "intervention_correct": False,
            "baseline_correct": False,
        },
        {"domain": "Math", "entropy_quartile": "bad", "cue_variant": "cueA"},
    ]
    table, domains, quartile_map, variants = cue_mod._build_table(rows)
    assert domains == ("Math",)
    assert quartile_map["Math"] == (1,)
    assert "cueA" in variants and "baseline" in variants

    table_rows = cue_mod._table_rows_from_stats(table, domains, quartile_map, variants)
    assert table_rows[0]["domain"] == "Math"
    assert table_rows[0]["cue_variant"] == "cueA"


def test_build_table_counts_baseline_wrong_correct():
    rows = [
        {
            "domain": "Math",
            "entropy_quartile": 1,
            "cue_variant": "baseline",
            "intervention_correct": False,
            "baseline_correct": False,
        },
        {
            "domain": "Math",
            "entropy_quartile": 1,
            "cue_variant": "cueA",
            "intervention_correct": True,
            "baseline_correct": False,
        },
    ]
    table, _, _, _ = cue_mod._build_table(rows)
    stats = table[("Math", 1, "cueA")]
    assert stats["baseline_wrong_total"] == 1
    assert stats["baseline_wrong_correct"] == 1


def test_print_table_rows_and_write_csv(tmp_path, capsys):
    cue_mod._print_table_rows(tuple())
    assert "no table rows" in capsys.readouterr().out

    rows = (
        {
            "domain": "Math",
            "quartile": 1,
            "cue_variant": "cueA",
            "N": 2,
            "baseline_acc": 0.5,
            "cue_acc": 1.0,
            "cue_acc_baseline_wrong": 1.0,
        },
    )
    cue_mod._print_table_rows(rows)
    out = capsys.readouterr().out
    assert "domain=Math" in out and "cueA" in out

    csv_path = tmp_path / "table.csv"
    cue_mod._write_table_csv(rows, csv_path)
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "cue_variant" in content and "cueA" in content


def test_print_table_rows_handles_empty_branches(monkeypatch, capsys):
    rows = (
        {
            "domain": "Math",
            "quartile": 1,
            "cue_variant": "baseline",
            "N": 1,
            "baseline_acc": 1.0,
            "cue_acc": 1.0,
            "cue_acc_baseline_wrong": 1.0,
        },
    )
    orig_sorted = builtins.sorted

    def fake_sorted(iterable, *args, **kwargs):
        fake_sorted.calls += 1
        result = list(orig_sorted(iterable, *args, **kwargs))
        if fake_sorted.calls == 1:  # domains
            result.append("ExtraDomain")
            return result
        if fake_sorted.calls == 2:  # cues
            result.append("ExtraCue")
            return result
        return result

    fake_sorted.calls = 0
    monkeypatch.setattr(builtins, "sorted", fake_sorted)
    cue_mod._print_table_rows(rows)
    out = capsys.readouterr().out
    assert "domain=Math" in out


def test_format_helpers_and_stats_updates(capsys):
    assert cue_mod._format_pct_or_na(None) == "n/a"
    assert cue_mod._format_pct_or_na(0.5) == "50.0%"
    assert cue_mod._format_pct(None) == "n/a"
    assert cue_mod._format_pct(0.25) == "25.0%"
    assert cue_mod._format_diff(None) == "n/a"
    assert cue_mod._format_diff(0.1) == "+10.0pp"
    assert cue_mod._pct(1, 0) is None

    stats = cue_mod._build_stats()
    cue_mod._update_stats(
        stats,
        {"intervention_correct": True, "entropy_quartile": 2, "output_len": 10, "tokens_total": 5},
    )
    assert stats["correct"] == 1 and stats["quartiles"][2]["correct"] == 1
    cue_mod._print_average_lengths(stats)
    out = capsys.readouterr().out
    assert "avg output_len" in out

    baseline_stats = cue_mod._build_stats()
    cue_mod._update_stats(baseline_stats, {"intervention_correct": True, "entropy_quartile": 2})
    cue_stats = cue_mod._build_stats()
    cue_mod._update_stats(cue_stats, {"intervention_correct": False, "entropy_quartile": 2})
    quartiles = cue_mod._collect_quartiles(baseline_stats, {"cue": cue_stats})
    cue_mod._print_quartile_details(baseline_stats, cue_stats, quartiles)
    assert 2 in quartiles


def test_sorted_row_labels_split_baseline_and_cues():
    rows = [
        {"cue_variant": "baseline", "intervention_correct": True},
        {"cue_variant": "cueA", "intervention_correct": False},
    ]
    baseline, cues = cue_mod._sorted_row_labels(rows)
    assert baseline["correct"] == 1
    assert "cueA" in cues and cues["cueA"]["total"] == 1


def test_ensure_flat_path_and_cleanup(tmp_path, monkeypatch):
    flat_file = tmp_path / "flat.jsonl"
    flat_file.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(flat_jsonl=flat_file, input_jsonl=None)
    path, cleanup = cue_mod._ensure_flat_path(args)
    assert path == flat_file and cleanup is None

    # Use input_jsonl path; stub flatten to create a temp file.
    args = SimpleNamespace(flat_jsonl=None, input_jsonl=tmp_path / "raw.jsonl")
    args.input_jsonl.write_text("{}", encoding="utf-8")
    created = tmp_path / "flattened.jsonl"

    def _fake_flatten(inp, out):
        created.write_text("row", encoding="utf-8")
        return str(created)

    monkeypatch.setattr(cue_mod, "flatten_math_cue_variants", _fake_flatten)
    path, cleanup = cue_mod._ensure_flat_path(args)
    assert path == created and cleanup == created
    cue_mod._cleanup_temp_path(cleanup)
    assert not created.exists()

    # Ensure temp file is removed on exception.
    args = SimpleNamespace(flat_jsonl=None, input_jsonl=tmp_path / "raw2.jsonl")
    args.input_jsonl.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cue_mod, "flatten_math_cue_variants", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        cue_mod._ensure_flat_path(args)


def test_ensure_flat_path_requires_input():
    args = SimpleNamespace(flat_jsonl=None, input_jsonl=None)
    with pytest.raises(RuntimeError):
        cue_mod._ensure_flat_path(args)


def test_parse_args_requires_inputs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        cue_mod.parse_args()


def test_main_happy_path(monkeypatch, tmp_path, capsys):
    data = [
        {
            "cue_variant": "baseline",
            "intervention_correct": True,
            "baseline_correct": True,
            "entropy_quartile": 1,
        },
        {
            "cue_variant": "cueA",
            "intervention_correct": False,
            "baseline_correct": False,
            "entropy_quartile": 1,
            "output_len": 5,
            "tokens_total": 3,
        },
    ]
    flat_path = tmp_path / "flat.jsonl"
    flat_path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--flat-jsonl", str(flat_path), "--table"])
    cue_mod.main()
    out = capsys.readouterr().out
    assert "Baseline accuracy" in out and "cueA" in out


def test_main_writes_table_csv(monkeypatch, tmp_path):
    data = [
        {"cue_variant": "baseline", "intervention_correct": True, "baseline_correct": True, "entropy_quartile": 1},
        {"cue_variant": "cueA", "intervention_correct": True, "baseline_correct": False, "entropy_quartile": 1},
    ]
    flat_path = tmp_path / "flat.csv.jsonl"
    flat_path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    csv_out = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["prog", "--flat-jsonl", str(flat_path), "--table-csv", str(csv_out)])
    cue_mod.main()
    assert csv_out.exists()
    content = csv_out.read_text()
    assert "cue_variant" in content and "cueA" in content


def test_write_table_csv_no_rows(tmp_path):
    path = tmp_path / "empty.csv"
    cue_mod._write_table_csv(tuple(), path)
    assert not path.exists()


def test_module_executes_main_under_dunder_main(monkeypatch, tmp_path):
    flat_path = tmp_path / "mini.jsonl"
    flat_path.write_text(
        json.dumps(
            {"cue_variant": "baseline", "intervention_correct": True, "baseline_correct": True, "entropy_quartile": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--flat-jsonl", str(flat_path)])
    monkeypatch.delitem(sys.modules, "src.analysis.cue_delta_accuracy", raising=False)
    runpy.run_module("src.analysis.cue_delta_accuracy", run_name="__main__")
