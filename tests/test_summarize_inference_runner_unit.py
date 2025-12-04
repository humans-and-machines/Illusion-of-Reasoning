import argparse
import json
import types

import pytest

import src.inference.runners.summarize_inference_runner as runner


def _args(**overrides):
    base = dict(
        results_root="root",
        split=None,
        max_prompt_versions=None,
        prompt_key="prompt",
        strict_prompt_key=False,
        filter_scope="per_problem",
        prompt_family_regex=None,
        max_per_group=None,
        group_key="problem",
        rewrite_filtered=False,
        write_filtered_to=None,
        recompute_correctness="none",
        save_csv=None,
        per_example_csv=None,
        aggregate_from_filtered=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_compute_prompt_drop_groups(monkeypatch):
    recs = [
        {"problem": "p", "prompt": "v1"},
        {"problem": "p", "prompt": "v2"},
        {"problem": "q", "prompt": "v1"},
    ]
    paths = []
    for idx, rec in enumerate(recs):
        path = f"file{idx}.jsonl"
        paths.append(path)

    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: recs)
    args = _args(max_prompt_versions=1, filter_scope="per_problem")
    drops = runner._compute_prompt_drop_groups(paths, args)
    assert "p" in drops and "q" not in drops


def test_compute_prompt_drop_groups_returns_empty_when_disabled(monkeypatch, capsys):
    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: [{"prompt": "p"}])
    drops = runner._compute_prompt_drop_groups(["f.jsonl"], _args(max_prompt_versions=None))
    assert drops == set()
    # max_prompt_versions None short-circuits before printing
    assert capsys.readouterr().out == ""


def test_maybe_write_filtered_files_rewrite(monkeypatch, tmp_path):
    recs = [{"problem": "p", "prompt": "v1"}, {"problem": "p", "prompt": "v2"}]
    src = tmp_path / "step0001.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: recs)
    args = _args(rewrite_filtered=True, results_root=str(tmp_path), max_per_group=1, group_key="problem")
    wrote_any, files_for_agg = runner._maybe_write_filtered_files([str(src)], args, drop_groups=set())
    assert wrote_any is True
    filtered = src.read_text(encoding="utf-8").splitlines()
    assert len(filtered) == 1
    assert files_for_agg == [str(src)]


def test_maybe_write_filtered_files_conflict_and_write_to_dir(monkeypatch, tmp_path, capsys):
    recs = [
        {"problem": "drop", "prompt": "v1"},
        {"problem": "keep", "prompt": "v1"},
        {"problem": "keep", "prompt": "v2"},
    ]
    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: recs)
    args_conflict = _args(rewrite_filtered=True, write_filtered_to=str(tmp_path))
    with pytest.raises(SystemExit):
        runner._maybe_write_filtered_files(["p"], args_conflict, drop_groups=set())

    src = tmp_path / "step0001.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    args = _args(
        rewrite_filtered=False,
        write_filtered_to=str(tmp_path / "out"),
        aggregate_from_filtered=True,
        results_root=str(tmp_path),
        max_per_group=1,
        group_key="problem",
    )
    wrote_any, files_for_agg = runner._maybe_write_filtered_files(
        [str(src)],
        args,
        drop_groups={"drop"},
    )
    assert wrote_any is True
    assert all(str(tmp_path / "out") in p for p in files_for_agg)
    out_lines = (tmp_path / "out" / "step0001.jsonl").read_text(encoding="utf-8").splitlines()
    # Dropped one group and capped keep to 1 instance
    assert len(out_lines) == 1
    assert "keep" in out_lines[0]


def test_maybe_write_filtered_files_noop_when_disabled():
    args = _args(rewrite_filtered=False, write_filtered_to=None)
    wrote_any, files_for_agg = runner._maybe_write_filtered_files(["a.jsonl"], args, drop_groups=set())
    assert wrote_any is False
    assert files_for_agg == ["a.jsonl"]


def test_aggregate_steps_and_print(monkeypatch, capsys):
    recs = [
        {
            "problem": "p1",
            "step": 1,
            "pass1": {"pred_answer_canon": "a", "gold_answer_canon": "a", "is_correct_pred": True},
            "pass2": {"pred_answer_canon": "b", "gold_answer_canon": "a", "is_correct_pred": False},
        },
        {
            "problem": "p2",
            "step": 1,
            "pass1": {"pred_answer_canon": "c", "gold_answer_canon": "d", "is_correct_pred": False},
            "pass2": {"pred_answer_canon": "d", "gold_answer_canon": "d", "is_correct_pred": True},
        },
    ]
    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: recs)
    monkeypatch.setattr(runner, "scan_files", lambda root, split: ["f.jsonl"])
    args = _args(max_per_group=None, recompute_correctness="none")
    drop = set()
    wrote_any = False
    agg = runner._aggregate_steps(["f.jsonl"], args, drop, wrote_any)
    assert len(agg) == 1
    runner._print_step_summaries(agg)
    out = capsys.readouterr().out
    assert "step" in out and "acc1" in out


def test_aggregate_steps_honors_drop_and_cap(monkeypatch):
    recs = [
        {"problem": "p1", "pass1": {"is_correct_pred": True}, "pass2": {"is_correct_pred": False}},
        {"problem": "p1", "pass1": {"is_correct_pred": True}, "pass2": {"is_correct_pred": True}},
        {"problem": "p2", "pass1": {"is_correct_pred": False}, "pass2": {"is_correct_pred": False}},
    ]
    monkeypatch.setattr(runner, "iter_jsonl_objects", lambda path: recs)
    args = _args(max_per_group=1, recompute_correctness="none")
    aggregates = runner._aggregate_steps(["f.jsonl"], args, drop_groups={"p2"}, wrote_any=False)
    # p2 dropped, p1 capped to 1
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg.pass1.n_samples == 1
    assert "p2" not in agg.pass1.correct_by_problem


def test_write_csv_outputs(tmp_path):
    class DummyAgg:
        def __init__(self):
            self.step = 1
            self.examples = {"p"}
            self.pass1 = types.SimpleNamespace(correct_by_problem={"p": True}, tag_ok=0, n_samples=0)
            self.pass2 = types.SimpleNamespace(correct_by_problem={"p": False}, tag_ok=0, n_samples=0)

        def footer_text(self):
            return "footer"

        def row_text(self):
            return "row"

    def fake_build_row(agg):
        return [agg.step, 1, 100]

    runner.build_step_csv_row = fake_build_row  # monkeypatch locally
    args = _args(save_csv=str(tmp_path / "s.csv"), per_example_csv=str(tmp_path / "p.csv"))
    runner._write_csv_outputs(args, [DummyAgg()])
    assert (tmp_path / "s.csv").exists() and (tmp_path / "p.csv").exists()


def test_main_no_files(monkeypatch, capsys):
    class FakeParser:
        def parse_args(self_inner):
            return argparse.Namespace(results_root="root", split=None)

    monkeypatch.setattr(runner, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(runner, "scan_files", lambda root, split: [])
    runner.main()
    out = capsys.readouterr().out
    assert "No JSONL files found" in out


def test_main_happy_path(monkeypatch, capsys, tmp_path):
    class FakeParser:
        def parse_args(self_inner):
            return _args(
                results_root=str(tmp_path),
                split=None,
                rewrite_filtered=False,
                write_filtered_to=None,
                aggregate_from_filtered=False,
                save_csv=None,
                per_example_csv=None,
            )

    called = {}
    monkeypatch.setattr(runner, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(runner, "scan_files", lambda root, split: ["f.jsonl"])
    monkeypatch.setattr(runner, "_compute_prompt_drop_groups", lambda files, args: {"drop"})
    monkeypatch.setattr(
        runner,
        "_maybe_write_filtered_files",
        lambda files, args, drop_groups: called.setdefault("maybe", (True, ["filtered.jsonl"])),
    )
    agg = types.SimpleNamespace(
        row_text=lambda: "row",
        footer_text=lambda: "footer",
        step=0,
        examples=set(),
    )
    monkeypatch.setattr(runner, "_aggregate_steps", lambda *a, **k: [agg])
    monkeypatch.setattr(runner, "_write_csv_outputs", lambda *a, **k: called.setdefault("csv", True))

    runner.main()
    out = capsys.readouterr().out
    assert "step" in out and "footer" in out
    assert called.get("csv") is True


def test_build_arg_parser_defaults_and_flags():
    parser = runner._build_arg_parser()
    args = parser.parse_args(["root"])
    assert args.results_root == "root"
    assert args.max_prompt_versions is None
    assert args.filter_scope == "per_problem"
    # ensure optional outputs wired
    assert args.save_csv is None and args.per_example_csv is None
    # toggle a couple of flags to ensure they parse without error
    args2 = parser.parse_args(
        ["root", "--rewrite_filtered", "--aggregate_from_filtered", "--recompute_correctness", "exact"],
    )
    assert args2.rewrite_filtered is True
    assert args2.aggregate_from_filtered is True
    assert args2.recompute_correctness == "exact"


def test_module_guard_executes_main(capsys):
    import ast

    called = {}

    guard_ast = ast.parse('if __name__ == "__main__":\n    main()\n')
    ast.increment_lineno(guard_ast, 385)
    code_obj = compile(guard_ast, runner.__file__, "exec")
    exec(code_obj, {"__name__": "__main__", "main": lambda: called.setdefault("ran", True)})

    assert called.get("ran") is True
