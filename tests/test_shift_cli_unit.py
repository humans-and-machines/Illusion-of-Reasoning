import sys

import src.annotate.cli.shift_cli as cli


def test_build_argparser_defaults():
    parser = cli.build_argparser()
    args = parser.parse_args(["/root"])
    assert args.results_root == "/root"
    assert args.backend == "azure"
    assert args.passes == "pass1"
    assert args.clean_failed_first is False


def test_main_runs_clean_and_annotation(monkeypatch, tmp_path, capsys):
    called = {"clean": [], "annotate": [], "scan": []}

    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kwargs: None)

    def fake_clean(path):
        called["clean"].append(path)

    def fake_scan(root, split):
        called["scan"].append((root, split))
        return [str(tmp_path / "f.jsonl")]

    def fake_progress(path_obj):
        return (2, 1)

    def fake_annotate(path, opts):
        called["annotate"].append((path, opts))

    monkeypatch.setattr(cli, "_clean_failed_root", fake_clean)
    monkeypatch.setattr(cli, "scan_jsonl", fake_scan)
    monkeypatch.setattr(cli, "count_progress", fake_progress)
    monkeypatch.setattr(cli, "annotate_file", fake_annotate)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(tmp_path),
            "--split",
            "test",
            "--clean_failed_first",
            "--passes",
            "p1,p2",
            "--backend",
            "portkey",
            "--endpoint",
            "ep",
            "--deployment",
            "dep",
            "--use_v1",
            "1",
            "--api_version",
            "v",
            "--dry_run",
            "--force_relabel",
            "--max_calls",
            "5",
            "--jitter",
            "0.1",
            "--loglevel",
            "DEBUG",
        ],
    )

    cli.main()

    assert called["clean"] == [str(tmp_path)]
    assert called["scan"] == [(str(tmp_path), "test")]
    assert called["annotate"][0][0].endswith("f.jsonl")
    opts = called["annotate"][0][1]
    assert opts.dry_run is True and opts.force_relabel is True
    assert opts.client_cfg["backend"] == "portkey"
