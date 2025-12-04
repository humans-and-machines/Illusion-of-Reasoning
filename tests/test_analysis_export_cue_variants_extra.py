import csv
import json

from src.analysis.export_cue_variants import DEFAULT_PASSES, build_argparser, export_cue_variants, iter_flat_rows


def test_iter_flat_rows_serializes_fields(tmp_path, monkeypatch):
    rec = {
        "problem": "p1",
        "step": 3,
        "sample_idx": 7,
        "pass1": {
            "is_correct_pred": True,
            "has_reconsider_cue": 0,
            "reconsider_markers": ["a", "b"],
            "entropy": "0.5",
            "shift_markers_v1": ["m1"],
        },
    }
    path = tmp_path / "step0003.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.analysis.export_cue_variants.iter_records_from_file",
        lambda p: [rec],
    )
    monkeypatch.setattr(
        "src.analysis.export_cue_variants.nat_step_from_path",
        lambda p: 3,
    )
    rows = list(iter_flat_rows([str(path)], DEFAULT_PASSES))
    assert rows[0]["variant"] == "pass1"
    assert rows[0]["is_baseline"] == 1 and rows[0]["is_reconsider_variant"] == 0
    assert rows[0]["reconsider_markers"] == '["a", "b"]'
    assert rows[0]["shift_markers_v1"] == '["m1"]'


def test_export_cue_variants_writes_csv(tmp_path, monkeypatch):
    # Prepare two passes
    rec = {
        "problem": "p1",
        "step": 1,
        "pass1": {"is_correct_pred": 1},
        "pass2": {"is_correct_pred": 0},
    }
    in_dir = tmp_path / "results"
    in_path = in_dir / "step0001.jsonl"
    in_dir.mkdir(parents=True, exist_ok=True)
    in_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.analysis.export_cue_variants.scan_jsonl_files",
        lambda root, split_substr=None: [str(in_path)],
    )
    monkeypatch.setattr(
        "src.analysis.export_cue_variants.iter_records_from_file",
        lambda p: [rec],
    )
    out_csv = tmp_path / "out.csv"
    export_cue_variants(str(in_dir), split_substr=None, out_csv=str(out_csv), passes=DEFAULT_PASSES[:2])

    with out_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {r["variant"] for r in rows} == {"pass1", "pass2"}


def test_build_argparser_defaults():
    parser = build_argparser()
    args = parser.parse_args(["rootdir", "--split", "test"])
    assert args.results_root == "rootdir"
    assert args.split == "test"
