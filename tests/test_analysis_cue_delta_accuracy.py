from analysis import cue_delta_accuracy as cue_mod


def test_build_table_and_rows_generate_expected_values():
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
            "cue_variant": "hint",
            "intervention_correct": False,
            "baseline_correct": False,
        },
        {
            "domain": "Math",
            "entropy_quartile": 2,
            "cue_variant": "hint",
            "intervention_correct": True,
            "baseline_correct": True,
        },
    ]
    table, domains, quartiles, variants = cue_mod._build_table(rows)
    assert domains == ("Math",)
    assert quartiles["Math"] == (1, 2)
    assert variants == ("baseline", "hint")

    table_rows = cue_mod._table_rows_from_stats(
        table,
        domains,
        quartiles,
        variants,
    )
    assert len(table_rows) == 1
    row = table_rows[0]
    assert row["N"] == 1
    assert row["baseline_acc"] == 1.0
    assert row["cue_acc"] == 0.0
    assert row["cue_acc_baseline_wrong"] == 0.0


def test_sorted_row_labels_and_pct_helpers(tmp_path):
    rows = [
        {
            "cue_variant": "baseline",
            "intervention_correct": True,
            "baseline_correct": True,
            "entropy_quartile": 1,
            "output_len": 10,
        },
        {
            "cue_variant": "alt",
            "intervention_correct": False,
            "baseline_correct": False,
            "entropy_quartile": 1,
            "tokens_total": 5,
        },
    ]
    baseline, cues = cue_mod._sorted_row_labels(rows)
    assert baseline["total"] == 1
    assert baseline["correct"] == 1
    alt_stats = cues["alt"]
    assert alt_stats["quartiles"][1]["total"] == 1
    assert alt_stats["quartiles"][1]["correct"] == 0
    assert cue_mod._pct(1, 0) is None
    assert cue_mod._format_pct(None) == "n/a"
    assert cue_mod._format_pct(0.5) == "50.0%"
    assert cue_mod._format_diff(None) == "n/a"
    assert cue_mod._format_diff(-0.1) == "-10.0pp"
    assert cue_mod._format_pct_or_na(None) == "n/a"


def test_write_table_csv_creates_file(tmp_path):
    rows = (
        {
            "domain": "Math",
            "quartile": 1,
            "cue_variant": "hint",
            "N": 2,
            "baseline_acc": 0.5,
            "cue_acc": 1.0,
            "cue_acc_baseline_wrong": 0.0,
        },
    )
    out_path = tmp_path / "table.csv"
    cue_mod._write_table_csv(rows, out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "cue_variant" in content
    assert "baseline_acc" in content
