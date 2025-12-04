import types

import pandas as pd

import src.analysis.graph_1 as g1


def test_parse_float_list_filters_bad_tokens():
    assert g1._parse_float_list("1, 2, bad, 3") == [1.0, 2.0, 3.0]
    assert g1._parse_float_list("   ") is None
    assert g1._parse_float_list(None) is None


def test_parse_float_list_skips_empty_parts_from_trailing_separator():
    assert g1._parse_float_list("1,2,") == [1.0, 2.0]


def test_main_smoke_with_stubs(monkeypatch, tmp_path, capsys):
    # Stub arg parser to avoid CLI parsing.
    args = types.SimpleNamespace(
        label_math="M1",
        label_math2="M2",
        min_per_group=1,
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=5,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        dataset_name="D",
        model_name="M",
        out_dir=str(tmp_path),
        plot_units="pp",
        dpi=100,
        width_in=1.0,
        height_scale=1.0,
        marker_size=1.0,
        overlay_title=None,
        overlay_width_in=1.0,
        overlay_height_scale=1.0,
        yticks_pp="-20,0,20",
        yticks_prob="-0.2,0,0.2",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_pp=1.0,
        ylim_pad_prob=0.01,
    )
    monkeypatch.setattr(g1, "_build_arg_parser", lambda: types.SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g1, "build_files_by_domain_for_args", lambda _args: ({"Crossword": ["f"]}, str(tmp_path)))
    monkeypatch.setattr(g1, "gpt_keys_for_mode", lambda mode: ["k"])
    monkeypatch.setattr(
        g1, "compute_effective_max_step", lambda args, hard_max_step=None: args.max_step or hard_max_step
    )
    monkeypatch.setattr(g1, "make_carpark_success_fn", lambda *a, **k: lambda *_a, **_k: True)
    monkeypatch.setattr(
        g1,
        "load_rows",
        lambda files_by_domain, load_config: pd.DataFrame(
            {"domain": ["Crossword", "Crossword"], "step": [0, 1], "aha": [0, 1], "correct": [1, 0]}
        ),
    )
    fake_df = pd.DataFrame(
        {"domain": ["Crossword"], "bin_left": [0], "bin_right": [1], "acc_noaha": [0.1], "acc_aha": [0.2], "step": [0]}
    )
    monkeypatch.setattr(g1, "_compute_per_step", lambda df, min_per_group: ({"Crossword": fake_df}, [fake_df]))
    monkeypatch.setattr(g1, "plot_panels", lambda *a, **k: None)
    monkeypatch.setattr(g1, "plot_overlay_all", lambda *a, **k: None)

    g1.main()
    out = capsys.readouterr().out
    assert "Raw effect per step" in out
    # Ensure files were written to the provided out_dir.
    assert any(p.name.endswith(".csv") for p in tmp_path.iterdir())
