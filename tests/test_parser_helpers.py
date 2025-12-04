import argparse

import pytest

import src.analysis.common.parser_helpers as ph


def test_standard_results_parser_invokes_add_results(monkeypatch):
    captured = {}

    def fake_add(parser, *, dataset_default, model_default, results_root_optional):
        captured["dataset_default"] = dataset_default
        captured["model_default"] = model_default
        captured["optional"] = results_root_optional
        parser.add_argument("--dummy", action="store_true")

    monkeypatch.setattr(ph, "add_results_root_split_and_output_args", fake_add)
    parser = ph.standard_results_parser(dataset_default="DS", model_default="Model")
    args = parser.parse_args([])

    assert captured == {"dataset_default": "DS", "model_default": "Model", "optional": False}
    assert hasattr(args, "dummy")


def test_add_binning_argument_defaults_and_choices():
    parser = argparse.ArgumentParser()
    ph.add_binning_argument(parser, default="quantile")
    args = parser.parse_args([])
    assert args.binning == "quantile"
    with pytest.raises(SystemExit):
        parser.parse_args(["--binning", "invalid"])


def test_add_carpark_softscore_args_defaults():
    parser = argparse.ArgumentParser()
    ph.add_carpark_softscore_args(parser, op_default="lt", threshold_default=0.25)
    args = parser.parse_args([])
    assert args.carpark_success_op == "lt"
    assert args.carpark_soft_threshold == 0.25


def test_add_entropy_range_args_sets_all(monkeypatch):
    parser = argparse.ArgumentParser()
    ph.add_entropy_range_args(parser)
    args = parser.parse_args(
        [
            "--bins",
            "20",
            "--binning",
            "uniform",
            "--share_bins",
            "per_domain",
            "--entropy_min",
            "0.1",
            "--entropy_max",
            "2.5",
        ],
    )
    assert args.bins == 20
    assert args.binning == "uniform"
    assert args.share_bins == "per_domain"
    assert args.entropy_min == pytest.approx(0.1)
    assert args.entropy_max == pytest.approx(2.5)
