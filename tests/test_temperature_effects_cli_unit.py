import src.analysis.temperature_effects_cli as te


def test_build_temperature_effects_arg_parser_defaults():
    parser = te.build_temperature_effects_arg_parser()
    args = parser.parse_args(["--temps", "0.3", "0.7"])
    assert args.label_crossword == "Crossword"
    assert args.include_math2 is True
    assert args.low_alias == 0.3
    assert args.parallel == "process" and args.workers == 40


def test_build_temperature_effects_arg_parser_custom_values():
    parser = te.build_temperature_effects_arg_parser()
    argv = [
        "--temps",
        "0.3",
        "--label_crossword",
        "CW",
        "--skip_substr",
        "foo",
        "bar",
        "--parallel",
        "thread",
        "--workers",
        "2",
        "--chunksize",
        "1",
        "--make_plot",
    ]
    args = parser.parse_args(argv)
    assert args.label_crossword == "CW"
    assert args.skip_substr == ["foo", "bar"]
    assert args.parallel == "thread"
    assert args.workers == 2
    assert args.chunksize == 1
    assert args.make_plot is True
