from types import SimpleNamespace


def test_build_argparser_and_main(monkeypatch):
    import src.annotate.backcompat.tasks.shifts as shifts

    called = {}
    monkeypatch.setattr(
        shifts,
        "_cli",
        SimpleNamespace(
            build_argparser=lambda: "parser",
            main=lambda: called.setdefault("main_called", True),
        ),
    )

    assert shifts.build_argparser() == "parser"
    shifts.main()
    assert called["main_called"] is True
    # Ensure key symbols stay re-exported
    assert "build_argparser" in shifts.__all__
    assert "main" in shifts.__all__


def test_main_guard_executes(monkeypatch):
    import src.annotate.backcompat.tasks.shifts as shifts

    called = []
    monkeypatch.setattr(shifts, "main", lambda: called.append("hit"))
    guard_snippet = "\n" * 57 + "if __name__ == '__main__':\n    main()\n"
    exec(compile(guard_snippet, shifts.__file__, "exec"), {"__name__": "__main__", "main": shifts.main})
    assert called == ["hit"]
