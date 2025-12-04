import sys

import src.annotate.cli.clean_cli as cli


def test_main_invokes_clean_root(monkeypatch):
    called = []

    def fake_clean_root(root):
        called.append(root)

    monkeypatch.setattr(cli, "clean_root", fake_clean_root)
    monkeypatch.setattr(sys, "argv", ["prog", "/tmp/root"])

    cli.main()

    assert called == ["/tmp/root"]
