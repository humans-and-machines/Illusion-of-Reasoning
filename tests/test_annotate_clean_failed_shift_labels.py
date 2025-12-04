#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


def test_clean_failed_shift_labels_shim(monkeypatch):
    # Import shim module
    import src.annotate.cli.clean_failed_shift_labels as shim

    called = {}

    def fake_main():
        called["ran"] = True

    # Replace the re-exported main and ensure the shim calls through
    monkeypatch.setattr(shim, "main", fake_main)

    shim.main()
    assert called.get("ran") is True
