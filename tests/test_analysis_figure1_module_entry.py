#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


def test_figure1_main_calls_component(monkeypatch):
    # Import once and replace the delegated entrypoint with a stub to ensure
    # this thin wrapper forwards correctly.
    from src.analysis import figure_1

    called = {"count": 0}

    def fake_main():
        called["count"] += 1

    monkeypatch.setattr(figure_1, "_figure1_main", fake_main)

    figure_1.main()
    assert called["count"] == 1
