#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import src.analysis.plotting_styles as styles


def test_parse_color_overrides_handles_empty():
    assert styles.parse_color_overrides(None) == {}
    assert styles.parse_color_overrides("") == {}


def test_parse_color_overrides_parses_pairs():
    overrides = styles.parse_color_overrides("bar_primary:#000, foo: #fff , invalid")
    assert overrides["bar_primary"] == "#000"
    assert overrides["foo"] == "#fff"
    assert "invalid" not in overrides


def test_cmap_colors_prefers_matplotlib_colormaps(monkeypatch):
    fake_colors = [(0.1, 0.2, 0.3, 0.9), (0.4, 0.5, 0.6, 0.7)]

    class FakeColormaps:
        def __getitem__(self, name):
            return SimpleNamespace(colors=fake_colors)

    monkeypatch.setattr(styles.matplotlib, "colormaps", FakeColormaps())
    result = styles.cmap_colors("any")
    assert result == [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]


def test_cmap_colors_falls_back_to_cm(monkeypatch):
    class FailingColormaps:
        def __getitem__(self, name):
            raise KeyError(name)

    monkeypatch.setattr(styles.matplotlib, "colormaps", FailingColormaps())
    monkeypatch.setattr(styles.cm, "get_cmap", lambda name: SimpleNamespace(colors=[(1, 0, 0)]))

    result = styles.cmap_colors("missing")
    assert result == [(1.0, 0.0, 0.0)]


def test_cmap_colors_handles_attribute_error(monkeypatch):
    class FakeColormap:
        # no colors attribute to trigger AttributeError
        pass

    class FakeColormaps:
        def __getitem__(self, name):
            return FakeColormap()

    monkeypatch.setattr(styles.matplotlib, "colormaps", FakeColormaps())
    monkeypatch.setattr(styles.cm, "get_cmap", lambda name: SimpleNamespace(colors=[(0.2, 0.3, 0.4, 0.5)]))
    assert styles.cmap_colors("attr_missing") == [(0.2, 0.3, 0.4)]


def test_darken_colors_scales_rgb():
    colors = [(0.5, 0.5, 0.5), (1.0, 0.2, 0.0)]
    darkened = styles.darken_colors(colors, factor=0.5)
    assert darkened == [(0.25, 0.25, 0.25), (0.5, 0.1, 0.0)]
