#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import src.analysis.h1_analysis as h1


def test_scan_files_passes_through(monkeypatch):
    monkeypatch.setattr(h1, "scan_jsonl_files", lambda root, split_substr=None: ["a.jsonl", "b.jsonl"])
    assert h1.scan_files("root", "train") == ["a.jsonl", "b.jsonl"]
