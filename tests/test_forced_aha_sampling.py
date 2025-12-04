import os
from types import SimpleNamespace

import pandas as pd

import src.analysis.forced_aha_sampling as fas


def test_prepare_forced_aha_samples_two_roots(monkeypatch, tmp_path):
    calls = {}

    def fake_load_root(root, split, variant, entropy_field, pass2_key):
        calls.setdefault("loads", []).append((root, split, variant, entropy_field, pass2_key))
        return pd.DataFrame({"root": [root], "variant": [variant]})

    args = SimpleNamespace(
        root1="r1",
        root2="r2",
        split="test",
        entropy_field="entropy",
        pass2_key="pass2",
        out_dir=None,
    )

    out_dir, df1, df2 = fas.prepare_forced_aha_samples(
        args,
        load_root_samples=fake_load_root,
        load_single_root_samples=None,  # unused
    )

    assert out_dir == os.path.join("r1", "forced_aha_effect")
    assert not df1.empty and not df2.empty
    assert calls["loads"][0][2] == "pass1"
    assert calls["loads"][1][2] == "pass2"


def test_prepare_forced_aha_samples_single_root_custom_pass2(monkeypatch, tmp_path):
    calls = {}

    def fake_single(root, split, entropy_field, pass2_key):
        calls["single"] = (root, split, entropy_field, pass2_key)
        return pd.DataFrame({"root": [root]}), pd.DataFrame({"root": [root]})

    args = SimpleNamespace(
        root1="r1",
        root2=None,
        split=None,
        entropy_field="entropy",
        pass2_key="pass2b",
        out_dir=str(tmp_path),
    )

    out_dir, df1, df2 = fas.prepare_forced_aha_samples(
        args,
        load_root_samples=None,  # unused
        load_single_root_samples=fake_single,
    )

    assert out_dir == str(tmp_path)
    assert not df1.empty and not df2.empty
    assert calls["single"] == ("r1", None, "entropy", "pass2b")
