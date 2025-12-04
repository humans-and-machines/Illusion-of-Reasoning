import types

import src.annotate.core.clean_failed_shift_core as shim


def test_clean_failed_shift_core_exports_from_clean_core(monkeypatch):
    fake = types.SimpleNamespace(
        BAD_RATIONALE_PREFIXES=("bad",),
        SHIFT_FIELDS=("f1",),
        clean_file=lambda path: "file",
        clean_root=lambda root: "root",
    )
    monkeypatch.setitem(shim.__dict__, "BAD_RATIONALE_PREFIXES", fake.BAD_RATIONALE_PREFIXES)
    monkeypatch.setitem(shim.__dict__, "SHIFT_FIELDS", fake.SHIFT_FIELDS)
    monkeypatch.setitem(shim.__dict__, "clean_file", fake.clean_file)
    monkeypatch.setitem(shim.__dict__, "clean_root", fake.clean_root)

    assert shim.BAD_RATIONALE_PREFIXES == ("bad",)
    assert shim.SHIFT_FIELDS == ("f1",)
    assert shim.clean_file("p") == "file"
    assert shim.clean_root("r") == "root"
