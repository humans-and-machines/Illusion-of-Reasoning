from types import SimpleNamespace

import src.training.utils.hierarchical_grpo_dependencies as deps


def test_canonicalize_device_none_returns_none():
    assert deps.canonicalize_device(None) is None


def test_canonicalize_device_returns_string_without_torch_device(monkeypatch):
    # When torch.device is missing, fall back to the string on the object.
    monkeypatch.setattr(deps, "torch", SimpleNamespace(device=None))
    device = SimpleNamespace(type="cpu")
    assert deps.canonicalize_device(device) == "cpu"


def test_canonicalize_device_handles_bad_torch_device(monkeypatch):
    # A non-type torch.device raises in isinstance and conversion; we keep the candidate.
    calls = []

    def bad_device(value):
        calls.append(value)
        raise ValueError("boom")

    monkeypatch.setattr(deps, "torch", SimpleNamespace(device=bad_device))
    device = SimpleNamespace(type="cuda:0")

    result = deps.canonicalize_device(device)

    assert result == "cuda:0"
    # Conversion attempts are made for both candidate forms.
    assert calls == ["cuda:0", "cuda:0"]
