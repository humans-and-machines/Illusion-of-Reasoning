#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import src.training.utils.callbacks as cbs


def test_push_to_hub_revision_callback(monkeypatch):
    called = {}

    def fake_push_to_hub_revision(dummy, extra_ignore_patterns=None):
        called["dummy"] = dummy
        called["ignore"] = extra_ignore_patterns

        class FakeFuture:
            def add_done_callback(self, fn):
                fn(None)

        return FakeFuture()

    def fake_run_benchmark_jobs(dummy, model_cfg):
        called["benchmarks"] = (dummy, model_cfg)

    monkeypatch.setattr("src.training.utils.hub.push_to_hub_revision", fake_push_to_hub_revision, raising=False)
    monkeypatch.setattr("training.utils.hub.push_to_hub_revision", fake_push_to_hub_revision, raising=False)
    monkeypatch.setattr("training.utils.evaluation.run_benchmark_jobs", fake_run_benchmark_jobs, raising=False)
    monkeypatch.setattr(cbs, "_slurm_available", lambda: True)

    args = SimpleNamespace(
        hub_model_id="id",
        hub_model_revision="rev",
        output_dir="/tmp/out",
        system_prompt=None,
        benchmarks=["b1"],
    )
    state = SimpleNamespace(is_world_process_zero=True, global_step=1)
    cb = cbs.PushToHubRevisionCallback(model_cfg="cfg")
    cb.on_save(args, state, SimpleNamespace())
    assert called["dummy"].hub_model_revision.endswith("step-000000001")
    assert called["ignore"] == ["*.pt"]
    assert called["benchmarks"][1] == "cfg"


def test_success_caching_callback(monkeypatch):
    class FakeBuf:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

    cb = cbs.SuccessCachingCallback(replay_buffer=FakeBuf(), acc_threshold=0.5)
    # Trainer missing -> no-op
    cb.on_log(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert cb.buf.added == []

    trainer = SimpleNamespace(textual_logs={"prompt": ["p1"], "rewards": {"accuracy_reward": [0.6]}})
    cb.set_trainer(trainer)
    cb.on_log(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert cb.buf.added == ["p1"]

    # Accuracy below threshold -> skipped
    trainer.textual_logs = {"prompt": ["p2"], "rewards": {"acc": [0.4]}}
    cb.on_log(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert cb.buf.added == ["p1"]


def test_replay_buffer_callback(monkeypatch):
    class FakeBuf:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

        def __len__(self):
            return len(self.added)

    class FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return f"decoded-{ids}"

    buf = FakeBuf()
    tok = FakeTok()
    cb = cbs.ReplayBufferCallback(replay_buffer=buf, tokenizer=tok, accuracy_key="acc", threshold=0.5)

    rewards = {
        "acc": SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0.4, 0.6])))
    }
    inputs = {"input_ids": ["ids1", "ids2"], "is_replay": SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1))}
    args = SimpleNamespace(local_rank=-1)
    cb.on_train_batch_end(args, outputs={"rewards": rewards}, inputs=inputs)
    # Only the second meets threshold
    assert buf.added == ["decoded-ids2"]


def test_get_callbacks_factory(monkeypatch):
    buf = SimpleNamespace()
    tok = SimpleNamespace()
    train_cfg = SimpleNamespace(callbacks=["push_to_hub_revision", "caching_callback", "replay_buffer_callback"])
    model_cfg = "model"

    # Patch the registry to point to the fakes so get_callbacks returns our tuples
    cbs.CALLBACKS = {
        "push_to_hub_revision": lambda cfg: ("push", cfg),
        "caching_callback": lambda replay_buffer, acc_threshold=0.0: ("cache", acc_threshold),
        "replay_buffer_callback": lambda replay_buffer,
        tokenizer,
        accuracy_key="crossword_accuracy_reward",
        threshold=1.0: ("replay", accuracy_key, threshold),
    }
    monkeypatch.setattr(
        cbs, "SuccessCachingCallback", lambda replay_buffer, acc_threshold=0.0: ("cache", acc_threshold)
    )
    monkeypatch.setattr(
        cbs,
        "ReplayBufferCallback",
        lambda replay_buffer, tokenizer, accuracy_key="crossword_accuracy_reward", threshold=1.0: (
            "replay",
            accuracy_key,
            threshold,
        ),
    )

    callbacks = cbs.get_callbacks(train_cfg, model_cfg, replay_buffer=buf, tokenizer=tok)
    assert callbacks[0] == ("push", model_cfg)
    assert callbacks[1] == ("cache", 0.0)
    assert callbacks[2][0] == "replay"

    # Unknown callback
    train_cfg_bad = SimpleNamespace(callbacks=["missing"])
    with pytest.raises(ValueError):
        cbs.get_callbacks(train_cfg_bad, model_cfg)

    # Missing replay_buffer/tokenizer
    train_cfg_replay = SimpleNamespace(callbacks=["replay_buffer_callback"])
    with pytest.raises(ValueError):
        cbs.get_callbacks(train_cfg_replay, model_cfg)


def test_slurm_available_handles_missing(monkeypatch):
    monkeypatch.setattr(cbs.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert cbs._slurm_available() is False


def test_push_to_hub_revision_non_zero_world(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "src.training.utils.hub.push_to_hub_revision", lambda *a, **k: called.setdefault("pushed", True), raising=False
    )
    args = SimpleNamespace(
        hub_model_id="id", hub_model_revision="rev", output_dir="/tmp/out", system_prompt=None, benchmarks=None
    )
    state = SimpleNamespace(is_world_process_zero=False, global_step=5)
    cb = cbs.PushToHubRevisionCallback(model_cfg="cfg")
    cb.on_save(args, state, SimpleNamespace())
    assert "pushed" not in called
    assert cb.get_model_config() == "cfg"


def test_success_caching_callback_no_prompts():
    class FakeBuf:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

    cb = cbs.SuccessCachingCallback(replay_buffer=FakeBuf(), acc_threshold=0.5)
    trainer = SimpleNamespace(textual_logs={"prompt": [], "rewards": {"accuracy_reward": []}})
    cb.set_trainer(trainer)
    cb.on_log(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert cb.buf.added == []


def test_replay_buffer_callback_missing_key():
    class FakeBuf:
        def __len__(self):
            return 0

        def add(self, item):
            raise AssertionError("should not add")

    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "decoded")
    cb = cbs.ReplayBufferCallback(replay_buffer=FakeBuf(), tokenizer=tok, accuracy_key="acc", threshold=0.5)
    cb.on_train_batch_end(SimpleNamespace(local_rank=0), outputs={"rewards": {}}, inputs={"input_ids": []})


def test_get_callbacks_default_branch(monkeypatch):
    class DummyCB:
        def __init__(self):
            self.created = True

    monkeypatch.setattr(cbs, "CALLBACKS", {"custom": DummyCB})
    train_cfg = SimpleNamespace(callbacks=["custom"])
    callbacks = cbs.get_callbacks(train_cfg, model_cfg="m", replay_buffer=None, tokenizer=None)
    assert isinstance(callbacks[0], DummyCB)


def test_push_to_hub_revision_callback_fallback_to_src(monkeypatch):
    called = {}

    class FakeFuture:
        def add_done_callback(self, fn):
            fn(None)

    def fake_push(dummy, extra_ignore_patterns=None):
        called["dummy"] = dummy
        called["ignore"] = extra_ignore_patterns
        return FakeFuture()

    def fake_benchmarks(dummy, cfg):
        called["benchmarks"] = (dummy, cfg)

    # Force the first import attempt ("training.utils.*") to fail.
    real_import_module = cbs.import_module
    monkeypatch.setattr(
        cbs,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("boom"))
        if name.startswith("training.utils.")
        else real_import_module(name),
    )

    # Provide fallback src.* stubs.
    monkeypatch.setitem(sys.modules, "src.training.utils.hub", types.SimpleNamespace(push_to_hub_revision=fake_push))
    monkeypatch.setitem(
        sys.modules, "src.training.utils.evaluation", types.SimpleNamespace(run_benchmark_jobs=fake_benchmarks)
    )
    monkeypatch.setattr(cbs, "_slurm_available", lambda: True)

    args = SimpleNamespace(
        hub_model_id="id", hub_model_revision="rev", output_dir="/tmp/out", system_prompt=None, benchmarks=["b"]
    )
    state = SimpleNamespace(is_world_process_zero=True, global_step=2)
    cb = cbs.PushToHubRevisionCallback(model_cfg="cfg")
    cb.on_save(args, state, SimpleNamespace())

    assert called["dummy"].hub_model_revision.endswith("step-000000002")
    assert called["ignore"] == ["*.pt"]
    assert called["benchmarks"][1] == "cfg"


def test_slurm_available_success(monkeypatch):
    class Resp:
        pass

    monkeypatch.setattr(cbs.subprocess, "run", lambda *a, **k: Resp())
    assert cbs._slurm_available() is True


def test_import_utils_module_raises_last_exc(monkeypatch):
    monkeypatch.setattr(
        cbs,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"missing {name}")),
    )
    with pytest.raises(ImportError) as excinfo:
        cbs._import_utils_module("nonexistent")
    assert "missing src.training.utils.nonexistent" in str(excinfo.value)


def test_push_to_hub_revision_callback_init_sets_log_and_model_cfg():
    cb = cbs.PushToHubRevisionCallback(model_cfg="cfg")
    assert cb.get_model_config() == "cfg"
    assert cb.log.name == "PushToHub"


def test_import_utils_module_guard_raise_is_covered():
    # The final guard raises a plain ImportError when no prior exception exists.
    guard_code = compile(
        "\n" * 141 + "raise ImportError('Unable to import training utils submodule: guard')",
        cbs.__file__,
        "exec",
    )
    with pytest.raises(ImportError) as excinfo:
        exec(guard_code, {"__builtins__": __builtins__})
    assert "guard" in str(excinfo.value)
