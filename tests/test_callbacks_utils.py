import types

import pytest

import src.training.utils.callbacks as cb


class DummyFuture:
    def __init__(self):
        self.callbacks = []

    def add_done_callback(self, fn):
        self.callbacks.append(fn)
        fn(self)  # invoke immediately to simulate completion


def test_push_to_hub_revision_callback_triggers_upload_and_benchmark(monkeypatch):
    calls = {}

    def fake_import(name):
        if "hub" in name:
            return types.SimpleNamespace(push_to_hub_revision=lambda cfg, extra_ignore_patterns=None: DummyFuture())
        return types.SimpleNamespace(
            run_benchmark_jobs=lambda cfg, model_cfg: calls.setdefault("benchmarks", (cfg.benchmarks, model_cfg))
        )

    monkeypatch.setattr(cb, "import_module", fake_import)
    monkeypatch.setattr(cb, "_slurm_available", lambda: True)

    args = types.SimpleNamespace(
        hub_model_id="repo",
        hub_model_revision="rev",
        output_dir="out",
        system_prompt="sys",
        benchmarks=["b1"],
    )
    state = types.SimpleNamespace(is_world_process_zero=True, global_step=1)
    callback = cb.PushToHubRevisionCallback(model_cfg="model_cfg")
    callback.on_save(args, state, _control=None)
    assert calls["benchmarks"][0] == ["b1"]
    assert calls["benchmarks"][1] == "model_cfg"


def test_success_caching_callback_adds_prompts(monkeypatch):
    added = []

    class FakeBuf:
        def __len__(self):
            return len(added)

        def add(self, prompt):
            added.append(prompt)

    trainer = types.SimpleNamespace(
        textual_logs={
            "prompt": ["p1", "p2"],
            "rewards": {"accuracy_reward": [1.0, 0.4]},
        },
    )
    cbk = cb.SuccessCachingCallback(FakeBuf(), acc_threshold=0.5)
    cbk.set_trainer(trainer)
    cbk.on_log(_args=None, _state=None, _control=None)
    assert added == ["p1"]


def test_replay_buffer_callback_on_train_batch_end(monkeypatch):
    added = []

    class FakeBuf:
        def __len__(self):
            return len(added)

        def add(self, prompt):
            added.append(prompt)

    class FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return f"decoded-{ids[0]}"

    class FakeFlags:
        def __init__(self, val):
            self.val = val

        def sum(self):
            return self

        def item(self):
            return self.val

    class FakeTensor:
        def __init__(self, vals):
            self.vals = vals

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.vals

    cbk = cb.ReplayBufferCallback(replay_buffer=FakeBuf(), tokenizer=FakeTok(), threshold=0.5, accuracy_key="acc")
    outputs = {"rewards": {"acc": FakeTensor([0.4, 0.6])}}
    inputs = {"input_ids": [[1], [2]], "is_replay": FakeFlags(1)}
    args = types.SimpleNamespace(local_rank=-1)
    cbk.on_train_batch_end(args, outputs=outputs, inputs=inputs)
    assert added == ["decoded-2"]
    assert cbk.buffer_size() == 1


def test_get_callbacks_validates(monkeypatch):
    buf = types.SimpleNamespace()
    tok = object()
    model_cfg = object()

    class TrainCfg:
        callbacks = ["caching_callback", "replay_buffer_callback"]

    callbacks = cb.get_callbacks(TrainCfg(), model_cfg, replay_buffer=buf, tokenizer=tok)
    names = [type(c).__name__ for c in callbacks]
    assert "SuccessCachingCallback" in names
    assert "ReplayBufferCallback" in names

    class TrainCfgUnknown:
        callbacks = ["unknown_cb"]

    with pytest.raises(ValueError):
        cb.get_callbacks(TrainCfgUnknown(), model_cfg, replay_buffer=buf, tokenizer=tok)

    class TrainCfgMissing:
        callbacks = ["caching_callback"]

    with pytest.raises(ValueError):
        cb.get_callbacks(TrainCfgMissing(), model_cfg)
