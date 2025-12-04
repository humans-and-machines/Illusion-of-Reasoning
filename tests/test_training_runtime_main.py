import importlib
import os
import sys
from types import SimpleNamespace

import pytest


_PREV_MODS = {
    name: sys.modules.get(name)
    for name in [
        "torch",
        "torch.distributed",
        "torch.utils.data",
        "torch.serialization",
        "transformers",
        "transformers.trainer_utils",
        "trl",
        "trl.trainer.grpo_trainer",
        "datasets",
        "statsmodels",
    ]
}


def _install_module_stubs():
    """Install lightweight module stubs before importing runtime_main."""
    try:
        import torch as real_torch  # type: ignore
    except ImportError:

        class FakeTensor:
            def __init__(self, data=None):
                self.data = data

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.data) if self.data is not None else []

            @property
            def shape(self):
                return getattr(self.data, "shape", ())

        fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: None)
        fake_torch.serialization = SimpleNamespace()
        fake_torch.no_grad = lambda *_a, **_k: (lambda fn: fn)
        fake_torch.Tensor = FakeTensor
        fake_torch.tensor = lambda data=None, **_k: FakeTensor(data)
        fake_torch.SymFloat = FakeTensor  # satisfy attr lookups in downstream code
        fake_torch.SymBool = FakeTensor
        fake_torch.full = lambda shape, value=0, dtype=None: FakeTensor(
            data=[[value] * shape[1]] if isinstance(shape, tuple) else [value]
        )
        fake_torch.ones = lambda shape, dtype=None: FakeTensor(
            data=[[1] * shape[1]] if isinstance(shape, tuple) else [1]
        )
        fake_torch.cat = lambda tensors, dim=0: FakeTensor(
            data=sum((t.data for t in tensors if hasattr(t, "data")), [])
        )
        fake_torch.long = "long"
        sys.modules["torch"] = fake_torch
        sys.modules["torch.distributed"] = SimpleNamespace()
        sys.modules["torch.utils.data"] = SimpleNamespace(DataLoader=object, RandomSampler=object)
        sys.modules["torch.serialization"] = fake_torch.serialization
    else:
        if not hasattr(real_torch, "no_grad"):
            real_torch.no_grad = lambda *_a, **_k: (lambda fn: fn)
        sys.modules["torch"] = real_torch

    try:
        import transformers as real_transformers  # type: ignore
    except ImportError:
        fake_transformers = SimpleNamespace(
            TrainerCallback=object,
            set_seed=lambda *_args, **_kwargs: None,
            AutoTokenizer=SimpleNamespace,
            AutoConfig=SimpleNamespace,
            StoppingCriteria=object,
            utils=SimpleNamespace(
                logging=SimpleNamespace(
                    set_verbosity=lambda *_: None,
                    enable_default_handler=lambda: None,
                    enable_explicit_format=lambda: None,
                )
            ),
        )
        sys.modules["transformers"] = fake_transformers
        sys.modules["transformers.trainer_utils"] = SimpleNamespace(get_last_checkpoint=lambda *_: None)
    else:
        sys.modules["transformers"] = real_transformers
        try:
            import transformers.trainer_utils as trainer_utils  # type: ignore  # noqa: F401
        except ImportError:
            sys.modules["transformers.trainer_utils"] = SimpleNamespace(get_last_checkpoint=lambda *_: None)

    class _StubConfig:
        def __init__(self, *_a, **_k):
            return None

        def to_dict(self):
            return vars(self).copy()

        def update_from_dict(self, values):
            for key, value in values.items():
                setattr(self, key, value)

    sys.modules["trl"] = SimpleNamespace(
        ScriptArguments=_StubConfig,
        GRPOConfig=_StubConfig,
        SFTConfig=_StubConfig,
        get_peft_config=lambda *_args, **_kwargs: None,
    )
    sys.modules["trl.trainer.grpo_trainer"] = SimpleNamespace(
        GRPOTrainer=type("StubGRPOTrainer", (), {"__init__": lambda self, *a, **k: None})
    )

    sys.modules["datasets"] = SimpleNamespace(
        utils=SimpleNamespace(logging=SimpleNamespace(set_verbosity=lambda *_: None))
    )
    sys.modules["statsmodels"] = importlib.import_module("types")  # unused, just to satisfy env import

    # Stub runtime.env to avoid optional-dep churn during import.
    sys.modules["src.training.runtime.env"] = SimpleNamespace(
        datasets=None,
        torch=sys.modules.get("torch"),
        dist=None,
        transformers=None,
        wandb=None,
        DataLoader=None,
        RandomSampler=None,
        AcceleratorState=None,
        ZeroStageEnum=None,
        ZeroParamStatus=None,
        get_last_checkpoint=lambda *_a, **_k: None,
        get_peft_config=lambda *_a, **_k: None,
        GRPOTrainer=type("StubGRPOTrainer", (), {}),
        TrainerCallback=type("StubTrainerCallback", (), {}),
        set_seed=lambda *_a, **_k: None,
        __all__=[],
    )


_install_module_stubs()


runtime_main = importlib.import_module("src.training.runtime.main")
runtime_main = importlib.reload(runtime_main)


@pytest.fixture(autouse=True)
def _restore_modules():
    yield
    for name, mod in _PREV_MODS.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _fake_tokenizer():
    class FakeTokenizer:
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "pad_token" and value is not None and getattr(self, "pad_token_id", None) is None:
                super().__setattr__("pad_token_id", getattr(self, "eos_token_id", None))

        def __init__(self):
            self.pad_token_id = None
            self.eos_token = "[EOS]"
            self.eos_token_id = 1
            self.padding_side = None
            self.pad_token = None

        def add_special_tokens(self, tokens):
            self.pad_token = tokens["pad_token"]
            self.pad_token_id = 2

    return FakeTokenizer()


def test_init_model_and_tokenizer_sets_padding(monkeypatch):
    tok = _fake_tokenizer()
    model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None, output_scores=None, return_dict_in_generate=None),
        generation_config=SimpleNamespace(
            sort_inputs=None, return_dict_in_generate=None, output_scores=None, do_sample=None
        ),
        resize_token_embeddings=lambda n: None,
    )

    monkeypatch.setattr(runtime_main, "get_tokenizer", lambda *_args, **_kwargs: tok)
    monkeypatch.setattr(runtime_main, "get_model", lambda *_args, **_kwargs: model)

    m, t = runtime_main._init_model_and_tokenizer(SimpleNamespace(), SimpleNamespace())
    assert t.padding_side == "left"
    assert m.config.pad_token_id is not None
    assert m.generation_config.do_sample is True


def test_maybe_load_easy_pool_env_toggle(monkeypatch):
    # No easy dataset -> returns None
    assert runtime_main._maybe_load_easy_pool(SimpleNamespace(easy_dataset_name=None), None, None) is None
    monkeypatch.setenv("EASY_DATASET", "on")
    monkeypatch.setattr(runtime_main, "_load_easy_pool", lambda *a, **k: "loaded")
    assert runtime_main._maybe_load_easy_pool(SimpleNamespace(easy_dataset_name=None), None, None) == "loaded"


def test_build_reward_funcs_wraps_rewards(monkeypatch):
    fake_model = SimpleNamespace(eval=lambda: fake_model, requires_grad_=lambda flag: fake_model)
    monkeypatch.setattr(runtime_main, "get_model", lambda *a, **k: fake_model)
    # reward_funcs can be list/dict/single
    monkeypatch.setattr(runtime_main, "get_reward_funcs", lambda *a, **k: {"r": lambda x: x})
    wrapped = runtime_main._build_reward_funcs(SimpleNamespace(), None, None, None)
    assert callable(wrapped["r"])

    monkeypatch.setattr(runtime_main, "get_reward_funcs", lambda *a, **k: [lambda x: x])
    wrapped_list = runtime_main._build_reward_funcs(SimpleNamespace(), None, None, None)
    assert isinstance(wrapped_list, list)


def test_build_reward_funcs_wraps_single_callable(monkeypatch):
    fake_model = SimpleNamespace(eval=lambda: fake_model, requires_grad_=lambda flag: fake_model)
    monkeypatch.setattr(runtime_main, "get_model", lambda *a, **k: fake_model)

    called = {}

    def fake_wrap(fn):
        called["seen"] = fn
        return "wrapped_single"

    monkeypatch.setattr(runtime_main, "_wrap_reward_for_nested", fake_wrap)
    monkeypatch.setattr(runtime_main, "get_reward_funcs", lambda *a, **k: lambda y: y * 2)

    result = runtime_main._build_reward_funcs(SimpleNamespace(), None, None, None)
    assert result == "wrapped_single"
    assert callable(called["seen"])


def test_prepare_datasets_adds_columns(monkeypatch):
    # Build a minimal dataset-like object
    class FakeSplit(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.column_names = list(rows[0].keys())

        def remove_columns(self, col):
            self.column_names = [c for c in self.column_names if c != col]
            return self

        def add_column(self, name, values):
            self.column_names.append(name)
            return self

        def shuffle(self, seed):
            return self

        def select(self, rng):
            return self

        def map(self, fn):
            return FakeDataset([fn(r) for r in self])

        def filter(self, fn):
            return self

    class FakeDataset(dict):
        def __init__(self, rows):
            super().__init__(train=FakeSplit(rows), test=FakeSplit(rows))

        def map(self, fn):
            return FakeDataset([fn(r) for r in self["train"]])

        def filter(self, fn):
            return self

    rows = [{"prompt": "p", "solution": "s", "messages": "m"}]
    fake_ds = FakeDataset(rows)

    monkeypatch.setattr(runtime_main, "get_dataset", lambda *_args, **_kwargs: fake_ds)
    monkeypatch.setattr(runtime_main, "_make_conversation", lambda ex, *_args: ex)
    tok = SimpleNamespace(
        pad_token_id=None, eos_token="[EOS]", eos_token_id=1, padding_side=None, add_special_tokens=lambda _: None
    )
    dataset, train_ds, eval_ds = runtime_main._prepare_datasets(
        SimpleNamespace(
            dataset_prompt_column="prompt",
            dataset_solution_column="solution",
            dataset_train_split="train",
            dataset_test_split="test",
        ),
        SimpleNamespace(do_eval=False, seed=0, system_prompt=None),
        tok,
    )
    assert isinstance(train_ds, runtime_main.ReplayMixDataset)
    assert "task" in dataset["train"].column_names
    assert "is_replay" in dataset["train"].column_names


def test_prepare_datasets_eval_branch_and_padding(monkeypatch):
    class FakeSplit(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.column_names = list(rows[0].keys())
            self.shuffled = False
            self.selected = False

        def remove_columns(self, col):
            self.column_names = [c for c in self.column_names if c != col]
            return self

        def add_column(self, name, values):
            self.column_names.append(name)
            return self

        def shuffle(self, seed):
            self.shuffled = True
            return self

        def select(self, rng):
            self.selected = True
            return self

        def map(self, fn):
            return FakeSplit([fn(r) for r in self])

        def filter(self, fn):
            return self

    class FakeDataset(dict):
        def __init__(self, rows):
            super().__init__(train=FakeSplit(rows), test=FakeSplit(rows))

        def map(self, fn):
            return FakeDataset([fn(r) for r in self["train"]])

        def filter(self, fn):
            return self

    rows = [{"prompt": "p", "solution": "s", "messages": "m"}]
    fake_ds = FakeDataset(rows)
    tok = SimpleNamespace(
        pad_token_id=None,
        eos_token=None,
        eos_token_id=None,
        padding_side=None,
        added=False,
        add_special_tokens=lambda tokens: tok.__setattr__("added", True),
    )
    monkeypatch.setattr(runtime_main, "get_dataset", lambda *_args, **_kwargs: fake_ds)
    monkeypatch.setattr(runtime_main, "_make_conversation", lambda ex, *_args: ex)
    dataset, train_ds, eval_ds = runtime_main._prepare_datasets(
        SimpleNamespace(
            dataset_prompt_column="prompt",
            dataset_solution_column="solution",
            dataset_train_split="train",
            dataset_test_split="test",
        ),
        SimpleNamespace(do_eval=True, seed=0, system_prompt=None),
        tok,
    )
    assert tok.added is True
    assert eval_ds is not None
    assert isinstance(train_ds, runtime_main.ReplayMixDataset)


def test_build_trainer_calls_callbacks_and_trainer(monkeypatch):
    called = {}
    monkeypatch.setattr(runtime_main, "ReplayBuffer", lambda *a, **k: "rb")
    monkeypatch.setattr(runtime_main, "get_callbacks", lambda *a, **k: ["cb"])
    monkeypatch.setattr(runtime_main, "LossLoggingCallback", lambda out_dir: f"ll-{out_dir}")

    class FakeTrainer:
        def __init__(self, **kwargs):
            called["kwargs"] = kwargs
            self.data_collator = SimpleNamespace(sort_by_length=True)

    monkeypatch.setattr(runtime_main, "GRPOTrainerReplay", FakeTrainer)
    monkeypatch.setattr(runtime_main, "get_peft_config", lambda model_args: "peft")

    ctx = SimpleNamespace(
        training_args=SimpleNamespace(output_dir="od"),
        model_args="ma",
        model="m",
        tokenizer="tok",
        reward_funcs="rf",
        easy_pool=None,
        train_ds="train",
        eval_ds="eval",
    )
    trainer = runtime_main._build_trainer(ctx)
    assert called["kwargs"]["reward_funcs"] == "rf"
    assert trainer.data_collator.sort_by_length is False


def test_init_model_and_tokenizer_adds_pad_token(monkeypatch):
    class Tok:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token = None
            self.eos_token_id = None
            self.added = False
            self.padding_side = None

        def add_special_tokens(self, tokens):
            self.added = True
            self.pad_token = tokens["pad_token"]
            self.pad_token_id = 5

        def __len__(self):
            return 10

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None, return_dict_in_generate=None, output_scores=None)
            self.generation_config = SimpleNamespace(
                sort_inputs=None, return_dict_in_generate=None, output_scores=None, do_sample=None
            )

        def resize_token_embeddings(self, _n):
            self.resized = True

    tok = Tok()
    model = Model()
    monkeypatch.setattr(runtime_main, "get_tokenizer", lambda *_args, **_kwargs: tok)
    monkeypatch.setattr(runtime_main, "get_model", lambda *_args, **_kwargs: model)

    m, t = runtime_main._init_model_and_tokenizer(SimpleNamespace(), SimpleNamespace())
    assert t.added is True and t.pad_token_id == 5
    assert getattr(m, "resized", False) is True


def test_main_runs_with_eval_and_push(monkeypatch, tmp_path):
    # Avoid real logging / filesystem checks
    monkeypatch.setattr(runtime_main, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime_main,
        "os",
        SimpleNamespace(
            path=runtime_main.os.path, environ={}, getcwd=os.getcwd, sep=os.sep, dirsep=os.sep, isdir=lambda p: True
        ),
    )
    monkeypatch.setattr(runtime_main, "get_last_checkpoint", lambda path: "chk")

    trainer_calls = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.accelerator = SimpleNamespace(is_main_process=True)
            self.model = SimpleNamespace(config=SimpleNamespace(use_cache=False, save_pretrained=lambda path: None))
            self.data_collator = SimpleNamespace(sort_by_length=False)

        def train(self, resume_from_checkpoint=None):
            trainer_calls["train_ckpt"] = resume_from_checkpoint
            return SimpleNamespace(metrics={"m": 1})

        def log_metrics(self, split, metrics):
            trainer_calls.setdefault("log", []).append((split, metrics))

        def save_metrics(self, split, metrics):
            trainer_calls.setdefault("save", []).append((split, metrics))

        def save_state(self):
            trainer_calls["saved_state"] = True

        def save_model(self, out_dir):
            trainer_calls["save_model"] = out_dir

        def create_model_card(self, **kwargs):
            trainer_calls["card"] = kwargs

        def evaluate(self):
            trainer_calls["eval"] = True
            return {"e": 2}

        def push_to_hub(self, **kwargs):
            trainer_calls["push"] = kwargs

    monkeypatch.setattr(runtime_main, "_init_model_and_tokenizer", lambda *a, **k: ("model", "tok"))
    monkeypatch.setattr(runtime_main, "_maybe_load_easy_pool", lambda *a, **k: "easy")
    monkeypatch.setattr(runtime_main, "_build_reward_funcs", lambda *a, **k: "rf")
    monkeypatch.setattr(runtime_main, "_prepare_datasets", lambda *a, **k: ("ds", "train", "eval"))
    monkeypatch.setattr(runtime_main, "_build_trainer", lambda ctx: FakeTrainer())

    script_args = SimpleNamespace(dataset_name="ds")
    training_args = SimpleNamespace(
        seed=0,
        local_rank=0,
        device="cpu",
        n_gpu=0,
        bf16=False,
        resume_from_checkpoint=None,
        output_dir=str(tmp_path),
        do_eval=True,
        push_to_hub=True,
    )
    model_args = SimpleNamespace()

    runtime_main.main(script_args, training_args, model_args)
    assert trainer_calls["train_ckpt"] == "chk"
    assert ("eval", {"e": 2}) in trainer_calls["log"]
    assert trainer_calls["push"]["dataset_name"] == "ds"
