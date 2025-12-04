import importlib
import importlib.util
import sys
import types


try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - allow running without torch installed
    torch = types.SimpleNamespace(
        tensor=lambda data=None, **_k: data,
        Tensor=type("FakeTensor", (), {})(),
        device=type("device", (), {})(),
    )
    torch.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    sys.modules["torch"] = torch

# Provide lightweight stubs for datasets/transformers logging to satisfy env imports.
stub_logging = types.SimpleNamespace(
    set_verbosity=lambda *_: None,
    enable_default_handler=lambda: None,
    enable_explicit_format=lambda: None,
)
sys.modules.setdefault(
    "datasets",
    types.SimpleNamespace(
        __spec__=importlib.util.spec_from_loader("datasets", loader=None),
        utils=types.SimpleNamespace(logging=stub_logging),
    ),
)
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        __spec__=importlib.util.spec_from_loader("transformers", loader=None),
        utils=types.SimpleNamespace(logging=stub_logging),
    ),
)

# Stub runtime.env module to provide expected attributes without heavy imports.
stub_env = types.SimpleNamespace(
    datasets=sys.modules["datasets"],
    transformers=sys.modules["transformers"],
    torch=torch,
    dist=None,
    GRPOTrainer=type("StubGRPOTrainer", (), {}),
    wandb=None,
    TrainerCallback=type("StubTrainerCallback", (), {}),
    get_last_checkpoint=lambda *a, **k: None,
    get_peft_config=lambda *a, **k: None,
    set_seed=lambda *a, **k: None,
    __all__=[
        "datasets",
        "transformers",
        "torch",
        "dist",
        "GRPOTrainer",
        "wandb",
        "TrainerCallback",
        "get_last_checkpoint",
        "get_peft_config",
        "set_seed",
    ],
)
sys.modules["src.training.runtime.env"] = stub_env

# Stub heavy downstream modules to avoid torch/transformers imports during collection.
sys.modules["src.training.grpo_trainer_replay_impl"] = types.SimpleNamespace(
    GRPOTrainerReplay=type("StubGRPOTrainerReplay", (), {})(),
)
sys.modules["src.training.grpo_trainer_replay_support"] = types.SimpleNamespace(
    LossLoggingCallback=lambda out_dir: f"ll-{out_dir}"
)
sys.modules["src.training.grpo_dataset"] = types.SimpleNamespace(
    _make_conversation=lambda ex, *a, **k: ex,
    _load_easy_pool=lambda *a, **k: "easy",
)
sys.modules["src.training.grpo_rewards_router"] = types.SimpleNamespace(
    _wrap_reward_for_nested=lambda fn: fn,
)

# Import the module under test after stubbing dependencies.
rm = importlib.import_module("src.training.runtime.main")


def test_configure_logging_sets_levels(monkeypatch):
    # Capture level changes on dataset/transformer logging utilities.
    called = {"datasets": [], "transformers": []}
    monkeypatch.setattr(
        rm,
        "datasets",
        types.SimpleNamespace(
            utils=types.SimpleNamespace(
                logging=types.SimpleNamespace(set_verbosity=lambda lvl: called["datasets"].append(lvl))
            )
        ),
    )
    monkeypatch.setattr(
        rm,
        "transformers",
        types.SimpleNamespace(
            utils=types.SimpleNamespace(
                logging=types.SimpleNamespace(
                    set_verbosity=lambda lvl: called["transformers"].append(lvl),
                    enable_default_handler=lambda: called.setdefault("def_handler", True),
                    enable_explicit_format=lambda: called.setdefault("explicit", True),
                )
            )
        ),
    )
    args = types.SimpleNamespace(get_process_log_level=lambda: 10)
    rm._configure_logging(args)
    assert called["datasets"] == [10] and called["transformers"] == [10]
    assert called["def_handler"] and called["explicit"]


def test_init_model_and_tokenizer_adds_pad(monkeypatch):
    class Tok:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token = "[EOS]"
            self.eos_token_id = 1
            self.pad_token = None
            self.padding_side = None

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "pad_token" and getattr(self, "pad_token_id", None) is None:
                super().__setattr__("pad_token_id", getattr(self, "eos_token_id", None))

        def add_special_tokens(self, mapping):
            self.pad_token = mapping.get("pad_token")
            return mapping

    tok = Tok()
    gen_cfg = types.SimpleNamespace(sort_inputs=None, return_dict_in_generate=None, output_scores=None, do_sample=None)
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(pad_token_id=None, return_dict_in_generate=None, output_scores=None),
        generation_config=gen_cfg,
        resize_token_embeddings=lambda n: None,
    )
    monkeypatch.setattr(rm, "get_tokenizer", lambda *a, **k: tok)
    monkeypatch.setattr(rm, "get_model", lambda *a, **k: model)
    m, t = rm._init_model_and_tokenizer(types.SimpleNamespace(), types.SimpleNamespace())
    assert t.pad_token_id == 1
    assert m.config.pad_token_id == 1
    assert t.padding_side == "left"
    assert m.generation_config.do_sample is True


def test_maybe_load_easy_pool(monkeypatch):
    called = {}
    monkeypatch.setenv("EASY_DATASET_NAME", "")
    assert rm._maybe_load_easy_pool(types.SimpleNamespace(easy_dataset_name=None), None, None) is None
    monkeypatch.setenv("EASY_DATASET", "on")
    monkeypatch.setattr(rm, "_load_easy_pool", lambda *a, **k: called.setdefault("loaded", "easy"))
    assert rm._maybe_load_easy_pool(types.SimpleNamespace(easy_dataset_name=None), None, None) == "easy"
    assert called["loaded"]


def test_main_happy_path(monkeypatch, tmp_path):
    # Stub out all heavy dependencies to exercise control flow.
    monkeypatch.setattr(rm, "set_seed", lambda seed: None)
    monkeypatch.setattr(rm, "_configure_logging", lambda args: None)
    monkeypatch.setattr(rm, "_init_model_and_tokenizer", lambda *a, **k: ("model", "tok"))
    monkeypatch.setattr(rm, "_maybe_load_easy_pool", lambda *a, **k: "easy_pool")
    monkeypatch.setattr(rm, "_build_reward_funcs", lambda *a, **k: "rewards")
    monkeypatch.setattr(rm, "_prepare_datasets", lambda *a, **k: ("dataset", "train_ds", "eval_ds"))
    monkeypatch.setattr(
        rm,
        "_build_trainer",
        lambda ctx: types.SimpleNamespace(
            accelerator=types.SimpleNamespace(is_main_process=True),
            train=lambda resume_from_checkpoint=None: types.SimpleNamespace(metrics={"t": 1}),
            log_metrics=lambda *a, **k: None,
            save_metrics=lambda *a, **k: None,
            save_state=lambda: None,
            save_model=lambda out: None,
            create_model_card=lambda dataset_name, tags: None,
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(use_cache=False, save_pretrained=lambda out: None)
            ),
            evaluate=lambda: {"e": 2},
            push_to_hub=lambda dataset_name, tags: None,
            data_collator=types.SimpleNamespace(sort_by_length=True),
        ),
    )
    monkeypatch.setattr(rm, "get_last_checkpoint", lambda out: None)

    class Args:
        def __init__(self):
            self.output_dir = str(tmp_path)
            self.resume_from_checkpoint = None
            self.local_rank = -1
            self.device = "cpu"
            self.n_gpu = 0
            self.bf16 = False
            self.do_eval = True
            self.push_to_hub = False
            self.seed = 0
            self.steps_per_generation = 0
            self.num_iterations = 0
            self.return_reward = False

    script_args = types.SimpleNamespace(dataset_name="ds")
    training_args = Args()
    model_args = types.SimpleNamespace()

    rm.main(script_args, training_args, model_args)
    # Ensure flags were set inside main
    assert training_args.return_reward is True
    assert training_args.steps_per_generation == 8
    assert training_args.num_iterations == 5
