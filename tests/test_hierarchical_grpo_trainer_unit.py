#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import types
from collections import defaultdict
from types import SimpleNamespace

import src.training.utils.hierarchical_grpo_generation as hgg
import src.training.utils.hierarchical_grpo_trainer as hgt


class ProcessingStub:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **_kwargs):
        rows = []
        for idx, _ in enumerate(text, start=1):
            rows.append([0, idx, idx + 1, idx + 2])
        input_ids = hgt.torch.tensor(rows, dtype=hgt.torch.long)
        attn = hgt.torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}

    def batch_decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        decoded = []
        for row in ids:
            decoded.append("".join(self.pad_token if tok == self.pad_token_id else f"t{tok}" for tok in row))
        return decoded


def _stub_init(self, *args, callbacks=None, **kwargs):
    self.args = SimpleNamespace(
        bf16=False,
        fp16=False,
        ds3_gather_for_generation=False,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )
    self.accelerator = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        process_index=0,
        is_main_process=True,
        gather=lambda x: x,
        unwrap_model=lambda model: SimpleNamespace(
            disable_adapter=lambda: types.SimpleNamespace(__enter__=lambda *_: None, __exit__=lambda *_: False)
        ),
    )
    self.processing_class = ProcessingStub()
    self.max_prompt_length = None
    self.use_vllm = False
    self.use_transformers_paged = False
    self.rollout_fn = None
    self.num_generations = 1
    self.max_completion_length = 3
    self.generation_config = None
    self.repetition_penalty = 1.0
    self.temperature = 0.1
    self.top_p = 0.9
    self.top_k = None
    self.min_p = None
    self.guided_decoding_regex = None
    self.reward_weights = hgt.torch.tensor([1.0])
    self.reward_func_names = ["r"]
    self.num_iterations = 1
    self.beta = 0.0
    self.scale_rewards = True
    self.ref_model = None
    self.is_fsdp_enabled = False
    self.model_wrapped = SimpleNamespace(config=SimpleNamespace(), training=True)
    self.model = self.model_wrapped
    self._textual_logs = {
        "prompt": [],
        "completion": [],
        "rewards": {name: [] for name in self.reward_func_names},
        "advantages": [],
    }
    self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    self.state = SimpleNamespace(num_input_tokens_seen=0)


def make_trainer(monkeypatch):
    monkeypatch.setattr(hgt, "TrainerCallback", type("TrainerCallback", (), {}), raising=False)
    monkeypatch.setattr(
        hgt.HierarchicalGRPOTrainer, "add_callback", lambda self, cb: setattr(self, "_added_cb", cb), raising=False
    )
    monkeypatch.setattr(hgt, "PaddingStrategy", SimpleNamespace(LONGEST="longest"), raising=False)
    monkeypatch.setattr(hgg, "PaddingStrategy", hgt.PaddingStrategy, raising=False)

    def patched_init(
        self,
        *args,
        rollout_fn=None,
        tokenizer=None,
        return_reason=True,
        callbacks=None,
        **kwargs,
    ):
        self.rollout_fn = rollout_fn
        self.tokenizer = tokenizer
        self.return_reason = return_reason
        callback_instances, callback_factories = [], []
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, hgt.TrainerCallback):
                    callback_instances.append(callback)
                else:
                    callback_factories.append(callback)
        _stub_init(self, *args, **kwargs)
        self.mask_truncated_completions = True
        for cb in callback_instances:
            self.add_callback(cb)
        for factory in callback_factories:
            self.add_callback(factory(self))

    monkeypatch.setattr(hgt.HierarchicalGRPOTrainer, "__init__", patched_init, raising=False)
    monkeypatch.setattr(hgt, "maybe_apply_chat_template", lambda example, proc: {"prompt": example["prompt"]})

    def _simple_pad(tensors, padding_value=0, padding_side="left"):
        max_len = max(len(t.data) if hasattr(t, "data") else len(t) for t in tensors) if tensors else 0
        padded = []
        for t in tensors:
            data = t.data if hasattr(t, "data") else t
            pad_len = max_len - len(data)
            if padding_side == "left":
                padded.append([padding_value] * pad_len + list(data))
            else:
                padded.append(list(data) + [padding_value] * pad_len)
        return hgt.torch.tensor(padded)

    monkeypatch.setattr(hgt, "pad", _simple_pad)
    monkeypatch.setattr(hgt, "broadcast_object_list", lambda obj, from_process=0: obj)
    monkeypatch.setattr(hgt, "gather_object", lambda obj: obj)
    monkeypatch.setattr(
        hgt,
        "unwrap_model_for_generation",
        lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda *_: SimpleNamespace(generate_batch=lambda *args, **kwargs: {}), __exit__=lambda *_: False
        ),
    )
    monkeypatch.setattr(hgg, "unwrap_model_for_generation", hgt.unwrap_model_for_generation, raising=False)
    monkeypatch.setattr(
        hgt,
        "profiling_context",
        lambda *a, **k: types.SimpleNamespace(__enter__=lambda *_: None, __exit__=lambda *_: False),
    )
    stub_tensor_cls = hgt.torch.tensor([0]).__class__
    monkeypatch.setattr(stub_tensor_cls, "to", lambda self, device=None: self, raising=False)
    monkeypatch.setattr(stub_tensor_cls, "min", lambda self: hgt.torch.tensor(self.data.min()), raising=False)
    monkeypatch.setattr(stub_tensor_cls, "max", lambda self: hgt.torch.tensor(self.data.max()), raising=False)
    return hgt.HierarchicalGRPOTrainer()


def test_init_splits_callbacks_and_reset(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer_cb = hgt.TrainerCallback()
    factory_called = {}

    def factory(tr):
        factory_called["trainer"] = tr
        return "cb"

    trainer.__init__(callbacks=[trainer_cb, factory])
    assert trainer.rollout_fn is None
    assert getattr(trainer, "_added_cb") == "cb"
    trainer._textual_logs["prompt"].append("p")
    trainer.reset_textual_logs()
    assert trainer.textual_logs == {}


def test_generate_and_prepare_prompts_truncation(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.max_prompt_length = 2
    inputs = [{"prompt": "p1"}, {"prompt": "p2"}]
    prompts, prompts_text, prompt_ids, prompt_mask = trainer._prepare_prompts(inputs, trainer.accelerator.device)
    assert prompts == ["p1", "p2"]
    assert prompt_ids.shape[1] == 2 and prompt_mask.shape[1] == 2
    assert all(txt.startswith("t") for txt in prompts_text)

    monkeypatch.setattr(trainer, "_prepare_prompts", lambda inputs, device: ("pr", "pt", "pid", "pmask"))
    monkeypatch.setattr(trainer, "_run_generation_backends", lambda batch: "comp")
    monkeypatch.setattr(trainer, "_postprocess_and_score", lambda inputs, batch, comp: {"ok": True, "batch": batch})
    out = trainer._generate_and_score_completions(inputs)
    assert out["ok"] is True


def test_run_generation_backends_branches(monkeypatch):
    trainer = make_trainer(monkeypatch)
    batch = hgt.GenerationBatch(
        prompts=[1],
        prompts_text=["a"],
        prompt_ids=hgt.torch.tensor([[1, 2]]),
        prompt_mask=hgt.torch.tensor([[1, 1]]),
        device=trainer.accelerator.device,
    )

    trainer.rollout_fn = lambda ids, max_new_tokens=None: ("full", hgt.torch.tensor([[1, 2, 3]]))
    assert trainer._run_generation_backends(batch)

    trainer.rollout_fn = None
    trainer.use_vllm = True
    monkeypatch.setattr(trainer, "_generate_with_vllm", lambda b: ("full", hgt.torch.tensor([[9]])))
    assert trainer._run_generation_backends(batch).data.tolist() == [[9]]

    trainer.use_vllm = False
    trainer.use_transformers_paged = True
    monkeypatch.setattr(trainer, "_generate_with_paged", lambda b: hgt.torch.tensor([[7]]))
    assert trainer._run_generation_backends(batch).data.tolist() == [[7]]

    trainer.use_transformers_paged = False
    monkeypatch.setattr(trainer, "_generate_with_hf", lambda ids, mask: ("full", hgt.torch.tensor([[5]])))
    assert trainer._run_generation_backends(batch).data.tolist() == [[5]]


def test_generate_with_vllm_handles_empty(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.accelerator.is_main_process = True
    trainer.accelerator.process_index = 0
    trainer.vllm_client = SimpleNamespace(generate=lambda **kwargs: [])

    # stub torch.zeros to accept dtype/device keywords from test torch stub
    def fake_zeros(*shape, **kwargs):
        dtype = kwargs.get("dtype", None)
        return hgt.torch.tensor([[] for _ in range(shape[0])], dtype=dtype)

    monkeypatch.setattr(hgt.torch, "zeros", fake_zeros)
    batch = hgt.GenerationBatch(
        prompts=["p1"],
        prompts_text=["p1"],
        prompt_ids=hgt.torch.tensor([[1, 2]]),
        prompt_mask=hgt.torch.tensor([[1, 1]]),
        device=trainer.accelerator.device,
    )

    class Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(hgt, "profiling_context", lambda *a, **k: Ctx())
    prompt_completion_ids, completion_padded = trainer._generate_with_vllm(batch)
    assert completion_padded.shape[1] == 0
    assert prompt_completion_ids.shape[1] == batch.prompt_ids.shape[1]


def test_generate_with_paged_sets_attn_and_dtype(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.args.bf16 = True
    trainer.args.fp16 = False
    monkeypatch.setattr(hgt, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(hgg, "is_flash_attn_2_available", lambda: True, raising=False)

    class Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(hgt, "profiling_context", lambda *a, **k: Ctx())
    # Provide missing torch dtype attributes for stubbed torch
    hgt.torch.bfloat16 = getattr(hgt.torch, "bfloat16", "bf16")
    hgt.torch.float16 = getattr(hgt.torch, "float16", "f16")

    class FakeOutput:
        def __init__(self):
            self.generated_tokens = [3, 4]

    def fake_generate_batch(input_ids, generation_config=None, progress_bar=False):
        return {"a": FakeOutput()}

    def fake_unwrap(model_wrapped, accelerator, gather_deepspeed3_params=False):
        class UCtx:
            def __enter__(self_inner):
                return SimpleNamespace(generate_batch=fake_generate_batch, to=lambda *a, **k: None)

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return UCtx()

    monkeypatch.setattr(hgt, "unwrap_model_for_generation", fake_unwrap)
    monkeypatch.setattr(hgg, "unwrap_model_for_generation", fake_unwrap, raising=False)

    class Proc:
        pad_token_id = 0

        def __call__(self, text):
            return SimpleNamespace(input_ids=[[1, 2]])

        def batch_decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return ["tok"]

    trainer.processing_class = Proc()
    batch = hgt.GenerationBatch(
        prompts=["p1"],
        prompts_text=["p1"],
        prompt_ids=hgt.torch.tensor([[1, 2]]),
        prompt_mask=hgt.torch.tensor([[1, 1]]),
        device=trainer.accelerator.device,
    )

    # Override pad to return an object with int() to avoid bool/int stubs.
    class Dummy:
        def __init__(self, data):
            self.data = data

        def __ne__(self, other):
            return self

        def int(self):
            return self

        def size(self, dim=None):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        @property
        def shape(self):
            inner = self.data[0] if self.data else []
            return (len(self.data), len(inner))

    monkeypatch.setattr(hgt, "pad", lambda tensors, padding_value=0, padding_side="left": Dummy(tensors))
    monkeypatch.setattr(
        hgg, "pad", lambda tensors, padding_value=0, padding_side="left": Dummy(tensors), raising=False
    )
    out = trainer._generate_with_paged(batch)
    assert out.shape[1] >= 1
    assert trainer.model_wrapped.config._attn_implementation == "paged_attention"


def test_build_completion_mask_with_and_without_truncation(monkeypatch):
    trainer = make_trainer(monkeypatch)
    completion_ids = hgt.torch.tensor([[5, 2, 7], [8, 9, 10]])
    comp_mask, comp_list, lengths, is_eos = trainer._build_completion_mask(completion_ids, trainer.accelerator.device)
    assert comp_list[0] == [5, 2]
    assert lengths.tolist() == [2, 0]

    trainer.mask_truncated_completions = False
    comp_mask2, comp_list2, lengths2, _ = trainer._build_completion_mask(completion_ids, trainer.accelerator.device)
    assert comp_list2[1] == [8, 9, 10]
    assert lengths2.tolist() == [2, 3]


def test_normalize_rewards_and_slice(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.num_generations = 1
    rewards = hgt.torch.tensor([1.0, 1.0])
    stats = trainer._normalize_rewards(rewards)
    assert stats.is_std_zero.all()
    adv, all_adv = trainer._slice_advantages_for_process(
        stats,
        hgt.GenerationBatch(
            prompts=[1, 2], prompts_text=[], prompt_ids=None, prompt_mask=None, device=trainer.accelerator.device
        ),
    )
    assert adv.shape[0] == 2 and all_adv.shape[0] == 2


def test_update_reward_metrics(monkeypatch):
    trainer = make_trainer(monkeypatch)
    batch = hgt.GenerationBatch(
        prompts=[1, 2],
        prompts_text=[],
        prompt_ids=hgt.torch.tensor([[1, 1], [1, 1]]),
        prompt_mask=hgt.torch.tensor([[1, 1], [1, 1]]),
        device=trainer.accelerator.device,
    )
    stats = hgt.RewardStatistics(
        advantages=hgt.torch.tensor([0.1, 0.2]),
        mean_grouped_rewards=hgt.torch.tensor([1.0, 2.0]),
        std_grouped_rewards=hgt.torch.tensor([0.0, 0.0]),
        is_std_zero=hgt.torch.tensor([1.0, 1.0]),
    )
    state = {
        "completion_lengths": hgt.torch.tensor([2, 3]),
        "is_eos": hgt.torch.tensor([[True, False], [False, False]]),
    }
    trainer._update_reward_metrics(batch, stats, state)
    assert trainer._metrics["train"]["reward"]
    assert trainer.state.num_input_tokens_seen > 0


def test_decode_completions_conversational(monkeypatch):
    trainer = make_trainer(monkeypatch)
    monkeypatch.setattr(hgt, "is_conversational", lambda inp: True, raising=False)
    monkeypatch.setattr(hgg, "is_conversational", lambda inp: True, raising=False)
    monkeypatch.setattr(hgt._deps, "is_conversational", lambda inp: True, raising=False)
    inputs = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "prev"}]]
    prompts = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "prev"}]]
    completion_ids = hgt.torch.tensor([[1, 2]])
    texts, comps = trainer._decode_completions(inputs, prompts, completion_ids)
    assert texts and isinstance(comps[0], list)


def test_compute_logps_ref_model_none(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.beta = 1.0
    trainer.ref_model = None
    trainer.num_iterations = 2
    trainer.args.gradient_accumulation_steps = 1

    def fake_get(self, model, ids, attn, keep, batch_size=None):
        return {"logps": hgt.torch.ones((2, 2))}

    monkeypatch.setattr(hgt.HierarchicalGRPOTrainer, "_get_per_token_logps_and_entropies", fake_get, raising=False)

    class WrapCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    trainer.accelerator.unwrap_model = lambda model: SimpleNamespace(disable_adapter=lambda: WrapCtx())
    batch = hgt.GenerationBatch(
        prompts=[1, 2],
        prompts_text=[],
        prompt_ids=hgt.torch.tensor([[1, 1], [1, 1]]),
        prompt_mask=hgt.torch.tensor([[1, 1], [1, 1]]),
        device=trainer.accelerator.device,
    )
    state = {
        "completion_ids": hgt.torch.tensor([[2, 2], [3, 3]]),
        "completion_mask": hgt.torch.tensor([[1, 1], [1, 1]]),
    }
    old_logps, ref_logps = trainer._compute_logps(batch, state)
    assert old_logps is not None and ref_logps is not None


def test_log_text_and_rewards(monkeypatch):
    trainer = make_trainer(monkeypatch)
    rewards_per_func = hgt.torch.tensor([[1.0]])
    all_adv = hgt.torch.tensor([0.5])
    trainer._log_text_and_rewards(["p"], ["c"], rewards_per_func, all_adv)
    assert trainer._textual_logs["prompt"] == ["p"]
    assert trainer._textual_logs["advantages"] == [0.5]


def test_postprocess_and_score_flow(monkeypatch):
    trainer = make_trainer(monkeypatch)
    batch = hgt.GenerationBatch(
        prompts=["p"],
        prompts_text=["p"],
        prompt_ids=hgt.torch.tensor([[1, 1]]),
        prompt_mask=hgt.torch.tensor([[1, 1]]),
        device=trainer.accelerator.device,
    )
    completion_ids = hgt.torch.tensor([[2, 2]])
    monkeypatch.setattr(trainer, "_compute_logps", lambda batch, state: (None, None))
    monkeypatch.setattr(
        trainer,
        "_compute_rewards",
        lambda inputs, batch, comps, comp_ids: (hgt.torch.tensor([[1.0]]), hgt.torch.tensor([1.0])),
    )
    monkeypatch.setattr(
        trainer,
        "_normalize_rewards_and_compute_advantages",
        lambda state, batch: (hgt.torch.tensor([0.1]), hgt.torch.tensor([0.1])),
    )
    result = trainer._postprocess_and_score([{"prompt": "p"}], batch, completion_ids)
    assert "prompt_ids" in result and "completion_mask" in result
