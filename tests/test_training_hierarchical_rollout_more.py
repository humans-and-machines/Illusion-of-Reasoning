import types

import pytest


hr = pytest.importorskip("src.training.utils.hierarchical_rollout")
torch = pytest.importorskip("torch")

# Skip when running with the lightweight torch stub that lacks tensor ops or device factory.
if not hasattr(torch, "tensor") or isinstance(torch, types.SimpleNamespace):
    pytest.skip("torch stub lacks required tensor/device attributes", allow_module_level=True)
try:
    _ = torch.tensor([]).view
    _ = torch.device("cpu")
except Exception:
    pytest.skip("torch stub lacks view or device constructor", allow_module_level=True)


class _FakeProcessingClass:
    def __init__(self, pad_token="<pad>", pad_id=0):
        self.pad_token = pad_token
        self.pad_token_id = pad_id
        self.eos_token_id = 9

    def __call__(
        self, text, return_tensors=None, padding=None, truncation=None, padding_side=None, add_special_tokens=None
    ):
        # Build incremental ids to detect truncation.
        ids = [list(range(1, len(text[0].split()) + 1))]
        attn = [[1] * len(ids[0])]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def batch_decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        outs = []
        for row in ids:
            if isinstance(row, torch.Tensor):
                row = row.tolist()
            outs.append(" ".join(str(tok) for tok in row))
        return outs


def test_prepare_prompts_truncates_and_strips_pad():
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.processing_class = _FakeProcessingClass()
    inst.max_prompt_length = 1
    inputs = [{"prompt": "foo bar"}]
    prompts, prompts_text, prompt_ids, prompt_mask = inst._prepare_prompts(inputs, device=torch.device("cpu"))
    assert prompts == ["foo bar"]
    assert prompt_ids.shape[1] == 1
    # After truncation, prompts_text should be regenerated without pad tokens.
    assert all("<pad>" not in p for p in prompts_text)


def test_run_generation_backends_routes(monkeypatch):
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.use_vllm = False
    inst.use_transformers_paged = False
    inst.rollout_fn = object()
    called = {}
    monkeypatch.setattr(
        inst,
        "_generate_two_stage",
        lambda prompt_ids: called.setdefault("two_stage", True) or (torch.tensor([[1]]), torch.tensor([[2]])),
    )
    batch = hr.GenerationBatch([], [], torch.tensor([[0]]), torch.tensor([[1]]), device=torch.device("cpu"))
    inst._run_generation_backends(batch)
    assert called.get("two_stage")

    # vLLM path
    inst.use_vllm = True
    inst.rollout_fn = None
    monkeypatch.setattr(
        inst,
        "_generate_with_vllm",
        lambda b: called.setdefault("vllm", True) or (torch.tensor([[1]]), torch.tensor([[2]])),
    )
    inst._run_generation_backends(batch)
    assert called.get("vllm")

    # paged path
    inst.use_vllm = False
    inst.use_transformers_paged = True
    monkeypatch.setattr(
        inst, "_generate_with_paged", lambda b: called.setdefault("paged", True) or torch.tensor([[3]])
    )
    inst._run_generation_backends(batch)
    assert called.get("paged")


def test_normalize_rewards_scaling_flags():
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.num_generations = 2
    inst.scale_rewards = True
    rewards = torch.tensor([1.0, 3.0, 1.0, 3.0])
    stats = inst._normalize_rewards(rewards)
    # When variance zero, is_std_zero flagged.
    assert stats.is_std_zero.tolist() == [True, True]

    inst.scale_rewards = False
    stats_no_scale = inst._normalize_rewards(rewards)
    assert stats_no_scale.advantages.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_update_reward_metrics_populates(monkeypatch):
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.model = types.SimpleNamespace(training=True)
    inst.accelerator = types.SimpleNamespace(
        gather=lambda x: x,
        process_index=0,
    )
    inst.state = types.SimpleNamespace(num_input_tokens_seen=0)
    inst._metrics = {
        "train": {
            "num_tokens": [],
            "completions/mean_length": [],
            "completions/min_length": [],
            "completions/max_length": [],
            "completions/mean_terminated_length": [],
            "completions/min_terminated_length": [],
            "completions/max_terminated_length": [],
            "completions/clipped_ratio": [],
            "reward": [],
            "reward_std": [],
            "frac_reward_zero_std": [],
        }
    }

    completion_lengths = torch.tensor([2, 4])
    is_eos = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    stats = hr.RewardStatistics(
        advantages=torch.tensor([0.0, 0.0]),
        mean_grouped_rewards=torch.tensor([1.0, 2.0]),
        std_grouped_rewards=torch.tensor([0.0, 0.0]),
        is_std_zero=torch.tensor([True, True]),
    )
    batch = hr.GenerationBatch(
        prompts=[0, 1],
        prompts_text=["p0", "p1"],
        prompt_ids=None,
        prompt_mask=torch.tensor([[1, 1], [1, 1]]),
        device=torch.device("cpu"),
    )
    state = {"completion_lengths": completion_lengths, "is_eos": is_eos}
    inst._update_reward_metrics(batch, stats, state)
    assert inst._metrics["train"]["num_tokens"]
    assert inst._metrics["train"]["reward"] == [1.5]
