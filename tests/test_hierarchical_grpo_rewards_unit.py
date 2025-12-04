"""Unit tests for hierarchical GRPO reward helpers."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.training.utils import hierarchical_grpo_rewards as rewards_mod
from src.training.utils.hierarchical_grpo_rewards import HierarchicalRewardMixin, _safe_metric_reduce


torch = rewards_mod.torch


def test_safe_metric_reduce_fallback_on_error():
    def bad_reducer(_tensor):
        raise TypeError("boom")

    assert _safe_metric_reduce(bad_reducer, object(), fallback=7.5) == 7.5


def test_public_wrappers_delegate_to_internal_methods():
    class WrapperTester(HierarchicalRewardMixin):
        def __init__(self):
            self.compute_called = False
            self.norm_called = False

        def _compute_rewards(self, *args, **kwargs):
            self.compute_called = True
            return ("per_func", "agg")

        def _normalize_rewards_and_compute_advantages(self, state, batch):
            self.norm_called = True
            return ("adv", "all_adv")

    mixin = WrapperTester()
    assert mixin.compute_rewards([], None, None, []) == ("per_func", "agg")
    assert mixin.compute_called
    assert mixin.normalize_rewards_and_compute_advantages({}, None) == ("adv", "all_adv")
    assert mixin.norm_called


def test_compute_logps_handles_reference_and_eval_batch_size():
    class DummyLogpMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.args = SimpleNamespace(
                steps_per_generation=1,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=3,
                per_device_eval_batch_size=4,
            )
            self.num_iterations = 2
            self.beta = 0.5
            self.model = SimpleNamespace(training=False)
            self.ref_model = SimpleNamespace()
            self.accelerator = SimpleNamespace(
                unwrap_model=lambda model: SimpleNamespace(disable_adapter=nullcontext),
            )
            self.calls = []

        def _get_per_token_logps_and_entropies(
            self,
            model,
            ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
        ):
            self.calls.append((model is self.model, batch_size))
            base = getattr(torch, "ones", None)
            if base:
                logps = base((ids.shape[0], logits_to_keep), device=getattr(ids, "device", None))
            else:
                logps = rewards_mod.torch.full((ids.shape[0], logits_to_keep), 1.0)
            return {"logps": logps}

    mixin = DummyLogpMixin()
    batch = SimpleNamespace(
        prompt_mask=torch.zeros((1, 1)),
        prompt_ids=torch.zeros((1, 1)),
    )
    state = {
        "completion_ids": torch.zeros((1, 2), dtype=getattr(torch, "long", None)),
        "completion_mask": torch.zeros((1, 2)),
    }

    old_logps, ref_logps = mixin._compute_logps(batch, state)

    assert old_logps is not None and ref_logps is not None
    assert mixin.calls[0][1] == mixin.args.per_device_eval_batch_size
    assert old_logps.shape == ref_logps.shape


def test_compute_logps_skips_when_not_needed():
    class SkipLogpMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.args = SimpleNamespace(
                steps_per_generation=1,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
            )
            self.num_iterations = 2
            self.beta = 0.0
            self.model = SimpleNamespace(training=True)
            self.ref_model = None
            self.accelerator = SimpleNamespace(
                unwrap_model=lambda model: SimpleNamespace(disable_adapter=nullcontext),
            )

        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            raise AssertionError("Should not be called when gradients are aligned.")

    mixin = SkipLogpMixin()
    batch = SimpleNamespace(
        prompt_mask=torch.zeros((1, 1)),
        prompt_ids=torch.zeros((1, 1)),
    )
    state = {
        "completion_ids": torch.zeros((1, 1)),
        "completion_mask": torch.zeros((1, 1)),
    }

    old_logps, ref_logps = mixin._compute_logps(batch, state)

    assert old_logps is None and ref_logps is None


def test_compute_rewards_applies_weights():
    class RewardMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.reward_weights = torch.tensor([0.5, 1.5])

        def _calculate_rewards(self, inputs, prompts, completions, comp_ids_list):
            _ = (inputs, prompts, completions, comp_ids_list)
            return torch.tensor([[1.0, 2.0]])

    mixin = RewardMixin()
    batch = SimpleNamespace(prompts=["p0"], device="cpu")

    per_func, rewards = mixin.compute_rewards([], batch, [], [[]])

    assert per_func.shape == (1, 2)
    assert rewards.shape == (1,)
    assert rewards.item() == pytest.approx(3.5)


def test_normalize_rewards_invokes_std_fallbacks():
    class FakeTensor:
        def __init__(self, data):
            self.data = torch.tensor(data)

        def mean(self, dim=None):
            return self.data.mean(dim=dim)

        def std(self, *args, **kwargs):
            if kwargs.get("unbiased", None) is False:
                raise TypeError("unsupported")
            return self.data.std(dim=kwargs.get("dim", None))

        def repeat_interleave(self, repeats, dim=0):
            return self.data.repeat_interleave(repeats, dim=dim)

        def __sub__(self, other):
            return self.data - other

    class StdFallbackMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.num_generations = 2
            self.scale_rewards = False

        def _build_reward_views(self, rewards, flat_rewards, prompt_count):
            _ = (rewards, flat_rewards, prompt_count)
            fake = FakeTensor([[1.0, 1.0], [2.0, 2.0]])
            return fake, fake

    mixin = StdFallbackMixin()
    stats = mixin._normalize_rewards(torch.tensor([1.0, 1.0, 2.0, 2.0]))

    assert stats.mean_grouped_rewards.shape[0] == 2
    assert stats.std_grouped_rewards.shape[0] == 2


def test_normalize_rewards_wrapper_updates_metrics():
    class NormalizeMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.num_generations = 2
            self.scale_rewards = False
            self.accelerator = SimpleNamespace(process_index=0)
            self._update_called = False

        def _update_reward_metrics(self, batch, stats, state):
            _ = (batch, stats, state)
            self._update_called = True

    mixin = NormalizeMixin()
    state = {"rewards": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    batch = SimpleNamespace(prompts=[0, 1])

    advantages, all_advantages = mixin._normalize_rewards_and_compute_advantages(state, batch)

    assert mixin._update_called
    assert advantages.shape[0] == len(batch.prompts)
    assert all_advantages.shape[0] == state["rewards"].shape[0]


def test_flatten_rewards_fallbacks_to_iterable():
    class NonListable:
        def tolist(self):
            raise ValueError("no tolist")

        def __iter__(self):
            return iter([1, 2, 3])

    class MinimalMixin(HierarchicalRewardMixin):
        pass

    mixin = MinimalMixin()
    assert mixin._flatten_rewards(NonListable()) == [1, 2, 3]


def test_build_reward_views_uses_fallback(monkeypatch):
    class ViewMixin(HierarchicalRewardMixin):
        def __init__(self):
            self.num_generations = 2

    class FakeRewards:
        def __init__(self, tensor):
            self.tensor = tensor

        def reshape(self, *shape):
            reshape_fn = getattr(self.tensor, "reshape", None)
            if reshape_fn is not None:
                return reshape_fn(*shape)
            return self.tensor.view(*shape)

    mixin = ViewMixin()
    fake_rewards = FakeRewards(torch.arange(4.0))

    def boom_tensor(*args, **kwargs):
        raise TypeError("cannot tensorize")

    monkeypatch.setattr(rewards_mod.torch, "tensor", boom_tensor)

    prompt_major, generation_major = mixin._build_reward_views(fake_rewards, [object()] * 4, 2)

    assert prompt_major.shape == (2, mixin.num_generations)
    assert generation_major.shape == prompt_major.shape


def test_bool_list_handles_scalar_like_objects():
    class NoList:
        def tolist(self):
            raise ValueError("no list")

    class MinimalMixin(HierarchicalRewardMixin):
        pass

    mixin = MinimalMixin()
    assert mixin._bool_list(NoList()) == [True]


def test_scale_advantages_handles_missing_any():
    class MinimalMixin(HierarchicalRewardMixin):
        pass

    mixin = MinimalMixin()
    advantages = torch.tensor([1.0, 2.0])
    repeated_stds, scaled = mixin._scale_advantages(advantages, [1, 1])

    assert repeated_stds == [1, 1]
    assert scaled.tolist() == advantages.tolist()


def test_scale_advantages_tolerates_bad_any(monkeypatch):
    class MinimalMixin(HierarchicalRewardMixin):
        pass

    class BadMask:
        dtype = "fake"

        def __init__(self, length):
            self.length = length

        def __eq__(self, _other):
            return self

        def any(self):
            raise TypeError("no any available")

        def __array__(self, dtype=None):
            import numpy as _np

            return _np.ones(self.length, dtype=dtype)

        def __float__(self):
            return 1.0

    mixin = MinimalMixin()
    advantages = torch.tensor([1.0, 3.0])
    mask = BadMask(len(advantages))

    repeated_stds, scaled = mixin._scale_advantages(advantages, mask)

    assert repeated_stds is mask
    assert scaled.tolist() == advantages.tolist()


def test_append_length_metrics_tolerates_bad_float_and_empty_terminals():
    class BadFloatTensor:
        def __init__(self, tensor):
            self.tensor = tensor
            self.device = getattr(tensor, "device", None)

        def float(self):
            raise TypeError("cannot cast to float")

        def __getitem__(self, idx):
            index = getattr(idx, "data", idx)
            return self.tensor.__getitem__(index)

        def __len__(self):
            return len(self.tensor)

    class LengthMetricsMixin(HierarchicalRewardMixin):
        def __init__(self):
            self._metrics = {"train": defaultdict(list)}
            self.accelerator = SimpleNamespace(gather=lambda x: x)

    mixin = LengthMetricsMixin()
    aggregated_completion_lengths = BadFloatTensor(torch.tensor([1.0, 2.0]))
    is_eos = torch.zeros((2, 1), dtype=getattr(torch, "bool", None))
    batch = SimpleNamespace(device="cpu")

    mixin._append_length_metrics(
        aggregated_completion_lengths,
        is_eos,
        batch,
        mode_str="train",
    )

    metrics = mixin._metrics["train"]
    assert metrics["completions/mean_length"]
    assert metrics["completions/min_length"]
    assert metrics["completions/max_length"]
    assert metrics["completions/clipped_ratio"][-1] == pytest.approx(0.5)
