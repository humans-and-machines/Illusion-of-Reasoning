import types

import pytest

import src.training.utils.hierarchical_rollout as hr


try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

skip_reason = None
if torch is None:
    skip_reason = "torch not available"
elif isinstance(torch, types.SimpleNamespace):
    skip_reason = "torch stub lacks required attributes"
else:
    try:
        if not hasattr(torch, "tensor") or not hasattr(torch.tensor([]), "view"):
            skip_reason = "torch stub lacks tensor ops"
    except Exception:
        skip_reason = "torch stub lacks tensor ops"

pytestmark = pytest.mark.skipif(skip_reason is not None, reason=skip_reason or "")
# Skip when running against a minimal torch stub without basic tensor ops.
if torch is not None and not hasattr(torch.tensor([]), "view"):
    pytest.skip("torch stub lacks view/sum ops required for hierarchical rollout unit tests", allow_module_level=True)


class _StubTokenizer:
    def __init__(self, pad_id=0, eos_id=9):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.pad_token = "<pad>"

    def encode(self, text, add_special_tokens=False):
        # simple mapping: number sequences become list of ints, tags map to ids.
        if text == "</think>":
            return [99]
        if text == "<answer>":
            return [100]
        return [ord(c) % 50 for c in str(text)]

    def batch_decode(self, seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        # convert ids back into joined string for compatibility
        out = []
        for seq in seqs:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            out.append(" ".join(map(str, seq)))
        return out


class _StubModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, *args, **kwargs):
        return torch.tensor(self.outputs)


def test_build_completion_mask_truncation():
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.processing_class = types.SimpleNamespace(eos_token_id=99)
    inst.mask_truncated_completions = True

    completion_ids = torch.tensor([[1, 99, 2], [3, 4, 5]])
    mask, ids_list, lengths, is_eos = inst._build_completion_mask(completion_ids, device=completion_ids.device)
    assert mask[0].tolist() == [1, 1, 0]
    assert ids_list[0] == [1, 99]
    assert ids_list[1] == []  # truncated row zeroed out
    assert lengths.tolist()[0] == 2
    assert is_eos.any(dim=1).tolist() == [True, False]


def test_normalize_rewards_and_slice():
    inst = hr.HierarchicalGRPOTrainer.__new__(hr.HierarchicalGRPOTrainer)
    inst.num_generations = 2
    inst.scale_rewards = True
    stats = inst._normalize_rewards(torch.tensor([1.0, 3.0, 5.0, 7.0]))
    assert stats.advantages.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])
    assert stats.is_std_zero.eq(False).all()

    batch = hr.GenerationBatch(
        prompts=[0, 1],
        prompts_text=["p0", "p1"],
        prompt_ids=None,
        prompt_mask=None,
        device=None,
    )
    inst.accelerator = types.SimpleNamespace(process_index=0)
    adv, all_adv = inst._slice_advantages_for_process(stats, batch)
    assert adv.tolist() == pytest.approx([-1.0, 1.0])
    assert all_adv.shape[0] == 4


def test_hierarchical_rollout_stage1_adds_tags():
    tok = _StubTokenizer(pad_id=0, eos_id=10)
    model = _StubModel(outputs=[[1, 2, 3]])
    hr_rollout = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None, max_reason_tokens=5)
    input_ids = torch.tensor([[5, 6]])
    reason_ids = hr_rollout._run_stage1_reasoning(input_ids, device=input_ids.device)
    assert reason_ids.shape[1] >= 3
    assert reason_ids[0, -2:].tolist() == [99, 100]  # </think> + <answer>


def test_hierarchical_rollout_stage2_vllm():
    tok = _StubTokenizer(pad_id=0, eos_id=10)
    vllm_calls = []

    class _StubVLLM:
        def generate(self, prompts, n, max_tokens, **kwargs):
            vllm_calls.append((prompts, n, max_tokens))
            return [[11, 12]]

    hr_rollout = hr.HierarchicalRollout(
        model=_StubModel(outputs=[[1]]), tokenizer=tok, vllm_client=_StubVLLM(), max_reason_tokens=5
    )
    reason_ids = torch.tensor([[7, 8]], dtype=torch.long)
    full = hr_rollout._run_stage2_answer(reason_ids, device=reason_ids.device, max_new_tokens=2)
    assert full.tolist()[0][-2:] == [11, 12]
    assert vllm_calls and vllm_calls[0][1] == 1


def test_hierarchical_rollout_call_uses_stages(monkeypatch):
    tok = _StubTokenizer()
    model = _StubModel(outputs=[[1]])
    rollout = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None)

    stage_calls = []
    monkeypatch.setattr(
        rollout,
        "_run_stage1_reasoning",
        lambda input_ids, device, **kw: stage_calls.append("s1") or torch.tensor([[1, 99, 100]]),
    )
    monkeypatch.setattr(
        rollout,
        "_run_stage2_answer",
        lambda reason_ids, device, **kw: stage_calls.append("s2") or torch.tensor([[1, 2, 3]]),
    )

    reason, full = rollout(input_ids=torch.tensor([[5]]), max_new_tokens=2)
    assert stage_calls == ["s1", "s2"]
    assert reason.shape[0] == 1 and full.shape[0] == 1
