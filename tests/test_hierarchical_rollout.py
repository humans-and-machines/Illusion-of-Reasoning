import types

import src.training.utils.hierarchical_rollout as hr


class _FakeTensor:
    def __init__(self, data, device="cpu"):
        self.data = list(data)
        self.device = device

    def tolist(self):
        return list(self.data)

    def __iter__(self):
        return iter(self.data)


class _FakeTorch:
    def __init__(self):
        self.flags = {"no_grad": 0}

    def tensor(self, data, device=None, dtype=None):
        return _FakeTensor(data, device=device)

    class _NoGrad:
        def __init__(self, flags):
            self.flags = flags

        def __enter__(self):
            self.flags["no_grad"] += 1

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.flags["no_grad"] -= 1

    def no_grad(self):
        return self._NoGrad(self.flags)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [1]
        if text == "<answer>":
            return [2]
        return [len(text)]

    def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if isinstance(sequences, _FakeTensor):
            sequences = [sequences]
        out = []
        for seq in sequences:
            data = seq.tolist() if hasattr(seq, "tolist") else seq
            out.append(" ".join(str(x) for x in data))
        return out


class _FakeModel:
    def __init__(self, stage1_out=None, stage2_out=None):
        self.stage1_out = stage1_out or [[3, 4]]
        self.stage2_out = stage2_out or [[5, 6]]

    def generate(self, *_args, **_kwargs):
        # Return lists of FakeTensors to mimic torch output
        if _kwargs.get("eos_token_id") == 1:  # stage1 reasoning
            outputs = self.stage1_out
        else:
            outputs = self.stage2_out
        return [_FakeTensor(seq) for seq in outputs]


class _FakeVLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompts, n, max_tokens, **kwargs):
        self.calls.append({"prompts": prompts, "n": n, "max_tokens": max_tokens})
        return list(self.outputs)


def test_get_tag_ids_and_stage1_reasoning(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(hr, "torch", fake_torch)
    monkeypatch.setattr(hr, "pad_sequence", lambda seqs, batch_first=True, padding_value=0: seqs)

    tok = _FakeTokenizer()
    model = _FakeModel(stage1_out=[[7, 8]])
    rollout = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None, max_reason_tokens=10)
    think_ids, answer_ids = rollout.get_tag_ids()
    assert think_ids == [1] and answer_ids == [2]

    reason_ids = rollout._run_stage1_reasoning(_FakeTensor([0, 1]), device="cpu")
    assert isinstance(reason_ids, list)
    assert reason_ids[0].tolist()[-2:] == [1, 2]  # think close + answer tag appended


def test_stage1_reasoning_with_vllm(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(hr, "torch", fake_torch)
    monkeypatch.setattr(hr, "pad_sequence", lambda seqs, batch_first=True, padding_value=0: seqs)

    tok = _FakeTokenizer()
    vllm = _FakeVLLM(outputs=[[9, 1]])  # already ends with </think> id
    rollout = hr.HierarchicalRollout(model=_FakeModel(), tokenizer=tok, vllm_client=vllm, max_reason_tokens=5)
    reason_ids = rollout._run_stage1_reasoning(_FakeTensor([0]), device="cpu")
    assert reason_ids[0].tolist()[-1] == 2  # answer tag appended
    assert vllm.calls[0]["max_tokens"] == 5


def test_stage2_answer_paths(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(hr, "torch", fake_torch)
    monkeypatch.setattr(hr, "pad_sequence", lambda seqs, batch_first=True, padding_value=0: seqs)

    tok = _FakeTokenizer()
    # Non-vLLM path uses model.generate output
    model = _FakeModel(stage2_out=[[1, 2, 3]])
    rollout = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None)
    full_ids = rollout._run_stage2_answer([_FakeTensor([1, 2])], device="cpu", max_new_tokens=4)
    assert full_ids[0].tolist() == [1, 2, 3]

    # vLLM path concatenates
    vllm = _FakeVLLM(outputs=[[4, 5]])
    rollout_vllm = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=vllm)
    full_ids_vllm = rollout_vllm._run_stage2_answer([_FakeTensor([7])], device="cpu", max_new_tokens=6)
    assert full_ids_vllm[0].tolist() == [7, 4, 5]
    assert vllm.calls[0]["max_tokens"] == 6


def test_call_uses_no_grad(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(hr, "torch", fake_torch)
    monkeypatch.setattr(hr, "pad_sequence", lambda seqs, batch_first=True, padding_value=0: seqs)

    tok = _FakeTokenizer()
    model = _FakeModel(stage1_out=[[1, 2, 3]], stage2_out=[[1, 2, 3, 4]])
    rollout = hr.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None)
    reason_ids, full_ids = rollout(_FakeTensor([0, 1]), max_new_tokens=3)

    assert fake_torch.flags["no_grad"] == 0  # exited context
    assert reason_ids[0].tolist()[-1] == 2
    assert isinstance(full_ids, list)


def test_call_skips_no_grad_when_torch_missing(monkeypatch):
    monkeypatch.setattr(hr, "torch", None)
    tok = _FakeTokenizer()
    rollout = hr.HierarchicalRollout(model=object(), tokenizer=tok, vllm_client=None)

    calls = {}

    def fake_stage1(input_ids, device, **_kwargs):
        calls["stage1"] = device
        return "reason"

    def fake_stage2(reason_ids, device, **_kwargs):
        calls["stage2"] = (reason_ids, device)
        return "full"

    monkeypatch.setattr(rollout, "_run_stage1_reasoning", fake_stage1)
    monkeypatch.setattr(rollout, "_run_stage2_answer", fake_stage2)

    out = rollout(types.SimpleNamespace(device="cuda"))
    assert out == ("reason", "full")
    assert calls["stage1"] == "cuda"
    assert calls["stage2"] == ("reason", "cuda")


def test_hierarchical_rollout_import_falls_back(monkeypatch):
    sentinel = "trainer_torch"
    monkeypatch.setattr(hr, "_trainer", types.SimpleNamespace(torch=sentinel), raising=False)
    hr._real_torch = None
    exec("\n" * 18 + "torch = _trainer.torch\n", hr.__dict__)
    assert hr.torch == sentinel
