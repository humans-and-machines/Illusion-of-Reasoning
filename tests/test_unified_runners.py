import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.inference.cli import unified_carpark as carpark_cli
from src.inference.cli import unified_crossword as crossword_cli
from src.inference.runners.unified_math_runner import run_math_inference
from src.inference.runners.unified_runner_base import MathTestConfig


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 99
        self._decode_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Minimal template: concatenate user content.
        user = [m["content"] for m in messages if m["role"] == "user"]
        return " ".join(user) + (" <GEN>" if add_generation_prompt else "")

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        # Encode to fixed-length tensors.
        bsz = len(texts)
        seq_len = 5
        input_ids = torch.ones((bsz, seq_len), dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}

    def decode(self, ids, skip_special_tokens=True):
        # Alternate between think/answer-ish strings.
        self._decode_calls += 1
        return f"gen{self._decode_calls}"

    def convert_tokens_to_ids(self, tok):
        return {"<|im_end|>": 97, "<|endoftext|>": 98}.get(tok, None)


class FakeGenerateOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    def generate(self, **kwargs):
        # Build a single sequence with 2 generated tokens beyond prompt.
        input_ids = kwargs["input_ids"]
        bsz, seq_len = input_ids.shape
        sequences = torch.cat([input_ids, torch.full((bsz, 2), 5, dtype=torch.long)], dim=1)
        # Create two score steps with simple logits.
        vocab = 10
        score_step = torch.zeros((bsz, vocab))
        scores = [score_step.clone(), score_step.clone()]
        return FakeGenerateOutput(sequences=sequences, scores=scores)


def test_run_math_inference_smoke(tmp_path):
    backend = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeModel())
    dataset = [{"problem": "2+2?", "answer": "4"}]

    cfg = MathTestConfig(
        dataset=dataset,
        output_dir=str(tmp_path),
        step=0,
        batch_size=1,
        num_samples=1,
        temperature=0.0,
        top_p=0.95,
        think_cap=5,
        answer_cap=5,
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        eos_ids=[backend.tokenizer.eos_token_id],
    )
    run_math_inference(backend=backend, config=cfg)

    outpath = tmp_path / "step0000_test.jsonl"
    assert outpath.exists()
    rows = [json.loads(l) for l in outpath.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["problem"] == "2+2?"
    assert "pass1" in row and row["pass1"]["pred_answer"] is not None
    assert row["pass2"] is None


def test_unified_carpark_runner_invokes_inference(monkeypatch, tmp_path):
    called = {}

    class DummyCarparkModule:
        def load_rush_dataset(self, **kwargs):
            called["load_kwargs"] = kwargs
            return [{"messages": "m", "solution": "s"}]

        def run_inference_on_split(self, **kwargs):
            called["run_kwargs"] = kwargs

    class DummyBackend:
        def __init__(self):
            self.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, convert_tokens_to_ids=lambda x: 2)
            self.model = "dummy-model"

    monkeypatch.setattr(carpark_cli, "_load_carpark_module", lambda: DummyCarparkModule())
    monkeypatch.setattr(carpark_cli, "HFBackend", SimpleNamespace(from_pretrained=lambda **_: DummyBackend()))

    argv = [
        "prog",
        "--model_name_or_path",
        "dummy",
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / ".hf" / "transformers"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / ".hf" / "hub"))

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / ".torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / ".triton"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    # Invoke main with patched argv
    monkeypatch.setattr(sys, "argv", argv)
    carpark_cli.main()

    assert "load_kwargs" in called
    assert "run_kwargs" in called


def test_unified_crossword_runner_invokes_inference(monkeypatch, tmp_path):
    called = {}

    class DummyCrosswordModule:
        def load_crossword_local(self, path):
            called["load_kwargs"] = {"dataset_path": path}
            return [{"clue": "C1", "answer": "A1"}]

        def run_inference_on_split(self, **kwargs):
            called["run_kwargs"] = kwargs

    class DummyBackend:
        def __init__(self):
            self.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, convert_tokens_to_ids=lambda x: 2)
            self.model = "dummy-model"

    monkeypatch.setattr(crossword_cli, "_load_crossword_module", lambda: DummyCrosswordModule())
    monkeypatch.setattr(crossword_cli, "HFBackend", SimpleNamespace(from_pretrained=lambda **_: DummyBackend()))

    dataset_path = tmp_path / "cw.jsonl"
    dataset_path.write_text('{"clue":"C1","answer":"A1"}\n')
    argv = [
        "prog",
        "--model_name_or_path",
        "dummy",
        "--output_dir",
        str(tmp_path),
        "--dataset_id",
        "CROSSWORD-LOCAL",
        "--dataset_path",
        str(dataset_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    crossword_cli.main()

    assert called["run_kwargs"]["split_name"] == "test"
    assert called["run_kwargs"]["batch_size"] == 8
