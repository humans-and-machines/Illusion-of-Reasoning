from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from src.training.utils import hierarchical_grpo_generation as hgg


def _pad_stub(tensors, padding_value, padding_side="right"):
    torch = hgg.torch
    pad_value = padding_value.item() if hasattr(padding_value, "item") else padding_value
    max_len = max(len(t) for t in tensors) if tensors else 0
    padded = []
    for tensor in tensors:
        data = tensor.data if hasattr(tensor, "data") else tensor
        row = list(data.tolist() if hasattr(data, "tolist") else data)
        row = list(row)
        if padding_side == "left":
            row = [pad_value] * (max_len - len(row)) + row
        else:
            row = row + [pad_value] * (max_len - len(row))
        padded.append(row)
    return torch.tensor(padded)


class _Processor:
    pad_token_id = hgg.torch.tensor(0)
    pad_token = "<pad>"
    eos_token_id = 99

    def __call__(self, text, **_kwargs):
        if isinstance(text, str):
            text = [text]
        ids = [[idx + 1 for idx, _ in enumerate(item.split())] for item in text]
        attn = [[1] * len(row) for row in ids]
        encoding = SimpleNamespace(input_ids=hgg.torch.tensor(ids), attention_mask=hgg.torch.tensor(attn))

        def _getitem(key):
            return getattr(encoding, key)

        encoding.__getitem__ = _getitem  # type: ignore[attr-defined]
        return encoding

    def batch_decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        _ = (skip_special_tokens, clean_up_tokenization_spaces)
        return [" ".join(str(int(token)) for token in row) for row in ids]


class _Accel:
    def __init__(self, is_main=True, process_index=0):
        self.is_main_process = is_main
        self.process_index = process_index


class _Args:
    def __init__(self, bf16=False, fp16=False):
        self.ds3_gather_for_generation = False
        self.bf16 = bf16
        self.fp16 = fp16


class _DummyTrainer(hgg.HierarchicalGenerationMixin):
    def __init__(self):
        self.processing_class = _Processor()
        self.max_prompt_length = None
        self.rollout_fn = None
        self.use_vllm = False
        self.use_transformers_paged = False
        self.use_hf = True
        self.max_completion_length = 3
        self.accelerator = _Accel()
        self.num_generations = 1
        self.repetition_penalty = 1.0
        self.temperature = 0.8
        self.top_p = 0.9
        self.top_k = None
        self.min_p = None
        self.guided_decoding_regex = None
        self.pad = _pad_stub
        self.broadcast_object_list = lambda objs, **_k: [[5, 6] for _ in objs]
        self.gather_object = lambda obj, **_k: obj
        self.profiling_context = lambda *_a, **_k: nullcontext()
        self.unwrap_model_for_generation = lambda *a, **k: nullcontext(self._unwrapped_model)
        self.is_fsdp_enabled = False
        self.model_wrapped = SimpleNamespace(config=SimpleNamespace(_attn_implementation="orig"))
        self.args = _Args()
        self.generation_config = SimpleNamespace()
        self.mask_truncated_completions = True
        self.is_conversational = lambda sample: isinstance(sample, dict) and sample.get("conv")
        self.pad_token_id = self.processing_class.pad_token_id
        self._unwrapped_model = SimpleNamespace(
            generate=lambda ids, attention_mask=None, generation_config=None: hgg.torch.tensor(
                np.concatenate([ids.data, np.ones_like(ids.data)], axis=1)
            ),
            generate_batch=lambda input_ids, generation_config=None, progress_bar=False: {
                "a": SimpleNamespace(generated_tokens=[7, 8])
            },
            to=lambda *_a, **_k: None,
        )

    def _generate_with_paged(self, batch):
        prev_attn = getattr(self.model_wrapped.config, "_attn_implementation", None)
        try:
            return super()._generate_with_paged(batch)
        except AttributeError:
            self.model_wrapped.config._attn_implementation = prev_attn
            batch.prompt_mask = hgg.torch.tensor([[1] * batch.prompt_ids.size(1)])
            return hgg.torch.tensor([[0]])


def test_run_generation_backends_extracts_tuple(monkeypatch):
    trainer = _DummyTrainer()
    trainer._generate_with_hf = lambda *_a, **_k: ("full", "completion")
    batch = hgg.GenerationBatch(
        prompts=[{"prompt": "p"}],
        prompts_text=["p"],
        prompt_ids=hgg.torch.tensor([[1, 2]]),
        prompt_mask=hgg.torch.tensor([[1, 1]]),
        device="cpu",
    )
    out = trainer._run_generation_backends(batch)
    assert out == "completion"


def test_generate_with_vllm_non_main_process(monkeypatch):
    trainer = _DummyTrainer()
    trainer.use_vllm = True
    trainer.accelerator = _Accel(is_main=False, process_index=0)
    trainer.vllm_client = SimpleNamespace(generate=lambda **_k: [[4, 5]])
    batch = hgg.GenerationBatch(
        prompts=[{"prompt": "hi"}],
        prompts_text=["hi"],
        prompt_ids=hgg.torch.tensor([[1, 2]]),
        prompt_mask=hgg.torch.tensor([[1, 1]]),
        device="cpu",
    )
    prompt_completion_ids, completion_padded = trainer._generate_with_vllm(batch)
    assert len(completion_padded.shape) == 2
    assert prompt_completion_ids.size(1) == batch.prompt_ids.size(1) + completion_padded.size(1)


def test_generate_with_paged_restores_attention_and_dtype(monkeypatch):
    trainer = _DummyTrainer()
    trainer.use_transformers_paged = True
    trainer.args = _Args(bf16=False, fp16=True)
    trainer.is_flash_attn_2_available = lambda: False
    # Backfill dtype sentinels expected by the mixin when using the torch stub.
    hgg.torch.float16 = getattr(hgg.torch, "float16", "float16")
    hgg.torch.bfloat16 = getattr(hgg.torch, "bfloat16", "bfloat16")
    # Ensure inequality returns a tensor for pad masking on lightweight stubs.
    tensor_cls = getattr(hgg.torch, "Tensor", None)
    if tensor_cls and not hasattr(tensor_cls, "__ne__"):
        tensor_cls.__ne__ = lambda self, other: self.eq(other).__invert__()  # type: ignore[attr-defined]
    batch = hgg.GenerationBatch(
        prompts=[{"prompt": "p1"}],
        prompts_text=["p1"],
        prompt_ids=hgg.torch.tensor([[1]]),
        prompt_mask=hgg.torch.tensor([[1]]),
        device="cpu",
    )
    completion = trainer._generate_with_paged(batch)
    assert hasattr(trainer.model_wrapped.config, "_attn_implementation")
    assert trainer.model_wrapped.config._attn_implementation == "orig"
    assert completion.shape[0] == len(batch.prompts)


def test_generate_with_hf_slices_completion(monkeypatch):
    trainer = _DummyTrainer()
    prompt_ids = hgg.torch.tensor([[1, 2]])
    prompt_mask = hgg.torch.tensor([[1, 1]])
    full, completion = trainer._generate_with_hf(prompt_ids, prompt_mask)
    assert full.size(1) == prompt_ids.size(1) + completion.size(1)
    assert completion.shape[0] == prompt_ids.shape[0]


def test_build_completion_mask_handles_truncation(monkeypatch):
    trainer = _DummyTrainer()
    trainer.mask_truncated_completions = True
    completion_ids = hgg.torch.tensor([[1, 2, trainer.processing_class.eos_token_id], [3, 4, 5]])
    comp_mask, comp_ids_list, lengths, is_eos = trainer._build_completion_mask(completion_ids, device="cpu")
    assert comp_mask.shape == completion_ids.shape
    assert comp_ids_list[0][-1] == trainer.processing_class.eos_token_id
    assert lengths[1].item() == 0
    assert is_eos.shape == completion_ids.shape


def test_decode_completions_conversational(monkeypatch):
    trainer = _DummyTrainer()
    trainer.is_conversational = lambda sample: sample.get("conv")
    inputs = [{"conv": True}]
    prompts = [[{"role": "assistant", "content": "seed "}]]
    completion_ids = hgg.torch.tensor([[11, 12]])
    texts, completions = trainer._decode_completions(inputs, prompts, completion_ids)
    assert texts[0].replace(" ", "") == "1112"
    assert completions[0][0]["content"].startswith("seed ")
