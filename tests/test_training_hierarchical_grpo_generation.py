import contextlib
import types

from src.training.utils import hierarchical_grpo_generation as hgg


def _pad_fn(tensors, padding_value=0, padding_side="right"):
    max_len = max((len(getattr(t, "data", t)) for t in tensors), default=0)
    padded = []
    for tensor in tensors:
        data = list(getattr(tensor, "data", tensor))
        pad_len = max_len - len(data)
        if padding_side == "right":
            data = data + [padding_value] * pad_len
        else:
            data = [padding_value] * pad_len + data
        padded.append(data)
    return hgg.torch.tensor(padded)


class _UnwrapCtx(contextlib.AbstractContextManager):
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *_):
        return False


class _DummyUnwrapped:
    def __init__(self, outputs):
        self.outputs = outputs
        self.config = types.SimpleNamespace(_attn_implementation=None)

    def generate_batch(self, *_a, **_k):
        return self.outputs

    def generate(self, *_a, **_k):
        return hgg.torch.tensor([[1, 2, 3, 4]])

    def to(self, *_a, **_k):
        return None


class _DummyMixin(hgg.HierarchicalGenerationMixin):
    def __init__(self):
        self.rollout_fn = None
        self.use_vllm = False
        self.use_transformers_paged = False
        self.max_completion_length = 2
        self.num_generations = 1
        self.repetition_penalty = 1.0
        self.temperature = 0.1
        self.top_p = 0.9
        self.top_k = None
        self.min_p = None
        self.guided_decoding_regex = None
        self.vllm_client = types.SimpleNamespace(generate=lambda **_k: [])
        self.accelerator = types.SimpleNamespace(process_index=0, is_main_process=True)
        self.processing_class = types.SimpleNamespace(
            pad_token_id=0,
            eos_token_id=1,
            pad_token="<pad>",
            __call__=lambda self, text: types.SimpleNamespace(
                input_ids=[[1, 2]],
                pad_token_id=0,
            ),
            batch_decode=lambda ids, skip_special_tokens=True: ["decoded"] * len(ids),
        )
        self.args = types.SimpleNamespace(
            ds3_gather_for_generation=False,
            bf16=False,
            fp16=False,
        )
        self.generation_config = object()
        self.is_fsdp_enabled = False
        self.model_wrapped = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="orig"))
        self.pad = _pad_fn
        self.broadcast_object_list = lambda objs, **_k: objs
        self.gather_object = lambda obj, **_k: obj
        self.profiling_context = lambda *a, **k: contextlib.nullcontext()
        self.unwrap_model_for_generation = lambda *a, **k: _UnwrapCtx(self.model_wrapped)
        self.FSDP = types.SimpleNamespace(summon_full_params=lambda *_a, **_k: contextlib.nullcontext())
        self.mask_truncated_completions = True

        class Processing:
            pad_token_id = 0
            eos_token_id = 1
            pad_token = "<pad>"

            def __call__(self, text):
                return types.SimpleNamespace(input_ids=[[1, 2]], pad_token_id=0)

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["decoded"] * len(ids)

        self.processing_class = Processing()


def test_run_generation_passthrough_non_tuple():
    mixin = _DummyMixin()
    mixin.use_vllm = True
    mixin._generate_with_vllm = lambda batch: "done"
    batch = hgg.GenerationBatch(
        prompts=[], prompts_text=[], prompt_ids=hgg.torch.tensor([[1, 2]]), prompt_mask=None, device="cpu"
    )
    assert mixin._run_generation_backends(batch) == "done"


def test_generate_with_vllm_empty_completion(monkeypatch):
    mixin = _DummyMixin()
    mixin.use_vllm = True
    mixin.accelerator = types.SimpleNamespace(process_index=0, is_main_process=False)
    mixin.broadcast_object_list = lambda objs, **_k: []
    mixin.gather_object = lambda obj, **_k: []
    monkeypatch.setattr(
        hgg.torch,
        "zeros",
        lambda dim0, dim1=0, dtype=None, device=None: hgg.torch.tensor([[] * dim1 for _ in range(dim0)]),
        raising=False,
    )
    batch = hgg.GenerationBatch(
        prompts=[1],
        prompts_text=["p"],
        prompt_ids=hgg.torch.tensor([[]]),
        prompt_mask=None,
        device="cpu",
    )
    prompt_completion, padded = mixin._generate_with_vllm(batch)
    assert prompt_completion.shape[1] == 0
    assert padded.shape[1] == 0


def test_generate_with_paged_and_restores_attn():
    mixin = _DummyMixin()
    mixin.use_transformers_paged = True
    mixin.is_flash_attn_2_available = lambda: False
    mixin.pad = _pad_fn
    tensor_cls = type(hgg.torch.tensor([1]))
    setattr(
        tensor_cls,
        "__ne__",
        lambda self, other: hgg.torch.tensor(self.data != (getattr(other, "data", other))),
    )
    hgg.torch.bfloat16 = getattr(hgg.torch, "bfloat16", "bfloat16")
    outputs = {0: types.SimpleNamespace(generated_tokens=[3, 4, 5])}
    mixin.unwrap_model_for_generation = lambda *a, **k: _UnwrapCtx(_DummyUnwrapped(outputs))
    mixin.args.bf16 = True
    batch = hgg.GenerationBatch(
        prompts=[1],
        prompts_text=["p"],
        prompt_ids=hgg.torch.tensor([[9, 9]]),
        prompt_mask=hgg.torch.tensor([[1, 1]]),
        device="cpu",
    )
    completion = mixin._generate_with_paged(batch)
    assert completion.shape[0] == 1
    assert mixin.model_wrapped.config._attn_implementation == "orig"


def test_generate_with_hf_and_completion_slice():
    mixin = _DummyMixin()
    mixin.unwrap_model_for_generation = lambda *a, **k: _UnwrapCtx(
        _DummyUnwrapped({0: types.SimpleNamespace(generated_tokens=[1])})
    )
    mixin.args.fp16 = True
    prompt_ids = hgg.torch.tensor([[1, 2, 3]])
    prompt_mask = hgg.torch.tensor([[1, 1, 1]])
    full, comp = mixin._generate_with_hf(prompt_ids, prompt_mask)
    assert comp.shape[1] == full.shape[1] - prompt_ids.shape[1]


def test_build_completion_mask_tolist_guard(monkeypatch):
    mixin = _DummyMixin()
    tensor_cls = type(hgg.torch.tensor([1]))
    original_tolist = tensor_cls.tolist
    monkeypatch.setattr(tensor_cls, "tolist", lambda self: (_ for _ in ()).throw(AttributeError()))
    monkeypatch.setattr(
        tensor_cls,
        "expand_as",
        lambda self, other: hgg.torch.tensor([[int(x) for x in range(other.size(1))] for _ in range(other.size(0))]),
        raising=False,
    )
    completion_ids = hgg.torch.tensor([[1, 2], [3, 4]])
    mask, ids_list, lengths, is_eos = mixin._build_completion_mask(completion_ids, device="cpu")
    assert mask.shape[0] == completion_ids.shape[0]
    assert isinstance(ids_list, list)
    monkeypatch.setattr(tensor_cls, "tolist", original_tolist)
