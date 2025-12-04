import types

import numpy as np
import pytest

import src.inference.backends as back


def test_load_torch_and_transformers_stub(monkeypatch):
    fake_torch = types.SimpleNamespace()

    def fake_import(name):
        if name == "torch":
            return fake_torch
        raise ImportError("missing")

    monkeypatch.setattr(back, "import_module", fake_import)

    torch_mod, tok_cls, model_cls, stop_cls = back._load_torch_and_transformers(require_transformers=False)
    assert torch_mod is fake_torch and tok_cls is None and model_cls is None
    inst = stop_cls(["a"])
    assert isinstance(inst, list) and inst == ["a"]

    with pytest.raises(ImportError):
        back._load_torch_and_transformers(require_transformers=True)


def test_load_hf_tokenizer_and_model(monkeypatch):
    class FakeTok:
        def __init__(self):
            self.padding_side = None
            self.pad_token = None
            self.eos_token = "eos"
            self.truncation_side = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    tok = back._load_hf_tokenizer(FakeTok, "src", revision=None, trust_remote_code=True, cache_dir=None)
    assert tok.padding_side == "left" and tok.truncation_side == "left"
    assert tok.pad_token == "eos"

    class FakeModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    torch_mod = types.SimpleNamespace(float16="f16", bfloat16="bf16")
    model = back._load_hf_model(
        FakeModel,
        torch_mod,
        "m",
        revision=None,
        trust_remote_code=True,
        cache_dir=None,
        dtype="bfloat16",
        device_map="auto",
        attn_implementation=None,
    )
    assert isinstance(model, FakeModel) and model.eval_called is True


def test_normalize_hf_load_config_prefers_instance():
    cfg = back.HFBackendLoadConfig(revision="rev", cache_dir="cache")
    normalized = back._normalize_hf_load_config(cfg)
    assert normalized is cfg


def test_build_hf_stopping_criteria(monkeypatch):
    class StubStopList(list):
        def __call__(self, criteria):
            return self.__class__(criteria or [])

    captured = {}
    monkeypatch.setattr(back, "StopOnSubstrings", lambda tok, stops: captured.setdefault("stops", stops) or "stub")
    assert back._build_hf_stopping_criteria(StubStopList, tokenizer="tok", stop_strings=None) is None
    res = back._build_hf_stopping_criteria(StubStopList, tokenizer="tok", stop_strings=["x"])
    assert isinstance(res, StubStopList) and captured["stops"] == ["x"]


def test_prepare_hf_inputs_uses_move_to_device(monkeypatch):
    called = {}

    class FakeTok:
        def __call__(self, prompts, **kwargs):
            called["prompts"] = prompts
            called["kwargs"] = kwargs
            return {"enc": True}

    monkeypatch.setattr(back, "move_inputs_to_device", lambda inputs: {"moved": inputs})
    out_inputs = back._prepare_hf_inputs(FakeTok(), ["a", "b"], max_length=5)
    assert out_inputs == {"moved": {"enc": True}}
    assert called["kwargs"]["max_length"] == 5


def test_hf_decode_and_classify(monkeypatch):
    calls = []

    def fake_decode(tokenizer, sequences, input_lengths, row_i, skip_special_tokens=True):
        gen_ids = np.array([1, 2, 3][: row_i + 2])
        return gen_ids, f"text{row_i}", len(gen_ids)

    def fake_classify(found_stop, has_eos, hit_max):
        calls.append((found_stop, has_eos, hit_max))
        return "reason"

    monkeypatch.setattr(back, "decode_generated_row", fake_decode)
    monkeypatch.setattr(back, "classify_stop_reason", fake_classify)

    seqs = types.SimpleNamespace(shape=(2, 3))
    ctx = back.HFDecodeContext(
        tokenizer=None,
        model=types.SimpleNamespace(config=types.SimpleNamespace(eos_token_id=2)),
        sequences=seqs,
        input_lengths=[0, 0],
        stop_strings=["text0"],
        max_new_tokens=2,
    )
    texts, reasons = back.HFBackend._decode_and_classify(ctx)
    assert texts == ["text0", "text1"]
    assert reasons == ["reason", "reason"]
    assert calls[0][0] is True and calls[1][2] is True  # first found stop, second hit max


def test_hf_backend_generate_and_stop_detection(monkeypatch):
    class FakeCtx:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub = type("TorchStub", (), {"inference_mode": lambda self=None: FakeCtx()})()
    stop_calls = {}
    monkeypatch.setattr(
        back,
        "_load_torch_and_transformers",
        lambda: (torch_stub, None, None, lambda criteria: stop_calls.setdefault("crit", criteria)),
    )
    monkeypatch.setattr(
        back,
        "_prepare_hf_inputs",
        lambda tokenizer, prompts, max_length: ({"input_ids": ["x"]}, [0, 0]),
    )

    class Model:
        def __init__(self):
            self.config = type("Cfg", (), {"eos_token_id": [2, 3]})()

        def generate(self, **kwargs):
            return type("GenOut", (), {"sequences": np.array([[1, 2], [4, 5]])})()

    tok = object()
    backend = back.HFBackend(tok, Model())

    monkeypatch.setattr(
        back,
        "decode_generated_row",
        lambda tokenizer, sequences, input_lengths, row_i, skip_special_tokens=True: (
            sequences[row_i],
            f"stop_{row_i}",
            None,
        ),
    )
    reasons_seen = []
    monkeypatch.setattr(
        back,
        "classify_stop_reason",
        lambda found, eos, maxed: reasons_seen.append((found, eos, maxed)) or "cls",
    )
    res = backend.generate(["p1", "p2"], stop_strings=["stop"], max_length=1, max_new_tokens=2)
    assert res.texts == ["stop_0", "stop_1"]
    assert stop_calls["crit"] and len(stop_calls["crit"]) == 1
    assert reasons_seen[0][0] is True  # stop string matched
    assert reasons_seen[0][1] is True  # eos detected via iterable
    assert reasons_seen[0][2] is True  # hit max_new_tokens


def test_hf_backend_from_pretrained_requires_transformers(monkeypatch):
    monkeypatch.setattr(back, "_load_torch_and_transformers", lambda: (None, None, None, None))
    with pytest.raises(ImportError):
        back.HFBackend.from_pretrained("model")


def test_azure_from_env(monkeypatch):
    fake_cfg = {"endpoint": "ep", "api_key": "k", "api_version": "v", "deployment": "d", "use_v1": True}

    class FakeClient:
        pass

    def fake_import(name):
        mod = types.SimpleNamespace(
            load_azure_config=lambda: fake_cfg,
            build_preferred_client=lambda endpoint, api_key, api_version, use_v1: (FakeClient(), True),
        )
        return mod

    monkeypatch.setattr(back, "import_module", fake_import)
    backend = back.AzureBackend.from_env(deployment="dep_override")
    assert isinstance(backend.client, FakeClient)
    assert backend.deployment == "dep_override" and backend.uses_v1 is True


def test_azure_from_env_falls_back_when_import_fails(monkeypatch):
    fake_cfg = {"endpoint": "ep", "api_key": "k", "api_version": "v", "deployment": "d"}
    monkeypatch.setattr(back, "import_module", lambda name: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(back, "load_azure_config", lambda: fake_cfg)
    monkeypatch.setattr(back, "build_preferred_client", lambda **kwargs: ("client", False))
    backend = back.AzureBackend.from_env()
    assert backend.client == "client"
    assert backend.deployment == "d"
    assert backend.uses_v1 is False


def test_azure_backend_init_requires_client():
    with pytest.raises(ValueError):
        back.AzureBackend(client=None, deployment="d", uses_v1=False)


def test_azure_call_responses_and_chat(monkeypatch):
    backend = back.AzureBackend(client=types.SimpleNamespace(), deployment="d", uses_v1=True)

    class RespObj:
        def __init__(self, text=None):
            self.output_text = text
            self.output = types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(content=[]))

    backend.client.responses = types.SimpleNamespace(create=lambda **kwargs: RespObj("ok"))
    text, fin, raw = backend._call_responses_api([], temperature=0.0, top_p=None, max_output_tokens=1)
    assert text == "ok" and fin == "stop"

    class Choice:
        def __init__(self):
            self.message = types.SimpleNamespace(content="chat")
            self.finish_reason = "len"

    backend.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: types.SimpleNamespace(choices=[Choice()]))
        )
    )
    text2, fin2, _ = backend._call_chat_completions([], temperature=0.0, top_p=None, max_output_tokens=1)
    assert text2 == "chat" and fin2 == "len"


def test_azure_call_routing_and_response_content(monkeypatch):
    class Part:
        def __init__(self, text):
            self.text = text

    class Output:
        def __init__(self):
            self.message = types.SimpleNamespace(content=[Part("a"), Part("b")])
            self.finish_reason = "message_finish"

    class Resp:
        def __init__(self):
            self.output_text = None
            self.output = Output()

    backend = back.AzureBackend(client=types.SimpleNamespace(), deployment="d", uses_v1=True)
    backend.client.responses = types.SimpleNamespace(create=lambda **kwargs: Resp())
    text, fin, _ = backend._call([], temperature=0.0, top_p=None, max_output_tokens=1)
    assert text == "ab"
    assert fin == "message_finish"

    # Fallback to chat.completions branch and error when missing completions
    backend.client = types.SimpleNamespace(chat=types.SimpleNamespace())
    with pytest.raises(RuntimeError):
        backend._call_chat_completions([], temperature=0.0, top_p=None, max_output_tokens=1)


def test_azure_generate_uses_call(monkeypatch):
    backend = back.AzureBackend(client=types.SimpleNamespace(), deployment="d", uses_v1=False)
    calls = []
    monkeypatch.setattr(backend, "_call", lambda msgs, **kw: calls.append(msgs) or ("txt", "fin", "raw"))
    res = backend.generate([[{"role": "user", "content": "hi"}]], temperature=0.1, top_p=0.9, max_output_tokens=5)
    assert res.texts == ["txt"] and res.finish_reasons == ["fin"]
    assert calls[0][0]["content"] == "hi"
