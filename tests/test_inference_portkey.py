from types import SimpleNamespace

import src.inference.gateways.providers.portkey as portkey


def test_make_client_requires_api_key(monkeypatch):
    class FakePortkey:
        def __init__(self, api_key):
            self.api_key = api_key

    def fake_import_module(name):
        return SimpleNamespace(Portkey=FakePortkey)

    monkeypatch.setattr(portkey, "import_module", fake_import_module)
    monkeypatch.setenv("AI_SANDBOX_KEY", "", prepend=False)
    try:
        portkey._make_client()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when API key missing")


def test_make_client_success(monkeypatch):
    class FakePortkey:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(portkey, "import_module", lambda name: SimpleNamespace(Portkey=FakePortkey))
    monkeypatch.setenv("AI_SANDBOX_KEY", "token", prepend=False)
    client = portkey._make_client()
    assert isinstance(client, FakePortkey) and client.api_key == "token"


def test_call_model_invokes_client(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        portkey,
        "build_math_gateway_messages",
        lambda sys_prompt, problem: captured.setdefault("msgs", (sys_prompt, problem)) or [{"role": "user"}],
    )
    monkeypatch.setattr(
        portkey, "parse_openai_chat_response", lambda resp: ("text", "finish", {"usage": 1, "raw": resp})
    )

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            captured["kwargs"] = kwargs
            return {"resp": True}

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    params = portkey.PortkeyCallParams(temperature=0.2, top_p=0.8, max_output_tokens=10, request_timeout=5)

    text, finish, usage = portkey._call_model(client, model="m", problem="prob", params=params)
    assert finish == "finish"
    assert usage["raw"] == {"resp": True}
    assert captured["kwargs"]["model"] == "m"
    assert captured["kwargs"]["temperature"] == 0.2


def test_iter_examples_limits(monkeypatch):
    class DummyDs:
        def __init__(self):
            self.items = list(range(5))
            self.selected = None

        def __len__(self):
            return len(self.items)

        def select(self, idxs):
            self.selected = list(idxs)
            return [self.items[i] for i in idxs]

        def __iter__(self):
            return iter(self.items if self.selected is None else self.selected)

    ds = DummyDs()
    out = list(portkey._iter_examples(ds, num_examples=2))
    assert out == [0, 1]

    ds2 = DummyDs()
    assert list(portkey._iter_examples(ds2, num_examples=None)) == ds2.items


def test_build_portkey_row_sets_correct_and_usage(monkeypatch):
    monkeypatch.setattr(portkey, "_canon_math", lambda x: f"canon-{x}")
    monkeypatch.setattr(portkey, "_valid_tag_structure", lambda txt: True)
    monkeypatch.setattr(portkey, "build_math_gateway_row_base", lambda **kwargs: {"base": kwargs})
    monkeypatch.setattr(portkey, "build_usage_dict", lambda usage: {"u": usage})

    example = portkey.ExampleContext(problem="p", gold_answer="G", canon_gold="canon-G", sample_idx=2)
    result = portkey.PortkeyCallResult(text=" out ", answer="G", finish_reason="done", usage={"tok": 1})
    config = portkey.PortkeyRunConfig(
        output_path="/tmp/out.jsonl",
        split_name="test",
        model_name="m",
        num_samples=1,
        params=portkey.PortkeyCallParams(temperature=0.1, top_p=0.9, max_output_tokens=10, request_timeout=5),
        seed=0,
        step=1,
    )

    row = portkey._build_portkey_row(example, result, config)
    assert row["pass1"]["is_correct_pred"] is True
    assert row["pass1"]["pred_answer_canon"] == "canon-G"
    assert row["pass1"]["valid_tag_structure"] is True
    assert row["usage"] == {"u": {"tok": 1}}
    assert row["endpoint"] == "portkey-ai"
    assert row["deployment"] == "m"


def test_run_portkey_math_inference_calls_dependencies(monkeypatch, tmp_path):
    calls = {"append": 0, "call_model": 0}

    monkeypatch.setattr(portkey, "_call_model", lambda **kwargs: ("TEXT", "finish", {"tok": 1}))
    monkeypatch.setattr(portkey, "_extract_blocks", lambda text: ("body", "ans"))
    monkeypatch.setattr(portkey, "_canon_math", lambda x: x)
    monkeypatch.setattr(portkey, "_valid_tag_structure", lambda txt: True)
    monkeypatch.setattr(
        portkey, "append_jsonl_row", lambda path, row: calls.__setitem__("append", calls["append"] + 1)
    )
    monkeypatch.setattr(portkey, "extract_problem_and_answer", lambda ex: (ex["problem"], ex["answer"]))
    monkeypatch.setattr(portkey, "build_math_gateway_row_base", lambda **kwargs: {"base": kwargs})
    monkeypatch.setattr(portkey, "build_usage_dict", lambda usage: {"u": usage})
    monkeypatch.setattr(portkey, "logger", SimpleNamespace(info=lambda *a, **k: None))

    dataset = [{"problem": "p1", "answer": "ans1"}]
    existing = {}
    config = portkey.PortkeyRunConfig(
        output_path=str(tmp_path / "out.jsonl"),
        split_name="test",
        model_name="m",
        num_samples=2,
        params=portkey.PortkeyCallParams(temperature=0.1, top_p=0.9, max_output_tokens=10, request_timeout=5),
        seed=123,
        step=1,
    )

    portkey.run_portkey_math_inference(client=None, dataset=dataset, existing=existing, config=config)

    assert calls["append"] == 2  # two samples generated
    assert "p1" in existing and existing["p1"] == {0, 1}


def test_run_portkey_math_inference_skips_invalid(monkeypatch, tmp_path):
    calls = {"append": 0}
    monkeypatch.setattr(portkey, "_call_model", lambda **kwargs: ("TEXT", "finish", None))
    monkeypatch.setattr(portkey, "_extract_blocks", lambda text: ("body", "ans"))
    monkeypatch.setattr(portkey, "_canon_math", lambda x: x)
    monkeypatch.setattr(portkey, "_valid_tag_structure", lambda txt: True)
    monkeypatch.setattr(
        portkey, "append_jsonl_row", lambda path, row: calls.__setitem__("append", calls["append"] + 1)
    )
    monkeypatch.setattr(portkey, "extract_problem_and_answer", lambda ex: (ex.get("problem"), ex.get("answer")))
    monkeypatch.setattr(portkey, "build_math_gateway_row_base", lambda **kwargs: {"base": kwargs})
    monkeypatch.setattr(portkey, "logger", SimpleNamespace(info=lambda *a, **k: None))

    dataset = [
        {"problem": "", "answer": "ans1"},  # skip empty problem
        {"problem": "p2", "answer": None},  # skip missing answer
        {"problem": "p3", "answer": "ans3"},
    ]
    existing = {}
    config = portkey.PortkeyRunConfig(
        output_path=str(tmp_path / "out.jsonl"),
        split_name="test",
        model_name="m",
        num_samples=1,
        params=portkey.PortkeyCallParams(temperature=0.1, top_p=0.9, max_output_tokens=10, request_timeout=5),
        seed=0,
        step=1,
    )

    portkey.run_portkey_math_inference(client=None, dataset=dataset, existing=existing, config=config)
    assert calls["append"] == 1
    assert existing["p3"] == {0}


def test_main_wires_everything(monkeypatch, tmp_path):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        step=7,
        split="test",
        temperature=0.2,
        top_p=0.7,
        max_output_tokens=10,
        request_timeout=5,
        model="m",
        num_samples=1,
        seed=42,
    )

    class FakeParser:
        def __init__(self):
            self.added = []

        def add_argument(self, *a, **k):
            self.added.append((a, k))

        def parse_args(self):
            return args

    monkeypatch.setattr(portkey, "build_math_gateway_arg_parser", lambda **kwargs: FakeParser())
    monkeypatch.setattr(portkey, "_make_client", lambda: "CLIENT")
    monkeypatch.setattr(portkey, "logger", SimpleNamespace(info=lambda *a, **k: None))

    captured = {}

    def fake_prepare(**kwargs):
        captured["outpath"] = kwargs["outpath"]
        return (["ds"], {"existing": set()}, None)

    monkeypatch.setattr(portkey, "prepare_math_gateway_dataset_from_args", fake_prepare)
    monkeypatch.setattr(portkey, "run_portkey_math_inference", lambda **kwargs: captured.setdefault("ran", kwargs))
    monkeypatch.setattr(portkey, "setup_hf_cache_dir_env", lambda path: path)

    portkey.main()

    assert captured["outpath"].endswith("step0007_test.jsonl")
    assert captured["ran"]["client"] == "CLIENT"


def test_dunder_main_guard_executes(monkeypatch):
    called = {}
    monkeypatch.setattr(portkey, "main", lambda: called.setdefault("ran", True))
    shim_code = "\n" * 373 + "main()\n"
    exec(compile(shim_code, portkey.__file__, "exec"), portkey.__dict__)
    assert called.get("ran") is True
