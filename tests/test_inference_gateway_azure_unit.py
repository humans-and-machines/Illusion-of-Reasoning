import argparse
import types

import pytest

import src.inference.gateways.providers.azure as az


def _args(**overrides):
    base = dict(
        endpoint="https://ep",
        deployment="dep",
        api_version="v",
        api_key="key",
        use_v1=1,
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=16,
        request_timeout=10,
        split="test",
        step=1,
        output_dir="out",
        seed=0,
        num_samples=1,
        max_retries=0,
        retry_backoff=1.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_make_client_uses_args(monkeypatch):
    monkeypatch.setattr(
        az, "load_azure_config", lambda: {"endpoint": "cfg_ep", "deployment": "cfg_dep", "api_version": "cfg_v"}
    )
    monkeypatch.setattr(az, "build_preferred_client", lambda **kwargs: ("client", True))
    args = _args(api_key="AKEY", endpoint="https://x", deployment="d", api_version="v", use_v1=1)
    client, uses_v1, endpoint, deployment, api_version = az._make_client(args)
    assert client == "client" and uses_v1 is True
    assert endpoint == "https://x" and deployment == "d" and api_version == "v"

    args_missing = _args(api_key=None, endpoint=None, deployment=None, api_version=None, use_v1=0)
    with pytest.raises(RuntimeError):
        az._make_client(args_missing)


def test_call_model_responses_and_chat(monkeypatch):
    params = az.AzureCallParams(temperature=0.1, top_p=0.2, max_output_tokens=5, request_timeout=1)

    class Choice:
        def __init__(self):
            self.finish_reason = "stop"
            self.message = types.SimpleNamespace(content="resp text")

    class RespOutput:
        def __init__(self):
            self.choices = [Choice()]

    class Resp:
        def __init__(self):
            self.output = RespOutput()
            self.usage = {"t": 1}

    client_v1 = types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kwargs: Resp()))
    text, fin, usage = az._call_model(client_v1, True, "dep", "prob", params)
    assert text == "resp text" and fin == "stop" and usage == {"t": 1}

    class ChatChoice:
        def __init__(self):
            self.finish_reason = "len"
            self.message = types.SimpleNamespace(content="chat text")

    client_chat = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kwargs: types.SimpleNamespace(choices=[ChatChoice()], usage={"u": 2})
            )
        )
    )
    text2, fin2, usage2 = az._call_model(client_chat, False, "dep", "prob", params)
    assert text2 == "chat text" and fin2 == "len" and usage2 == {"u": 2}


def test_build_result_row(monkeypatch):
    monkeypatch.setattr(az, "_canon_math", lambda x: f"canon-{x}")
    monkeypatch.setattr(az, "_extract_blocks", lambda text: ("think", "answer"))
    monkeypatch.setattr(az, "_valid_tag_structure", lambda text: True)
    monkeypatch.setattr(
        az,
        "build_math_gateway_row_base",
        lambda **kwargs: {"base": kwargs["problem"], "sample_idx": kwargs["sample_idx"]},
    )
    monkeypatch.setattr(az, "build_usage_dict", lambda usage: {"usage_built": usage})

    args = _args(split="s", step=2, endpoint="e", deployment="d", api_version="v")
    params = az.AzureCallParams(temperature=0.5, top_p=0.8, max_output_tokens=10, request_timeout=5)
    row = az._build_result_row(
        problem="prob",
        gold_answer="gold",
        sample_idx=0,
        text="t",
        finish_reason="stop",
        usage={"raw": 1},
        args=args,
        call_params=params,
    )
    assert row["base"] == "prob"
    assert row["pass1"]["pred_answer_canon"] == "canon-answer"
    assert row["pass1"]["is_correct_pred"] is False
    assert row["usage"]["usage_built"]["raw"] == 1


def test_generate_samples_loop(monkeypatch, tmp_path):
    dataset = [("prob1", "gold1", 0), ("prob2", "gold2", 1)]
    monkeypatch.setattr(az, "_prepare_dataset", lambda args, outpath: (dataset, {}))
    monkeypatch.setattr(az, "iter_math_gateway_samples", lambda ds, num_samples, existing: ds)

    calls = []
    monkeypatch.setattr(
        az,
        "call_with_gateway_retries",
        lambda fn, args, logger, sample_idx, problem_snippet: calls.append(sample_idx)
        or ("text", "fin", {"n": sample_idx}),
    )

    rows = []
    monkeypatch.setattr(az, "append_jsonl_row", lambda path, row: rows.append((path, row)))
    monkeypatch.setattr(
        az, "_build_result_row", lambda **kwargs: {"k": kwargs["problem"], "sample_idx": kwargs["sample_idx"]}
    )

    args = _args(output_dir=str(tmp_path), deployment="dep")
    params = az.AzureCallParams(temperature=0.5, top_p=0.9, max_output_tokens=5, request_timeout=1)
    out_path = str(tmp_path / "out.jsonl")
    az._generate_samples(client="c", uses_v1=False, args=args, call_params=params, output_path=out_path)

    assert rows[0][0] == out_path and rows[-1][1]["sample_idx"] == 1
    assert set(calls) == {0, 1}
