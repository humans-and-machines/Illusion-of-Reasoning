import base64
from types import SimpleNamespace

import pytest

from src.training.utils import evaluation


def test_register_lighteval_task_formats_task_list():
    cfg = {}
    evaluation.register_lighteval_task(cfg, "suite", "task", "a,b", num_fewshot=2)
    assert cfg["task"] == "suite|a|2|0,suite|b|2|0"


def test_get_lighteval_tasks_returns_registered():
    tasks = evaluation.get_lighteval_tasks()
    assert "math_500" in tasks


def test_run_lighteval_job_builds_command(monkeypatch):
    calls = {}

    def fake_get_gpu_count(model_name, revision, num_gpus=8):
        calls["get_gpu_count"] = (model_name, revision, num_gpus)
        return 4

    def fake_param_count(repo_id):
        calls["param_count"] = repo_id
        return 10_000_000_000

    monkeypatch.setattr(evaluation, "get_gpu_count_for_vllm", fake_get_gpu_count)
    monkeypatch.setattr(evaluation, "get_param_count_from_repo_id", fake_param_count)
    monkeypatch.setattr(evaluation, "subprocess", SimpleNamespace(run=lambda cmd, check: calls.update({"cmd": cmd})))

    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        benchmarks=["math_500"],
        system_prompt="special prompt",
    )
    model_args = SimpleNamespace(trust_remote_code=False)

    evaluation.run_lighteval_job("math_500", training_args, model_args)
    assert calls["cmd"][0:3] == evaluation.VLLM_SLURM_PREFIX[:3]
    # For param_count < 30B, logic forces 2 GPUs
    assert "--gres=gpu:2" in calls["cmd"][-1]
    assert base64.b64encode(b"special prompt").decode() in calls["cmd"][-1]


def test_run_lighteval_job_uses_tensor_parallel_for_large_models(monkeypatch):
    calls = {}

    def fake_get_gpu_count(model_name, revision, num_gpus=8):
        calls["num_gpus"] = 4
        return 4

    def fake_param_count(repo_id):
        return 35_000_000_000

    monkeypatch.setattr(evaluation, "get_gpu_count_for_vllm", fake_get_gpu_count)
    monkeypatch.setattr(evaluation, "get_param_count_from_repo_id", fake_param_count)
    monkeypatch.setattr(evaluation, "subprocess", SimpleNamespace(run=lambda cmd, check: calls.update({"cmd": cmd})))

    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        benchmarks=["math_500"],
        system_prompt=None,
    )
    model_args = SimpleNamespace(trust_remote_code=True)

    evaluation.run_lighteval_job("math_500", training_args, model_args)
    # tensor_parallel True for large models, keep the original GPU count
    assert "--gres=gpu:4" in calls["cmd"][-1]
    assert calls["cmd"][-1].split()[-2] == "True"


def test_run_benchmark_jobs_expands_all(monkeypatch):
    launched = []
    monkeypatch.setattr(evaluation, "run_lighteval_job", lambda bench, ta, ma: launched.append(bench))

    training_args = SimpleNamespace(benchmarks=["all"])
    model_args = SimpleNamespace()
    evaluation.run_benchmark_jobs(training_args, model_args)
    assert set(launched) == set(evaluation.get_lighteval_tasks())

    launched.clear()
    training_args = SimpleNamespace(benchmarks=["math_500", "bogus"])
    with pytest.raises(ValueError):
        evaluation.run_benchmark_jobs(training_args, model_args)
