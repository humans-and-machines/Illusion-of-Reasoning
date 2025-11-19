"""Helpers for registering and running LightEval benchmarks via Slurm/vLLM."""

import base64
import os
import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, ModelConfig, SFTConfig


# We need a special environment setup to launch vLLM from within Slurm
# training jobs.
# - Reference implementation: see the LightEval Slurm launcher in the
#   `huggingface/brrr` repository (lighteval/one_job_runner.py).
user_home_directory = os.path.expanduser("~")
_VLLM_ENV_BOOTSTRAP = "for f in /etc/profile.d/*.sh; do source $f; done; "
_VLLM_HOME_EXPORT = f"export HOME={user_home_directory}; "
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    _VLLM_ENV_BOOTSTRAP + _VLLM_HOME_EXPORT + "sbatch ",
]


def register_lighteval_task(
    configs: Dict[str, str],
    eval_suite: str,
    task_name: str,
    task_list: str,
    num_fewshot: int = 0,
):
    """Register a LightEval task configuration.

    - Core tasks can be added from this table:
      https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - Custom tasks that require their own metrics / scripts should live in
      ``scripts/evaluation/extended_lighteval_tasks``.

    Args:
        configs (Dict[str, str]): The dictionary to store the task configuration.
        eval_suite (str): The evaluation suite.
        task_name (str): The name of the task.
        task_list (str): The comma-separated list of tasks in the format
            ``extended|{task_name}|{num_fewshot}|0`` or
            ``lighteval|{task_name}|{num_fewshot}|0``.
        num_fewshot (int, optional): The number of few-shot examples.
    """
    # Format task list in lighteval format
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


LIGHTEVAL_TASKS = {}

register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)


def get_lighteval_tasks():
    """Return the list of registered LightEval task names."""
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str,
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """Submit a single LightEval benchmark job via Slurm/vLLM."""
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    # For large models >= 30b params or those running the MATH benchmark, we
    # need to shard them across the GPUs to avoid OOM.
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 2  # Hack while cluster is full
        tensor_parallel = False

    cmd = VLLM_SLURM_PREFIX.copy()
    job_name = f"or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}"
    cmd_args = [
        f"--gres=gpu:{num_gpus}",
        f"--job-name={job_name}",
        "slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        # encode to base64 to avoid issues with special characters
        # we decode in the sbatch script
        prompt_encoded = base64.b64encode(training_args.system_prompt.encode()).decode()
        cmd_args.append(prompt_encoded)
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """Launch all requested benchmarks for a trained model."""
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a
        # `chat` option that just evaluates on `ifeval` and `mt_bench`, etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
