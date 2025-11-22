# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for building and running distilabel-based generation pipelines."""

from dataclasses import dataclass, field
import importlib
from typing import Optional


@dataclass
class GenerationSettings:
    """Grouped generation hyperparameters for distilabel."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 8192
    num_generations: int = 1


@dataclass
class DistilabelPipelineConfig:
    """Configuration for a distilabel text-generation pipeline."""

    prompt_column: Optional[str] = None
    prompt_template: str = "{{ instruction }}"
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    input_batch_size: int = 64
    client_replicas: int = 1
    timeout: int = 900
    retries: int = 0


def _require_distilabel_deps():
    """Import distilabel modules lazily and return their core classes.

    This avoids import-time failures when distilabel is not installed while still
    raising a clear runtime error if the script is executed without the package.
    """
    try:
        llms_mod = importlib.import_module("distilabel.llms")
        pipeline_mod = importlib.import_module("distilabel.pipeline")
        steps_mod = importlib.import_module("distilabel.steps")
        tasks_mod = importlib.import_module("distilabel.steps.tasks")
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "This script requires the 'distilabel' package. "
            "Install it with `pip install distilabel`.",
        ) from exc

    return (
        llms_mod.OpenAILLM,
        pipeline_mod.Pipeline,
        steps_mod.StepResources,
        tasks_mod.TextGeneration,
    )


def build_distilabel_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    config: Optional[DistilabelPipelineConfig] = None,
):
    """
    Build and configure a distilabel :class:`Pipeline` for text generation.

    :param model: Model name or identifier understood by the distilabel/OpenAI LLM client.
    :param base_url: Base URL of the vLLM (or OpenAI-compatible) server.
    :param config: Optional :class:`DistilabelPipelineConfig` describing prompt,
        batching, and generation settings. When ``None``, a default config is used.
    :returns: A configured distilabel :class:`Pipeline` object with a single
        :class:`TextGeneration` step attached.
    :raises RuntimeError: If the ``distilabel`` package is not installed.
    """
    if config is None:
        config = DistilabelPipelineConfig()

    (
        openai_llm_cls,
        pipeline_cls,
        step_resources_cls,
        text_generation_cls,
    ) = _require_distilabel_deps()

    generation_kwargs = {"max_new_tokens": config.generation.max_new_tokens}

    if config.generation.temperature is not None:
        generation_kwargs["temperature"] = config.generation.temperature

    if config.generation.top_p is not None:
        generation_kwargs["top_p"] = config.generation.top_p

    with pipeline_cls().ray() as dl_pipeline:
        text_generation_cls(
            llm=openai_llm_cls(
                base_url=base_url,
                api_key="something",
                model=model,
                timeout=config.timeout,
                max_retries=config.retries,
                generation_kwargs=generation_kwargs,
            ),
            template=config.prompt_template,
            input_mappings=(
                {"instruction": config.prompt_column}
                if config.prompt_column is not None
                else {}
            ),
            input_batch_size=config.input_batch_size,
            num_generations=config.generation.num_generations,
            group_generations=True,
            resources=step_resources_cls(replicas=config.client_replicas),
        )

    return dl_pipeline


if __name__ == "__main__":
    import argparse

    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as datasets_import_error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "This script requires the 'datasets' package. "
            "Install it with `pip install datasets`.",
        ) from datasets_import_error

    load_dataset = datasets_module.load_dataset

    parser = argparse.ArgumentParser(
        description=(
            "Run distilabel pipeline for generating responses with DeepSeek R1"
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas for parallel processing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0)",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to HF Hub",
    )

    args = parser.parse_args()

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(
        f"Loading '{args.hf_dataset}' "
        f"(config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) "
        "dataset...",
    )
    dataset = load_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
    )
    print("Dataset loaded!")

    pipeline_config = DistilabelPipelineConfig(
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template,
        generation=GenerationSettings(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_generations=args.num_generations,
        ),
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )
    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        config=pipeline_config,
    )

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")

    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")
