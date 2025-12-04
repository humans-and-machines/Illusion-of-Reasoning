"""Shared runtime imports and environment patches for GRPO training.

This module centralises side-effectful configuration (torch/ZeRO pickling,
NLTK data path, etc.) so that the higher-level training modules can remain
small and focused.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional


if TYPE_CHECKING:
    import datasets as datasets_module
    import torch as torch_module
    import torch.distributed as dist_module
    import transformers as transformers_module
    from torch import serialization as torch_serialization_module
    from torch.utils.data import DataLoader as DataLoaderType
    from torch.utils.data import RandomSampler as RandomSamplerType
    from transformers import TrainerCallback as TrainerCallbackType
    from transformers import set_seed as set_seed_fn
    from transformers.trainer_utils import get_last_checkpoint as get_last_checkpoint_fn

    import wandb as wandb_module
    from accelerate.state import AcceleratorState as AcceleratorStateType
    from deepspeed.runtime.zero.config import ZeroStageEnum as ZeroStageEnumType
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus as ZeroParamStatusType
    from trl import get_peft_config as get_peft_config_fn
    from trl.trainer.grpo_trainer import GRPOTrainer as GRPOTrainerCls

_EXPORT_CACHE: Dict[str, Optional[object]] = {}
Loader = Callable[[], Mapping[str, Optional[object]]]


class _StubTrainerCallback:  # pylint: disable=too-few-public-methods
    """Fallback TrainerCallback used when transformers is unavailable."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _stub_set_seed(_seed: int | None = None) -> None:  # pragma: no cover - lint fallback
    return None


def _stub_get_last_checkpoint(_path: str | None = None) -> None:  # pragma: no cover - lint fallback
    return None


def _stub_get_peft_config(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - lint fallback
    return None


class _StubGRPOTrainer:  # pylint: disable=too-few-public-methods
    """Fallback GRPOTrainer used when trl is unavailable."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _prepare_inputs(self, inputs):  # pragma: no cover - stubbed for tests
        return inputs

    def _generate_and_score_completions(self, inputs):  # pragma: no cover - stubbed for tests
        return inputs


def _register_exports(exports: Mapping[str, Optional[object]]) -> None:
    _EXPORT_CACHE.update(exports)
    globals().update(exports)


def _load_datasets() -> Mapping[str, Optional[object]]:
    try:
        datasets_module = importlib.import_module("datasets")  # type: ignore
    except ImportError:  # pragma: no cover - optional in lint environments
        datasets_module = None
    return {"datasets": datasets_module}


def _load_wandb() -> Mapping[str, Optional[object]]:
    try:
        wandb_module = importlib.import_module("wandb")  # type: ignore
    except ImportError:  # pragma: no cover - optional in lint environments
        wandb_module = None
    return {"wandb": wandb_module}


def _load_accelerate() -> Mapping[str, Optional[object]]:
    if "AcceleratorState" in _EXPORT_CACHE:
        return {"AcceleratorState": _EXPORT_CACHE["AcceleratorState"]}

    try:
        accelerate_state = importlib.import_module("accelerate.state")  # type: ignore
        accelerator_state_type = getattr(accelerate_state, "AcceleratorState")
    except ImportError:  # pragma: no cover - optional in lint environments
        accelerator_state_type = None  # type: ignore[assignment]

    return {"AcceleratorState": accelerator_state_type}


def _load_deepspeed_bundle() -> Mapping[str, Optional[object]]:
    if "ZeroStageEnum" in _EXPORT_CACHE and "ZeroParamStatus" in _EXPORT_CACHE:
        return {
            "ZeroStageEnum": _EXPORT_CACHE["ZeroStageEnum"],
            "ZeroParamStatus": _EXPORT_CACHE["ZeroParamStatus"],
        }

    try:
        zero_config = importlib.import_module("deepspeed.runtime.zero.config")  # type: ignore
        zero_partition = importlib.import_module("deepspeed.runtime.zero.partition_parameters")  # type: ignore
        zero_stage_enum_type = getattr(zero_config, "ZeroStageEnum")
        zero_param_status_type = getattr(zero_partition, "ZeroParamStatus")
    except ImportError:  # pragma: no cover - optional in lint environments
        zero_stage_enum_type = zero_param_status_type = None  # type: ignore[assignment]

    return {
        "ZeroStageEnum": zero_stage_enum_type,
        "ZeroParamStatus": zero_param_status_type,
    }


def _load_transformers_bundle() -> Mapping[str, Optional[object]]:
    if (
        "transformers" in _EXPORT_CACHE
        and "TrainerCallback" in _EXPORT_CACHE
        and "set_seed" in _EXPORT_CACHE
        and "get_last_checkpoint" in _EXPORT_CACHE
    ):
        return {
            "transformers": _EXPORT_CACHE["transformers"],
            "TrainerCallback": _EXPORT_CACHE["TrainerCallback"],
            "set_seed": _EXPORT_CACHE["set_seed"],
            "get_last_checkpoint": _EXPORT_CACHE["get_last_checkpoint"],
        }

    try:
        transformers_module = importlib.import_module("transformers")  # type: ignore
        trainer_callback_type = getattr(transformers_module, "TrainerCallback")
        set_seed_fn = getattr(transformers_module, "set_seed")
        trainer_utils = importlib.import_module("transformers.trainer_utils")  # type: ignore
        get_last_checkpoint_fn = getattr(trainer_utils, "get_last_checkpoint")
    except (ImportError, AttributeError):  # pragma: no cover - optional in lint or stubbed envs
        transformers_module = None  # type: ignore[assignment]
        trainer_callback_type = _StubTrainerCallback  # type: ignore[assignment]
        set_seed_fn = _stub_set_seed  # type: ignore[assignment]
        get_last_checkpoint_fn = _stub_get_last_checkpoint  # type: ignore[assignment]

    return {
        "transformers": transformers_module,
        "TrainerCallback": trainer_callback_type,
        "set_seed": set_seed_fn,
        "get_last_checkpoint": get_last_checkpoint_fn,
    }


def _patch_torch_serialization(
    torch_module: Any,
    torch_serialization_module: Any,
    zero_stage_enum: Optional[object],
    zero_param_status: Optional[object],
) -> None:
    setattr(torch_module, "serialization", torch_serialization_module)

    torch_default_weights_attr = "_default_weights_only"
    setattr(torch_serialization_module, torch_default_weights_attr, False)

    if zero_stage_enum is not None and zero_param_status is not None:
        torch_serialization_module.add_safe_globals(
            {
                ("deepspeed.runtime.zero.config", "ZeroStageEnum"): zero_stage_enum,
                (
                    "deepspeed.runtime.zero.partition_parameters",
                    "ZeroParamStatus",
                ): zero_param_status,
            }
        )

    orig_load = torch_module.load
    torch_module.load = functools.partial(orig_load, weights_only=False)  # type: ignore[arg-type]


def _load_torch_bundle() -> Mapping[str, Optional[object]]:
    if (
        "torch" in _EXPORT_CACHE
        and "dist" in _EXPORT_CACHE
        and "torch_serialization" in _EXPORT_CACHE
        and "DataLoader" in _EXPORT_CACHE
        and "RandomSampler" in _EXPORT_CACHE
    ):
        return {
            "torch": _EXPORT_CACHE["torch"],
            "dist": _EXPORT_CACHE["dist"],
            "torch_serialization": _EXPORT_CACHE["torch_serialization"],
            "DataLoader": _EXPORT_CACHE["DataLoader"],
            "RandomSampler": _EXPORT_CACHE["RandomSampler"],
        }

    try:
        torch_module = importlib.import_module("torch")  # type: ignore
    except ImportError:  # pragma: no cover - optional in lint environments
        return {
            "torch": None,
            "dist": None,
            "torch_serialization": None,
            "DataLoader": None,
            "RandomSampler": None,
        }
    try:
        dist_module = importlib.import_module("torch.distributed")  # type: ignore
        torch_serialization_module = importlib.import_module("torch.serialization")  # type: ignore
        torch_utils_data = importlib.import_module("torch.utils.data")  # type: ignore
        data_loader_type = getattr(torch_utils_data, "DataLoader")
        random_sampler_type = getattr(torch_utils_data, "RandomSampler")
    except ImportError:  # pragma: no cover - optional in lint environments
        dist_module = sys.modules.get("torch.distributed")
        torch_serialization_module = sys.modules.get("torch.serialization")
        torch_utils_data = sys.modules.get("torch.utils.data")
        data_loader_type = getattr(torch_utils_data, "DataLoader", None) if torch_utils_data else None
        random_sampler_type = getattr(torch_utils_data, "RandomSampler", None) if torch_utils_data else None

    deepspeed_exports = _load_deepspeed_bundle()
    _register_exports(deepspeed_exports)

    if torch_serialization_module is not None:
        _patch_torch_serialization(
            torch_module,
            torch_serialization_module,
            deepspeed_exports.get("ZeroStageEnum"),
            deepspeed_exports.get("ZeroParamStatus"),
        )

    return {
        "torch": torch_module,
        "dist": dist_module,
        "torch_serialization": torch_serialization_module,
        "DataLoader": data_loader_type,
        "RandomSampler": random_sampler_type,
    }


def _load_trl_bundle() -> Mapping[str, Optional[object]]:
    if "get_peft_config" in _EXPORT_CACHE and "GRPOTrainer" in _EXPORT_CACHE:
        return {
            "get_peft_config": _EXPORT_CACHE["get_peft_config"],
            "GRPOTrainer": _EXPORT_CACHE["GRPOTrainer"],
        }

    try:
        trl_module = importlib.import_module("trl")  # type: ignore
        get_peft_config_fn = getattr(trl_module, "get_peft_config")
        grpo_trainer = importlib.import_module("trl.trainer.grpo_trainer")  # type: ignore
        grpo_trainer_cls = getattr(grpo_trainer, "GRPOTrainer")
    except ImportError:  # pragma: no cover - optional in lint environments
        get_peft_config_fn = _stub_get_peft_config  # type: ignore[assignment]
        grpo_trainer_cls = _StubGRPOTrainer  # type: ignore[assignment]

    return {
        "get_peft_config": get_peft_config_fn,
        "GRPOTrainer": grpo_trainer_cls,
    }


_LAZY_LOADERS: Dict[str, Loader] = {
    "datasets": _load_datasets,
    "torch": _load_torch_bundle,
    "dist": _load_torch_bundle,
    "torch_serialization": _load_torch_bundle,
    "DataLoader": _load_torch_bundle,
    "RandomSampler": _load_torch_bundle,
    "transformers": _load_transformers_bundle,
    "TrainerCallback": _load_transformers_bundle,
    "set_seed": _load_transformers_bundle,
    "get_last_checkpoint": _load_transformers_bundle,
    "wandb": _load_wandb,
    "AcceleratorState": _load_accelerate,
    "ZeroStageEnum": _load_deepspeed_bundle,
    "ZeroParamStatus": _load_deepspeed_bundle,
    "get_peft_config": _load_trl_bundle,
    "GRPOTrainer": _load_trl_bundle,
}

# ─────────────────── NLTK data path (WordNet) ───────────────────────────
NLTK_DATA_DEFAULT = Path(__file__).resolve().parents[2] / ".nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DEFAULT))

if TYPE_CHECKING:
    datasets: Optional[datasets_module]
    torch: Optional[torch_module]
    dist: Optional[dist_module]
    torch_serialization: Optional[torch_serialization_module]
    DataLoader: Optional[DataLoaderType]
    RandomSampler: Optional[RandomSamplerType]
    transformers: Optional[transformers_module]
    TrainerCallback: TrainerCallbackType
    set_seed: set_seed_fn
    get_last_checkpoint: get_last_checkpoint_fn
    wandb: Optional[wandb_module]
    AcceleratorState: Optional[AcceleratorStateType]
    ZeroStageEnum: Optional[ZeroStageEnumType]
    ZeroParamStatus: Optional[ZeroParamStatusType]
    get_peft_config: get_peft_config_fn
    GRPOTrainer: GRPOTrainerCls

_EXPORTED_NAMES = (
    "datasets",
    "torch",
    "dist",
    "transformers",
    "wandb",
    "DataLoader",
    "RandomSampler",
    "AcceleratorState",
    "ZeroStageEnum",
    "ZeroParamStatus",
    "get_last_checkpoint",
    "get_peft_config",
    "GRPOTrainer",
    "TrainerCallback",
    "set_seed",
)

# Provide stub bindings for __all__ so static analyzers see the exports. They
# are removed immediately to preserve __getattr__-based lazy loading at runtime.
# pylint: disable=invalid-name
datasets = torch = dist = transformers = wandb = None  # type: ignore[assignment]
DataLoader = RandomSampler = None  # type: ignore[assignment]
AcceleratorState = ZeroStageEnum = ZeroParamStatus = None  # type: ignore[assignment]
get_last_checkpoint = get_peft_config = None  # type: ignore[assignment]
GRPOTrainer = TrainerCallback = set_seed = None  # type: ignore[assignment]
# pylint: enable=invalid-name

__all__ = list(_EXPORTED_NAMES)


# Remove the placeholders so attribute access still flows through __getattr__.
def _clear_placeholder_exports() -> None:
    for _export_name in _EXPORTED_NAMES:
        globals().pop(_export_name, None)


_clear_placeholder_exports()


def __getattr__(name: str) -> Optional[object]:
    if name in _EXPORT_CACHE:
        return _EXPORT_CACHE[name]

    if name not in _LAZY_LOADERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    exports = _LAZY_LOADERS[name]()
    _register_exports(exports)
    return _EXPORT_CACHE.get(name)


def __dir__() -> list[str]:
    return sorted(__all__)
