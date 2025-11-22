"""Dataset loading utilities for SFT training."""

import logging
from importlib import import_module
from typing import Any

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def _require_datasets_module():
    """Import and return the `datasets` module, raising with a clear message if missing."""
    try:
        return import_module("datasets")
    except ImportError as import_exc:  # pragma: no cover - optional dependency
        logger.error("The 'datasets' package is required: pip install datasets")
        raise import_exc


def get_dataset(args: ScriptArguments) -> Any:
    """
    Load a dataset or a mixture of datasets based on the configuration.

    :param args: Script arguments containing dataset configuration, including
        either a single ``dataset_name`` or a ``dataset_mixture`` definition.
    :returns: The loaded dataset object, or a mapping such as ``{\"train\": ds}``
        when a train/test split is created.
    :raises ValueError: If neither ``dataset_name`` nor ``dataset_mixture`` is set,
        or if a mixture configuration fails to load any datasets.
    """
    datasets_module = _require_datasets_module()

    if args.dataset_name and not args.dataset_mixture:
        logger.info("Loading dataset: %s", args.dataset_name)
        return datasets_module.load_dataset(args.dataset_name, args.dataset_config)

    if args.dataset_mixture:
        mixture = args.dataset_mixture
        logger.info(
            "Creating dataset mixture with %d datasets",
            len(mixture.datasets),
        )
        seed = mixture.seed
        datasets_list = []

        for dataset_config in mixture.datasets:
            logger.info(
                "Loading dataset for mixture: %s (config: %s)",
                dataset_config.dataset_id,
                dataset_config.config,
            )
            dataset_obj = datasets_module.load_dataset(
                dataset_config.dataset_id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                dataset_obj = dataset_obj.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                target_size = int(len(dataset_obj) * dataset_config.weight)
                dataset_obj = dataset_obj.shuffle(seed=seed).select(
                    range(target_size),
                )
                logger.info(
                    "Subsampled dataset '%s' (config: %s) with weight=%s to %d examples",
                    dataset_config.dataset_id,
                    dataset_config.config,
                    dataset_config.weight,
                    len(dataset_obj),
                )

            datasets_list.append(dataset_obj)

        if not datasets_list:
            raise ValueError("No datasets were loaded from the mixture configuration")

        combined_dataset = datasets_module.concatenate_datasets(datasets_list)
        combined_dataset = combined_dataset.shuffle(seed=seed)
        logger.info(
            "Created dataset mixture with %d examples",
            len(combined_dataset),
        )

        if mixture.test_split_size is not None:
            combined_dataset = combined_dataset.train_test_split(
                test_size=mixture.test_split_size,
                seed=seed,
            )
            logger.info(
                "Split dataset into train and test sets with test size: %s",
                mixture.test_split_size,
            )
            return combined_dataset

        return {"train": combined_dataset}

    raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
