import os
import sys

# Ensure the top-level package `src` is importable.
# We want the repository root (parent of this docs/ directory) on sys.path,
# not the src/ directory itself, so that imports like `import src.training` work.
sys.path.insert(0, os.path.abspath(".."))

project = "Illusion-of-Reasoning"
author = "Illusion-of-Reasoning contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Some training/runtime dependencies are heavy or optional; mock them so that
# autodoc can import modules without requiring the full training stack (and
# without pulling in third-party docstrings that confuse docutils).
autodoc_mock_imports = [
    "datasets",
    "httpx",
    "trl",
    "torch",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",
    "torch.utils.data.sampler",
    "accelerate",
    "accelerate.state",
    "deepspeed",
    "deepspeed.runtime",
    "deepspeed.runtime.zero",
    "deepspeed.runtime.zero.config",
    "deepspeed.runtime.zero.partition_parameters",
    "transformers",
    "transformers.trainer",
    "transformers.trainer_callback",
    "transformers.utils",
    "transformers.utils.generic",
    "wandb",
    "openai",
    "pydantic",
    "portkey_ai",
    "portkey_ai.api_resources",
    "portkey_ai.api_resources.apis",
    "portkey_ai.api_resources.types",
]

# Avoid documenting re-exported third-party classes whose docstrings are
# defined in external projects (they often contain cross-references that
# produce docutils errors in this project’s build).
autodoc_default_options = {
    "exclude-members": (
        "DataLoader,RandomSampler,AcceleratorState,ZeroStageEnum,ZeroParamStatus,"
        "Trainer,TrainerCallback,PaddingStrategy,GRPOTrainer,Line2D"
    )
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
