# Training package reference

This document captures the current responsibilities in `src/training/` so you
have a stable map while reorganising the code. The goal is to describe what
lives where today, not the desired end-state.

## CLI entry points

- `cli/grpo.py` – Thin public CLI (re-exported via `src/training/grpo.py`)
  that boots `trl.TrlParser`, merges
  dataclass configs (reward/chat/wandb/Hub), and forwards the parsed arguments
  to `grpo_impl.main`. Nothing training-specific happens here beyond argument
  plumbing, so renames should keep this import-friendly surface.
- `generate.py` – Stand-alone distilabel helper that builds a `Pipeline`
  around `distilabel.llms.OpenAILLM`. It is effectively a CLI too (argparse
  under `if __name__ == "__main__"`), so it makes sense to group it with the
  other entrypoints when restructuring.

## Runtime stack

- `runtime/main.py` – The real training driver (line 1). Sets logging,
  seeds, mutates `TrainingArguments` (e.g., `return_reward`, `steps_per_generation`),
  builds the tokenizer/model, prepares datasets, reward functions, callbacks,
  replay buffer, and instantiates `GRPOTrainerReplay`. All orchestration logic
  (Easy-pool mixing, replay mixing, trainer wiring) lives here.
- `grpo_runtime.py` / `grpo_runtime_impl.py` / `grpo_runtime_impl_full.py` –
  Compatibility shims that re-export `runtime.main` so older imports continue to work.
- `runtime/env.py` – Shared environment patches and heavy imports (torch,
  deepspeed, accelerate, wandb, transformers). Handles ZeRO pickling overrides
  and defaults like `NLTK_DATA`. Any changes to runtime dependencies should be
  staged here first.

## Data layer

- `grpo_dataset.py` – Dataset utilities (line 1). Converts raw dataset rows
  into chat-format prompts, strips stray `<think>/<answer>` tags, injects
  system prompts, augments metadata, estimates prompt token lengths, and loads
  EASY pools (`_load_easy_pool`). All dataset mapping logic consumed by
  `_prepare_datasets` in the runtime comes from this module.
- `utils/replay_dataset.py` and `utils/replay_buffer.py` – Provide the
  `ReplayMixDataset` wrapper and UCB-based `ReplayBuffer` used by the runtime
  to blend fresh samples with replayed ones. `ReplayMixDataset` handles token
  buffering and mix scheduling; `ReplayBuffer` manages add/sample/update with
  prompt de-duplication and running stats (line 1 in `replay_buffer.py`).

## Rewards

- `rewards.py` – Thin registry (line 1) that exposes `get_reward_funcs`,
  mapping friendly names to implementations in `rewards_core.py` and
  `rush_rewards.py`. Keeps signatures backward compatible with legacy callers.
- `rewards_core.py` – Shared reward logic for crossword/math accuracy as well
  as Rush basics that still fit in the file. Math/crossword canonicalisation,
  accuracy scoring, and helper utilities live here.
- `rush_rewards.py` – Factored-out Rush Hour reward helpers (line 1). Includes
  token parsing, canonicalisation, and shaped/exact scoring that used to bloat
  `rewards_core.py`.
- `grpo_rewards_router.py` – Task-aware helpers that wrap reward functions so
  they can operate on nested (B×K) completions. Also infers task labels from
  script args/environment hints and adapts reference answers.

## Utilities and callbacks

- `utils/callbacks.py` – Trainer callbacks for push-to-hub,
  success-caching (scraping textual logs into the replay buffer), and any
  future hooks. Includes stubs so the module can be imported without
  transformers installed.
- `utils/hub.py`, `utils/evaluation.py`, `utils/model_utils.py`, etc. – Support
  modules used by callbacks and runtime wiring (e.g., pushing revisions, firing
  benchmark jobs, building models/tokenizers, configuring wandb logging).
- `grpo_trainer_replay_impl.py`, `grpo_trainer_replay_support.py` – Custom
  trainer subclass plus auxiliary dataclasses (`ReplaySettings`,
  `TemperatureSchedule`, `LossLoggingCallback`). Consumed exclusively by the
  runtime builder.

Use this file as a checklist when you start moving pieces into subpackages:
each bullet should have a new home after the re-org, and the compatibility
shims can shrink once imports are updated.

## Target layout sketch

Before touching imports, the planned directory split is:

```
src/training/
├── cli/
│   ├── grpo.py              # thin entrypoints → import runtime
│   ├── sft.py               # future SFT CLI (currently implicit)
│   └── generate.py          # argparse wrapper around pipelines.generate
├── configs/
│   └── __init__.py          # exposes ScriptArguments + GRPO/SFT configs
├── runtime/
│   ├── main.py              # former grpo_runtime_main
│   ├── env.py               # former grpo_runtime_env
│   ├── trainer.py           # GRPOTrainerReplay + support dataclasses
│   ├── replay_impl.py       # replay buffer wiring helpers
│   └── __init__.py          # public surface (main)
├── data/
│   ├── dataset.py           # former grpo_dataset
│   ├── easy_pool.py         # _load_easy_pool extraction
│   ├── replay_dataset.py    # existing utils wrapper renamed here
│   └── __init__.py
├── rewards/
│   ├── registry.py          # former rewards.get_reward_funcs
│   ├── core.py              # rewards_core
│   ├── rush.py              # rush_rewards
│   ├── router.py            # grpo_rewards_router helpers
│   └── __init__.py
├── pipelines/
│   ├── distilabel.py        # helper currently embedded in generate.py
│   └── __init__.py
└── utils/
    ├── callbacks.py
    ├── hub.py
    ├── evaluation.py
    ├── model_utils.py
    └── ...
```

Key principles for the move:

- Keep `src/training/__init__.py` importing the new package surfaces (e.g.,
  `from .cli.grpo import main as grpo_main`) so external callers have a stable
  path while the tree changes.
- `cli/` modules should only parse arguments and hand over to `runtime.main`.
- `runtime/` owns trainer construction, replay wiring, and any stateful
  machinery (`GRPOTrainerReplay`, callbacks, buffers).
- `data/` isolates dataset munging so runtime code can import from
  `training.data.dataset`.
- `rewards/` provides a central registry plus submodules for math/crossword/
  rush logic; router helpers wrap the callable outputs for nested completions.
- `pipelines/` keeps distilabel/vLLM utilities and can grow to house recipe
  generation or evaluation flows.

As files get moved, use this sketch to tick items off and ensure every module
has a clearly defined home.
