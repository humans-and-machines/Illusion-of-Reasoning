# Recipes

Ready-to-run GRPO configs live here. Each YAML under `recipes/<model>/<task>/` can be passed directly to `src/training/grpo.py` (optionally via the SLURM launchers in `scripts/training/`). The `accelerate_configs/` folder contains a DeepSpeed Zero-3 template for multi-GPU runs.

## Layout

- `Qwen2.5-1.5B-Instruct/grpo/`
  - `config_math.yaml`: OpenR1-Math 220k, pushes to `od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1`.
  - `config_crosswords.yaml`: Guardian Cryptonite split, crossword-specific reward shaping.
  - `config_carpark.yaml`: Rush Hour (car park) solver with shaped reward and replay/caching callbacks.
- `Qwen2.5-7B-Instruct/grpo/config_math.yaml`: 7B math variant (lower per-device batch; same rewards).
- `Llama3.1-8B-instruct/grpo/config_math.yaml`: Llama 3.1 math variant (Zero-3 friendly checkpoint saving).
- `accelerate_configs/zero3.yaml`: Accelerate + DeepSpeed Zero-3 base; set `num_processes` to your training GPU count (scripts do this automatically).

## How to launch training

1) Spin up a vLLM server (if using `use_vllm: true` in the recipe). In `scripts/training/*grpo*.slurm` this runs on GPU 0; training uses the remaining GPUs.
2) Launch GRPO with Accelerate. Example (math on Qwen 1.5B):

```bash
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/training/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml \
  --use_vllm \
  --run_name debug-math \
  --overwrite_output_dir false
```

3) Monitor logs: the SLURM launchers write to `logs/liv_vllm_*.log` (server) and `logs/liv_train_*.log` (trainer). W&B reporting is enabled in all recipes; set `WANDB_MODE=disabled` to turn it off.

## Tips

- Adjust `per_device_train_batch_size`, `gradient_accumulation_steps`, and `num_generations` to match available memory. Zero-3 offload settings are in `accelerate_configs/zero3.yaml`.
- Hugging Face auth is required when `push_to_hub: true` or when downloading gated models; export `HUGGING_FACE_HUB_TOKEN`.
- To change the output location, edit `output_dir` in the recipe or pass `--output_dir` to `grpo.py`.
- The Slurm launchers (`scripts/training/training-*.slurm`) show the exact environment variables and cache dirs we used; use them as templates for local or cluster runs.
