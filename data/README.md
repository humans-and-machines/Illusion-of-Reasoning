# Data directory

This tree contains the code to build curated datasets used in the project. Generated artifacts are written to `results/` (or a user-specified output path) — keep this folder clean of large outputs.

## Contents
- `human_assessment_questions.txt`: Prompt/assessment questions used for manual labeling; static reference file.
- `car_park/`: Rush Hour dataset builder (`build_rush_small_balanced.py`).
- `crossword/`: Crossword dataset builders (`make_hf_crossword_dataset.py`, `csv_to_hf_crossword_dataset.py`) and helpers.

## How this ties to the paper
The Illusion-of-Reasoning study probes mid-trace “Aha!” shifts across three lenses: representational change (cryptic crosswords), symbolic reasoning (math), and spatial planning (Rush Hour). This folder generates the datasets for those benchmarks:
- Cryptic crosswords: normalize/push clue–answer pairs for shift detection and entropy-based interventions.
- Math: canonicalize openR1/MATH-style problems so traces can be scored automatically.
- Rush Hour: synthesize and optimally solve $4{\times}4/5{\times}5/6{\times}6$ boards, bucketed by difficulty, to study mid-move reconsideration.

## Prerequisites
- Python 3.9+ with the dependencies from `environment.yml` (or install the minimal runtime dependencies below).
- For Hugging Face pushes: a valid `HF_TOKEN` or authenticated `huggingface_hub` login.
- Packages: `datasets`, `tqdm`, `huggingface_hub`, and for crosswords `pandas` when converting CSVs.

## Rush Hour (car_park) — generate balanced datasets
The main entry point is `data/car_park/build_rush_small_balanced.py`, which can enumerate 4×4/5×5 puzzles and sample 6×6 puzzles, solve them optimally, label difficulty, optionally balance, split, and write JSONL files.

Example:
```bash
python data/car_park/build_rush_small_balanced.py \
  --sizes 4 5 6 \
  --out_dir results/rush-balanced \
  --split 0.8,0.1,0.1 \
  --difficulty_scheme fixed \
  --push_to_hub --dataset_id your-username/rush-balanced
```
Shortcut helper:
- `data/car_park/build.sh` wraps the above with env overrides and optional push.
  - Download/build locally: `bash data/car_park/build.sh`
  - Push to HF: `PUSH_TO_HUB=1 DATASET_ID=your-username/rush-balanced bash data/car_park/build.sh`
  - Customize knobs via env (examples): `SIZES="4 5 6" LIMIT_PER_SIZE=200000 SAMPLE_SIZES="6" TARGET_PER_SIZE="6:200000" MAX_PIECES_PER_SIZE="4:8,5:10,6:12" MIN_EMPTIES_PER_SIZE="4:1,5:2,6:3" MAX_NODES=500000 BALANCE_PER_SIZE=1`.
Key flags:
- `--sample_sizes 6` to sample instead of enumerate the listed sizes (useful for 6×6).
- `--target_per_size 6:50000` to set per-size row targets when sampling.
- `--balance_per_size` to downsample to equal easy/medium/hard per size.

Outputs: `train.jsonl`, `validation.jsonl`, `test.jsonl`, plus a README in the chosen `--out_dir`.

## Crosswords — convert bundles or CSVs to HF datasets
- Dryad/Cryptonite bundles: `data/crossword/make_hf_crossword_dataset.py`
```bash
python data/crossword/make_hf_crossword_dataset.py \
  --in_file disjoint.json \  # or cryptonite-official-split.zip
  --hub_user your-username \
  --max_len 0 \
  --skip_push    # omit to push to HF
```

- Simple CSVs: `data/crossword/csv_to_hf_crossword_dataset.py`
```bash
python data/crossword/csv_to_hf_crossword_dataset.py \
  --in_csv path/to/clues.csv \
  --hub_user your-username \
  --name mini-crosswords \
  --split 0.8,0.1,0.1 \
  --skip_push    # omit to push to HF
```

Both scripts emit HF `DatasetDict` splits and can push to the Hub or save locally under `hf_datasets/`.

### Raw Cryptonite bundle helper
- `data/crossword/build_cryptonite_official.sh` downloads the original `cryptonite-official-split.zip` from the upstream Cryptonite repo and (optionally) converts/pushes it to HF via `make_hf_crossword_dataset.py`.
  - Defaults: downloads to `data/crossword/raw/cryptonite-official-split.zip`, uses `HUB_USER` for pushes, `SKIP_PUSH=1` to keep local.
  - Examples:
    - Download only: `RUN_CONVERT=0 bash data/crossword/build_cryptonite_official.sh`
    - Download + build locally: `HUB_USER=your-hf-handle SKIP_PUSH=1 bash data/crossword/build_cryptonite_official.sh`
    - Download + build + push: `HUB_USER=your-hf-handle SKIP_PUSH=0 bash data/crossword/build_cryptonite_official.sh`

## Inspecting examples
Use `data/crossword/sample_hf_outputs.py` to peek at a local or Hub-hosted crossword dataset:
```bash
python data/crossword/sample_hf_outputs.py --repo your-username/Guardian-disjoint_word_init --n 5
```
