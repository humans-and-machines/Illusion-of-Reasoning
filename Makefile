.PHONY: help install-dev install-local ensure-path venv install-venv conda-local conda-local-hook lint format test precommit docs docs-clean clean-local

help:
	@echo "Targets:"
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  install-local- user base under $(PWD)/.local"
	@echo "  ensure-path  - append $(PWD)/.local/bin to your shell rc"
	@echo "  conda-local  - create local conda env at ./openr1 via environment.yml"
	@echo "  conda-local-hook - reinstall the conda activation hook that pins caches locally"
	@echo "  venv         - create .venv in repo"
	@echo "  install-venv - install project into .venv"
	@echo "  lint         - ruff check . && pylint src"
	@echo "  format       - ruff check --fix . && isort ."
	@echo "  test         - pytest -q"
	@echo "  precommit    - pre-commit run -a"
	@echo "  docs         - build Sphinx HTML to _build/html"
	@echo "  docs-clean   - remove _build"
	@echo "  clean-local  - remove local envs/caches in this repo"

install-dev:
	pip install -e .[dev]

install-local:
	export PYTHONUSERBASE=$(PWD)/.local; \
	  python -m pip install --upgrade pip; \
	  pip install --user -e .[dev]
	@echo "Add to PATH for local scripts: export PATH=\"$(PWD)/.local/bin:$$PATH\""

ensure-path:
	bash tools/ensure_local_path.sh --apply

venv:
	python -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install-venv: venv
	. .venv/bin/activate && pip install -e .[dev]

conda-local:
	@set -euo pipefail; \
	# Avoid alias conflict between CONDA_ENVS_DIRS and CONDA_ENVS_PATH (cluster sets PATH);
	# conda 24+ errors if both are present. Prefer CONDA_ENVS_DIRS.
	unset CONDA_ENVS_PATH; \
	ROOT_DIR="$(PWD)"; \
	echo "Creating local conda env under $$ROOT_DIR/openr1"; \
	CONDARC="$$ROOT_DIR/.condarc"; \
	CONDA_PKGS_DIRS="$$ROOT_DIR/.conda_pkgs"; \
	CONDA_ENVS_DIRS="$$ROOT_DIR/.conda_envs"; \
	PIP_CACHE_DIR="$$ROOT_DIR/.pip_cache"; \
	PIP_CONFIG_FILE="$$ROOT_DIR/.pip/pip.conf"; \
	TMPDIR="$$ROOT_DIR/.tmp"; \
	mkdir -p "$$CONDA_PKGS_DIRS" "$$CONDA_ENVS_DIRS" "$$PIP_CACHE_DIR" "$$TMPDIR" "$$ROOT_DIR/.pip"; \
	if [ ! -f "$$ROOT_DIR/.pip/pip.conf" ]; then \
	  printf "[global]\ncache-dir = %s/.pip_cache\ndisable-pip-version-check = true\n" "$$ROOT_DIR" > "$$ROOT_DIR/.pip/pip.conf"; \
	fi; \
	if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then \
	  . /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh; \
	elif command -v conda >/dev/null 2>&1; then \
	  eval "$(conda shell.bash hook)"; \
	else \
	  echo "Conda not found on PATH" >&2; exit 1; \
	fi; \
	CONDARC="$$CONDARC" CONDA_PKGS_DIRS="$$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$$CONDA_ENVS_DIRS" \
	PIP_CACHE_DIR="$$PIP_CACHE_DIR" PIP_CONFIG_FILE="$$ROOT_DIR/.pip/pip.conf" TMPDIR="$$TMPDIR" \
	XDG_CACHE_HOME="$$ROOT_DIR/.xdg_cache" CUDA_CACHE_PATH="$$ROOT_DIR/.cuda_cache" \
	TORCH_HOME="$$ROOT_DIR/.torch_cache" HF_HOME="$$ROOT_DIR/.hf_home" \
	HUGGINGFACE_HUB_CACHE="$$ROOT_DIR/.hf_home/hub" HF_DATASETS_CACHE="$$ROOT_DIR/datasets_cache" \
	TRANSFORMERS_CACHE="$$ROOT_DIR/.hf_cache" WANDB_DIR="$$ROOT_DIR/wandb" \
	WANDB_CACHE_DIR="$$ROOT_DIR/.wandb_cache" WANDB_CONFIG_DIR="$$ROOT_DIR/.wandb" \
	WANDB_DATA_DIR="$$ROOT_DIR/.wandb" \
	  conda env create -p "$$ROOT_DIR/openr1" -f "$$ROOT_DIR/environment.yml"; \
	echo "✅ Env created at: $$ROOT_DIR/openr1"; \
	echo "Activate with: conda activate $$ROOT_DIR/openr1"
	@tools/install_conda_hooks.sh

conda-local-hook:
	@tools/install_conda_hooks.sh

lint:
	ruff check .
	pylint --rcfile=.pylintrc src

format:
	ruff check --fix .
	isort .

test:
	pytest -q

precommit:
	pre-commit run -a

docs:
	python -m sphinx -b html docs _build/html

docs-clean:
	rm -rf _build

clean-local:
	rm -rf openr1 .venv .local .conda_envs .conda_pkgs .pip .pip_cache .tmp \
		.torchinductor .triton wandb .wandb .wandb_cache .hf_home .hf_cache \
		.xdg_cache .cuda_cache .torch_cache datasets_cache logs docs/_build _build
