"""Utility script to download required NLTK data into a local .nltk_data directory."""

import os
from pathlib import Path


try:
    import nltk  # type: ignore[import-error]
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("The NLTK downloader utility requires the 'nltk' package to be installed.") from exc

TARGET = str(Path(__file__).resolve().parents[3] / ".nltk_data")
os.makedirs(TARGET, exist_ok=True)

# Make sure nltk looks there first
nltk.data.path.insert(0, TARGET)

# Packages you actually use (and a few compat ones):
PKGS = [
    "punkt",  # tokenizer
    "punkt_tab",  # newer NLTK splits this out
    "words",  # english word list
    "averaged_perceptron_tagger",  # older tagger name
    "averaged_perceptron_tagger_eng",  # newer tagger name (>= 3.8)
    # optionally:
    # "wordnet", "omw-1.4"
]

for package in PKGS:
    try:
        nltk.data.find(package if "/" in package else f"corpora/{package}")
        print(f"[nltk] already have {package}")
    except LookupError:
        print(f"[nltk] downloading {package} -> {TARGET}")
        nltk.download(package, download_dir=TARGET, quiet=True)

print("Done.")
