# download_nltk.py
import os
import nltk

TARGET = "/n/fs/similarity/open-r1/openr1/nltk_data"
os.makedirs(TARGET, exist_ok=True)

# Make sure nltk looks there first
nltk.data.path.insert(0, TARGET)

# Packages you actually use (and a few compat ones):
PKGS = [
    "punkt",                         # tokenizer
    "punkt_tab",                     # newer NLTK splits this out
    "words",                         # english word list
    "averaged_perceptron_tagger",    # older tagger name
    "averaged_perceptron_tagger_eng" # newer tagger name (>= 3.8)
    # optionally:
    # "wordnet", "omw-1.4"
]

for p in PKGS:
    try:
        nltk.data.find(p if "/" in p else f"corpora/{p}")
        print(f"[nltk] already have {p}")
    except LookupError:
        print(f"[nltk] downloading {p} -> {TARGET}")
        nltk.download(p, download_dir=TARGET, quiet=True)

print("Done.")
