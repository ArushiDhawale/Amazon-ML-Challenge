import pandas as pd

def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def load_all(base="dataset/train"):
    s1 = load_source(f"{base}/{'train' if 'train' in base else 'test'}_source1.tsv")
    s2 = load_source(f"{base}/{'train' if 'train' in base else 'test'}_source2.tsv")
    s3 = load_source(f"{base}/{'train' if 'train' in base else 'test'}_source3.tsv")
    return s1, s2, s3

import re

SUFFIXES = ["inc", "incorporated", "corp", "corporation", "ltd", "limited",
            "pvt", "private", "llc", "co", "company"]

def normalize_name(s):
    s = s.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(tokens).strip()