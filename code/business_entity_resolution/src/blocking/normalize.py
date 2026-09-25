import re

SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "llc", "co", "company"
}

def normalize_name(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(tokens).strip()