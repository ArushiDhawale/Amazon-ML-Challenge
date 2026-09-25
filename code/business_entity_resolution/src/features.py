import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    union = len(s1.union(s2))
    return float(len(s1.intersection(s2)) / union) if union > 0 else 0.0

def compute_token_overlap(tokens1: List[str], tokens2: List[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    denom = min(len(s1), len(s2))
    return float(len(s1.intersection(s2)) / denom) if denom > 0 else 0.0

def compute_levenshtein_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(Levenshtein.normalized_similarity(s1, s2))
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return float(1.0 - (dist / max_len)) if max_len > 0 else 0.0

def compute_jaro_winkler_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(JaroWinkler.similarity(s1, s2))
    return compute_levenshtein_sim(s1, s2)

def parse_candidate_pairs(candidate_file_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(candidate_file_path, sep="\t", dtype=str, keep_default_na=False)
    s1_col = "source1_entity_id"
    
    # Check for all known candidate/match column variations
    possible_cand_cols = ["matched_entity_ids", "candidate_entity_ids", "candidate_entity_id"]
    cand_col = None
    for col in possible_cand_cols:
        if col in df_raw.columns:
            cand_col = col
            break
            
    if cand_col is None:
        raise KeyError(f"Could not find candidate column. Available columns: {list(df_raw.columns)}")
    
    pairs: List[Tuple[str, str]] = []
    for _, row in df_raw.iterrows():
        s1_id = str(row[s1_col]).strip()
        cand_str = str(row[cand_col]).strip()
        if not s1_id or not cand_str:
            continue
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        for c in cands:
            pairs.append((s1_id, c))
            
    df_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    df_pairs.drop_duplicates(inplace=True)
    return df_pairs

def build_pairwise_features(df_pairs: pd.DataFrame, records_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    n_pairs = len(df_pairs)
    name_jaccard = np.zeros(n_pairs, dtype=np.float32)
    name_levenshtein = np.zeros(n_pairs, dtype=np.float32)
    name_jaro_winkler = np.zeros(n_pairs, dtype=np.float32)
    addr_token_overlap = np.zeros(n_pairs, dtype=np.float32)
    addr_levenshtein = np.zeros(n_pairs, dtype=np.float32)
    country_match = np.zeros(n_pairs, dtype=np.float32)

    s1_ids = df_pairs["source1_entity_id"].values
    cand_ids = df_pairs["candidate_entity_id"].values

    for i in range(n_pairs):
        r1 = records_map.get(s1_ids[i], {})
        r2 = records_map.get(cand_ids[i], {})

        name1 = r1.get("business_name_normalized", "")
        name2 = r2.get("business_name_normalized", "")
        addr1 = r1.get("business_address_normalized", "")
        addr2 = r2.get("business_address_normalized", "")
        c1 = r1.get("country_normalized", "")
        c2 = r2.get("country_normalized", "")

        t_name1 = name1.split() if name1 else []
        t_name2 = name2.split() if name2 else []
        t_addr1 = addr1.split() if addr1 else []
        t_addr2 = addr2.split() if addr2 else []

        name_jaccard[i] = compute_token_jaccard(t_name1, t_name2)
        name_levenshtein[i] = compute_levenshtein_sim(name1, name2)
        name_jaro_winkler[i] = compute_jaro_winkler_sim(name1, name2)
        addr_token_overlap[i] = compute_token_overlap(t_addr1, t_addr2)
        addr_levenshtein[i] = compute_levenshtein_sim(addr1, addr2)
        country_match[i] = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    return pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "name_jaccard": name_jaccard,
        "name_levenshtein": name_levenshtein,
        "name_jaro_winkler": name_jaro_winkler,
        "addr_token_overlap": addr_token_overlap,
        "addr_levenshtein": addr_levenshtein,
        "country_match": country_match,
    })