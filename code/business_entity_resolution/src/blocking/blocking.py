import os
import re
import heapq
import time
import pandas as pd
from collections import defaultdict

SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "the", "and", "of",
    "for", "in", "on", "at", "to", "a", "an", "group", "holdings"
}

def clean_country(c):
    if not c or pd.isna(c):
        return ""
    return str(c).strip().lower()

def tokenize(text):
    if not text or pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[&]", " and ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return [t for t in text.split() if len(t) >= 2 and t not in SUFFIXES]

def build_candidates(s1, s2, s3, top_k=75, max_token_freq=35000, sample_size=None):
    s1 = s1.copy()
    s2 = s2.copy()
    s3 = s3.copy()
    
    if sample_size and sample_size < len(s1):
        print(f"Sampling {sample_size} records from Source 1 for fast pipeline validation...")
        s1 = s1.iloc[:sample_size].copy()

    s1['norm_country'] = s1['country'].apply(clean_country)
    s2['norm_country'] = s2['country'].apply(clean_country)
    s3['norm_country'] = s3['country'].apply(clean_country)

    s23 = pd.concat([
        s2[['entity_id', 'business_name', 'business_address', 'norm_country']], 
        s3[['entity_id', 'business_name', 'business_address', 'norm_country']]
    ], ignore_index=True)
    
    candidates_dict = {sid: [] for sid in s1['entity_id']}
    
    for country, s1_country_df in s1.groupby('norm_country'):
        if not country:
            continue
        s23_country_df = s23[s23['norm_country'] == country]
        if s23_country_df.empty:
            continue
            
        total_s1 = len(s1_country_df)
        print(f"\nIndexing country '{country}': S1={total_s1}, Pool={len(s23_country_df)}...")
        
        # 1. Inverted Index (Name + Address)
        inverted_index = defaultdict(list)
        for eid, name, addr in zip(s23_country_df['entity_id'], s23_country_df['business_name'], s23_country_df['business_address']):
            tokens = set(tokenize(name))
            tokens.update(tokenize(addr))
            for tok in tokens:
                inverted_index[tok].append(eid)
                
        # 2. Frequency pruning
        inverted_index = {tok: eids for tok, eids in inverted_index.items() if len(eids) <= max_token_freq}

        # 3. Query S1 with live progress reporting
        start_time = time.time()
        for idx, (s1_id, name, addr) in enumerate(zip(s1_country_df['entity_id'], s1_country_df['business_name'], s1_country_df['business_address'])):
            if (idx + 1) % 10000 == 0 or (idx + 1) == total_s1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / max(elapsed, 0.001)
                print(f"  Processed {idx + 1}/{total_s1} records ({rate:.1f} rec/s)...", flush=True)

            tokens = set(tokenize(name))
            tokens.update(tokenize(addr))
            if not tokens:
                continue
                
            overlap_counts = defaultdict(int)
            for tok in tokens:
                eids = inverted_index.get(tok)
                if eids:
                    for match_id in eids:
                        overlap_counts[match_id] += 1
            
            if overlap_counts:
                # Fast top-k using heapq
                if len(overlap_counts) <= top_k:
                    top_matches = sorted(overlap_counts.keys(), key=overlap_counts.__getitem__, reverse=True)
                else:
                    top_matches = heapq.nlargest(top_k, overlap_counts.keys(), key=overlap_counts.__getitem__)
                candidates_dict[s1_id] = top_matches

    rows = []
    for s1_id in s1['entity_id']:
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidates_dict.get(s1_id, []))
        })
    return pd.DataFrame(rows)

def write_candidate_pairs(df, path="output/candidate_pairs.tsv"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)