import re
import pandas as pd
from collections import defaultdict

SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "llc", "co", "company"
}

def tokenize(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[&]", " and ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in SUFFIXES]
    return tokens

def build_candidates(s1, s2, s3, top_k=25):
    # Combine S2 and S3 pool
    s23 = pd.concat([s2[['entity_id', 'business_name', 'country']], 
                     s3[['entity_id', 'business_name', 'country']]], ignore_index=True)
    
    # Process country-by-country to keep partitions isolated and memory small
    candidates_dict = {sid: [] for sid in s1['entity_id']}
    
    for country, s1_country_df in s1.groupby('country'):
        s23_country_df = s23[s23['country'] == country]
        if s23_country_df.empty:
            continue
            
        print(f"Indexing {country}: S1={len(s1_country_df)}, Pool={len(s23_country_df)}...")
        
        # 1. Build Inverted Index on S2+S3 (Token -> List of Entity IDs)
        inverted_index = defaultdict(list)
        for eid, name in zip(s23_country_df['entity_id'], s23_country_df['business_name']):
            tokens = set(tokenize(name))
            for tok in tokens:
                inverted_index[tok].append(eid)
                
        # 2. Query S1 records against the Inverted Index
        for s1_id, name in zip(s1_country_df['entity_id'], s1_country_df['business_name']):
            tokens = set(tokenize(name))
            if not tokens:
                continue
                
            # Count token overlap frequency
            overlap_counts = defaultdict(int)
            for tok in tokens:
                for match_id in inverted_index.get(tok, []):
                    overlap_counts[match_id] += 1
            
            if overlap_counts:
                # Pick top_k candidates with highest token overlap
                sorted_cands = sorted(overlap_counts.keys(), key=lambda x: overlap_counts[x], reverse=True)[:top_k]
                candidates_dict[s1_id] = sorted_cands

    # Convert to DataFrame format
    rows = []
    for s1_id in s1['entity_id']:
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidates_dict.get(s1_id, []))
        })
    return pd.DataFrame(rows)

def write_candidate_pairs(df, path="output/candidate_pairs.tsv"):
    df.to_csv(path, sep="\t", index=False)