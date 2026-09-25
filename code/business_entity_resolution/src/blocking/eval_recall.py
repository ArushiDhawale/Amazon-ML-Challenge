import os
import pandas as pd
from load_data import load_all, load_source
from blocking import build_candidates

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset/train")
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

print("Loading training data...")
s1, s2, s3 = load_all(TRAIN_DIR)
gt_df = load_source(GT_PATH)

# Test on 5,000 matched S1 records to check recall ceiling quickly
gt_matches = gt_df[gt_df['matched_entity_ids'] != '']
sample_ids = set(gt_matches['source1_entity_id'].sample(n=min(5000, len(gt_matches)), random_state=42))
s1_sample = s1[s1['entity_id'].isin(sample_ids)].copy()

print(f"Running candidate generator on {len(s1_sample)} sample records...")
cand_df = build_candidates(s1_sample, s2, s3, top_k=25)

gt = gt_df.set_index("source1_entity_id")["matched_entity_ids"]
cand = cand_df.set_index("source1_entity_id")["candidate_entity_ids"]

total, found = 0, 0
for s1_id in s1_sample['entity_id']:
    true_ids = set(gt.get(s1_id, "").split(",")) if gt.get(s1_id, "") else set()
    cand_ids = set(cand.get(s1_id, "").split(",")) if cand.get(s1_id, "") else set()
    total += len(true_ids)
    found += len(true_ids & cand_ids)

recall = (found / total) * 100 if total else 0.0
print(f"\n==========================================")
print(f"Sample True Links: {total}")
print(f"Found Links:       {found}")
print(f"Recall Ceiling:    {recall:.2f}%")
print(f"==========================================")