import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalization import normalize_dataframe
from features import parse_candidate_pairs, build_pairwise_features

def load_and_normalize_sources(data_dir: str, prefix: str):
    records = {}
    for src_id in [1, 2, 3]:
        fname = f"{prefix}_source{src_id}.tsv"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {fname} (not present)")
            continue
        print(f"Loading and normalizing {fname}...")
        df = pd.read_csv(fpath, sep="\t", dtype=str)
        df_norm = normalize_dataframe(df)
        for _, row in df_norm.iterrows():
            records[row["entity_id"]] = {
                "business_name_normalized": row["business_name_normalized"],
                "business_address_normalized": row["business_address_normalized"],
                "country_normalized": row["country_normalized"],
            }
    return records

def main():
    parser = argparse.ArgumentParser(description="Day 1 Pairwise Feature Pipeline")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidate TSV")
    parser.add_argument("--data-dir", type=str, default="dataset/train", help="Dataset directory")
    parser.add_argument("--prefix", type=str, default="train", help="'train' or 'test'")
    parser.add_argument("--output", type=str, default="output/pairwise_features.parquet", help="Parquet output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    records_map = load_and_normalize_sources(args.data_dir, args.prefix)
    print(f"Total normalized records in memory: {len(records_map)}")

    print(f"Parsing candidates from {args.candidates}...")
    df_pairs = parse_candidate_pairs(args.candidates)
    print(f"Unique candidate pairs to score: {len(df_pairs)}")

    if len(df_pairs) == 0:
        print("No candidate pairs found.")
        sys.exit(0)

    print("Computing features...")
    df_features = build_pairwise_features(df_pairs, records_map)

    assert df_features.isna().sum().sum() == 0, "Error: NaN values detected in features!"
    assert len(df_features) == len(df_pairs), "Error: Pair count mismatch!"

    df_features.to_parquet(args.output, index=False)
    print(f"SUCCESS: Wrote {len(df_features)} feature rows to {args.output}")

if __name__ == "__main__":
    main()