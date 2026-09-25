import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Person 3: Generate compliant matching_results.tsv from scored pairs")
    parser.add_argument("--scored", default="../../../output/scored_pairs.parquet", help="Path to scored pairs parquet")
    parser.add_argument("--candidates", default="../../../output/candidate_pairs.tsv", help="Path to candidates TSV to capture all Source 1 IDs")
    parser.add_argument("--output", default="../../../output/matching_results.tsv", help="Path to output matching_results.tsv")
    parser.add_argument("--threshold", type=float, default=0.6, help="Match probability threshold")
    args = parser.parse_args()

    print(f"1. Loading scored pairs from {args.scored}...")
    df = pd.read_parquet(args.scored)

    # Determine ID columns
    id_cols = [c for c in df.columns if "id" in c.lower() and c != "label"]
    s1_col = id_cols[0]
    cand_col = id_cols[1]

    # Get the complete list of Source 1 entities evaluated
    if os.path.exists(args.candidates):
        cand_df = pd.read_csv(args.candidates, sep="\t", usecols=[0])
        all_s1_ids = cand_df.iloc[:, 0].astype(str).unique()
    else:
        all_s1_ids = df[s1_col].astype(str).unique()

    print(f"2. Filtering candidate pairs with match_probability >= {args.threshold}...")
    filtered = df[df["match_probability"] >= args.threshold].copy()
    
    # Sort pairs by highest probability first
    filtered = filtered.sort_values(by=[s1_col, "match_probability"], ascending=[True, False])

    # Group candidate IDs per Source 1 entity (comma-separated)
    matches_dict = (
        filtered.groupby(s1_col)[cand_col]
        .apply(lambda ids: ",".join(ids.astype(str).unique()))
        .to_dict()
    )

    # Build compliant submission table including singletons
    print("3. Formatting final matching results (including singletons)...")
    records = []
    for s1_id in all_s1_ids:
        records.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matches_dict.get(s1_id, "")  # empty string for singletons
        })

    submission_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    submission_df.to_csv(args.output, sep="\t", index=False)
    
    matched_count = sum(1 for r in records if r["matched_entity_ids"] != "")
    print(f"SUCCESS: Wrote {len(submission_df):,} total entities ({matched_count:,} matched, {len(submission_df) - matched_count:,} singletons) to {args.output}")

if __name__ == "__main__":
    main()