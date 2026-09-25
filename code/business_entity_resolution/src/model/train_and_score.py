import os
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def load_ground_truth(gt_path):
    """Loads ground truth links and sets label 1."""
    gt_df = pd.read_csv(gt_path, sep="\t")
    # Ground truth expected columns: source1_entity_id, match_entity_id (or similar)
    cols = gt_df.columns
    s1_col = [c for c in cols if "source1" in c.lower() or "s1" in c.lower()][0]
    cand_col = [c for c in cols if c != s1_col][0]
    
    gt_pairs = set(zip(gt_df[s1_col].astype(str), gt_df[cand_col].astype(str)))
    return gt_pairs, s1_col, cand_col

def main():
    parser = argparse.ArgumentParser(description="Person 3: Train baseline classifier and score pairs")
    parser.add_argument("--features", default="../../../output/pairwise_features.parquet", help="Path to pairwise features parquet")
    parser.add_argument("--ground-truth", default="../../../dataset/train/train_ground_truth.tsv", help="Path to train ground truth TSV")
    parser.add_argument("--output-scored", default="../../../output/scored_pairs.parquet", help="Path to output scored pairs parquet")
    parser.add_argument("--threshold", type=float, default=0.6, help="Decision threshold for match prediction")
    args = parser.parse_args()

    print("1. Loading features from:", args.features)
    df = pd.read_parquet(args.features)
    print(f"   Loaded {len(df):,} candidate pairs.")

    print("2. Loading ground truth from:", args.ground_truth)
    gt_pairs, s1_col, cand_col = load_ground_truth(args.ground_truth)
    print(f"   Loaded {len(gt_pairs):,} ground truth links.")

    # Match ID columns
    id_cols = [c for c in df.columns if "id" in c.lower()]
    df_s1_col = id_cols[0]
    df_cand_col = id_cols[1]

    # Create binary target
    print("3. Generating labels (positives vs candidate hard-negatives)...")
    s1_arr = df[df_s1_col].astype(str).values
    cand_arr = df[df_cand_col].astype(str).values
    df["label"] = [1 if (s1, cand) in gt_pairs else 0 for s1, cand in zip(s1_arr, cand_arr)]

    pos_count = df["label"].sum()
    neg_count = len(df) - pos_count
    print(f"   Positives: {pos_count:,} | Negatives (hard negatives): {neg_count:,}")

    # Determine feature columns
    exclude_cols = {df_s1_col, df_cand_col, "label"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print("   Feature columns used:", feature_cols)

    X = df[feature_cols]
    y = df["label"]

    # Simple train/val split for Day 1 self-check
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("4. Training LightGBM classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    val_preds_prob = clf.predict_proba(X_val)[:, 1]
    print(f"   Validation AUC-ROC: {roc_auc_score(y_val, val_preds_prob):.4f}")
    print("   Classification report (threshold 0.5):")
    print(classification_report(y_val, (val_preds_prob >= 0.5).astype(int), digits=4))

    # Score all pairs
    print("5. Generating match probabilities for all pairs...")
    df["match_probability"] = clf.predict_proba(X)[:, 1]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_scored)), exist_ok=True)
    df.to_parquet(args.output_scored, index=False)
    print(f"   Saved scored pairs to: {args.output_scored}")

if __name__ == "__main__":
    main()