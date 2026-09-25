import argparse
import pandas as pd


def compute_macro_f05(
    gt_path: str, pred_path: str, beta: float = 0.5
) -> dict[str, float]:
  print(f"Loading ground truth from: {gt_path}")
  gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)

  print(f"Loading predictions from: {pred_path}")
  pred = pd.read_csv(pred_path, sep="\t", keep_default_na=False)

  # Detect column names dynamically
  s1_col = [c for c in pred.columns if "source1" in c.lower()][0]
  m_col = [c for c in pred.columns if c != s1_col][0]
  gt_s1 = [c for c in gt.columns if "source1" in c.lower()][0]
  gt_m = [c for c in gt.columns if c != gt_s1][0]

  # Correctly parse ground truth by splitting comma-separated IDs
  print("Mapping ground truth pairs (handling multi-match comma strings)...")
  gt_map = {}
  s1_vals = gt[gt_s1].astype(str).values
  m_vals = gt[gt_m].astype(str).values

  for s1, m_str in zip(s1_vals, m_vals):
    s1 = s1.strip()
    m_str = m_str.strip()
    if s1 not in gt_map:
      gt_map[s1] = set()
    if m_str:
      for single_id in m_str.split(","):
        single_id = single_id.strip()
        if single_id:
          gt_map[s1].add(single_id)

  beta_sq = beta**2
  f05_scores = []
  precisions = []
  recalls = []

  singleton_correct = 0
  singleton_total = 0

  for _, row in pred.iterrows():
    s1_id = str(row[s1_col]).strip()
    pred_str = str(row[m_col]).strip()

    pred_set = (
        set(p.strip() for p in pred_str.split(",") if p.strip())
        if pred_str
        else set()
    )
    actual_set = gt_map.get(s1_id, set())

    # Case 1: Ground truth is a singleton (no matches)
    if len(actual_set) == 0:
      singleton_total += 1
      if len(pred_set) == 0:
        singleton_correct += 1
        f05_scores.append(1.0)
        precisions.append(1.0)
        recalls.append(1.0)
      else:
        # Predicted a match where none exists (false positive)
        f05_scores.append(0.0)
        precisions.append(0.0)
        recalls.append(0.0)
      continue

    # Case 2: Ground truth has matches, but model predicted singleton
    if len(pred_set) == 0:
      f05_scores.append(0.0)
      precisions.append(0.0)
      recalls.append(0.0)
      continue

    # Case 3: Both predicted and actual matches exist
    tp = len(pred_set & actual_set)
    fp = len(pred_set - actual_set)
    fn = len(actual_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precisions.append(p)
    recalls.append(r)

    if p + r == 0:
      f05_scores.append(0.0)
    else:
      score = (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)
      f05_scores.append(score)

  total_entities = len(pred)
  macro_f05 = sum(f05_scores) / total_entities if total_entities > 0 else 0.0
  macro_p = sum(precisions) / total_entities if total_entities > 0 else 0.0
  macro_r = sum(recalls) / total_entities if total_entities > 0 else 0.0

  return {
      "total_entities": total_entities,
      "macro_f05": macro_f05,
      "macro_precision": macro_p,
      "macro_recall": macro_r,
      "singleton_accuracy": (
          singleton_correct / singleton_total if singleton_total > 0 else 1.0
      ),
  }


def main():
  parser = argparse.ArgumentParser(
      description="Person 4: Macro-F0.5 Evaluation"
  )
  parser.add_argument(
      "--ground-truth",
      default="../../../dataset/train/train_ground_truth.tsv",
      help="Ground truth TSV path",
  )
  parser.add_argument(
      "--pred",
      default="../../../output/matching_results.tsv",
      help="Matching results TSV path",
  )
  args = parser.parse_args()

  metrics = compute_macro_f05(args.ground_truth, args.pred)

  print("\n" + "=" * 45)
  print("          EVALUATION RESULTS")
  print("=" * 45)
  print(f"Total Evaluated Entities : {metrics['total_entities']:,}")
  print(f"Macro Precision          : {metrics['macro_precision']:.4f}")
  print(f"Macro Recall             : {metrics['macro_recall']:.4f}")
  print(f"Macro F0.5 Score         : {metrics['macro_f05']:.4f}")
  print(f"Singleton Accuracy       : {metrics['singleton_accuracy']:.4f}")
  print("=" * 45)


if __name__ == "__main__":
  main()