import csv
import sys

input_path = "../../../output/pairwise_candidates.tsv"
output_path = "../../../output/candidate_pairs.tsv"

print("Grouping pairwise candidates into submission format...")

count = 0
current_s1 = None
cands = []

with (
    open(input_path, "r", encoding="utf-8") as fin,
    open(output_path, "w", encoding="utf-8", newline="") as fout,
):
  reader = csv.reader(fin, delimiter="\t")
  writer = csv.writer(fout, delimiter="\t")

  header = next(reader)
  writer.writerow(["source1_entity_id", "candidate_entity_ids"])

  for row in reader:
    if not row:
      continue
    s1, cand = row[0], row[1]
    if s1 != current_s1:
      if current_s1 is not None:
        writer.writerow([current_s1, ",".join(cands)])
        count += 1
      current_s1 = s1
      cands = [cand]
    else:
      cands.append(cand)

  if current_s1 is not None:
    writer.writerow([current_s1, ",".join(cands)])
    count += 1

print(f"SUCCESS: Formatted {count:,} entities into {output_path}")