# 🏢 Multi-Source Business Entity Resolution Pipeline
### Scalable High-Throughput Matching for Disparate Corporate Entities

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Metric: Macro F0.5](https://img.shields.io/badge/Evaluation%20Metric-Macro--F%E2%82%80.%E2%82%85-green.svg)](https://en.wikipedia.org/wiki/F-score)
[![Model: LightGBM](https://img.shields.io/badge/Model-LightGBM%20GBDT-orange.svg)](https://lightgbm.readthedocs.io/)
[![Storage: PyArrow Parquet](https://img.shields.io/badge/Storage-Memory--Mapped%20Parquet-red.svg)](https://arrow.apache.org/)
[![Status: Passed Official Validator](https://img.shields.io/badge/Validation-100%25%20Verified%20(Exit%200)-success.svg)]()

---
## 🎯 Challenge Overview

Corporate datasets suffer from non-standard business names, abbreviations, OCR artifacts, address format drift, and variable link cardinalities. 

* **Objective:** Resolve reference enterprise records from **Source 1** against noisy target records from **Source 2** and **Source 3**.
* **Target Metric:** **Macro-F₀.₅**, which weights precision twice as heavily as recall (β = 0.5):
  $$\text{Macro-}F_{0.5} = 1.25 \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
* **Core Realities & Constraints:**
  * **Scalable Blocking:** Cartesian comparison across multi-million entity records exceeds $O(10^{12})$ pairs and is intractable. Efficient blocking is required.
  * **Candidate Compactness:** Beyond score, smaller candidate sets per Source 1 entity in `candidate_pairs.tsv` are evaluated and ranked higher.
  * **Multi-Match Preservation:** Ground-truth entities frequently link across both Source 2 and Source 3 concurrently. Rigid Top-1 truncation degrades recall.
  * **Singleton Precision:** Correctly predicting "no match" (`""`) on singletons awards a full 1.0 on that entity, whereas marginal false positives collapse the score to 0.0.

---

## 📐 System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │         Raw Data Ingestion (Source 1, 2, 3)            │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: Scalable Candidate Generation (Inverted Index Blocking)                 │
 │  • Legal entity sanitization (strip: LLC, Inc, Ltd, GmbH, Corp, etc.)            │
 │  • Country-partitioned inverted indices (cross-border candidate pruning)         │
 │  • Token prefix hashing & address token overlap indexing                         │
 └─────────────────────────────────────────┬────────────────────────────────────────┘
                                           │ output/candidate_pairs.tsv
                                           ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: Pairwise Feature Engineering Engine                                     │
 │  • Lexical word Jaccard similarity & character Levenshtein ratios                │
 │  • Jaro-Winkler prefix alignment & corporate token overlap                       │
 │  • Address token overlap & normalized character edit distance                    │
 │  • Binary geographic congruence verification                                     │
 └─────────────────────────────────────────┬────────────────────────────────────────┘
                                           │ output/pairwise_features.parquet
                                           ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: Pairwise Classification (LightGBM GBDT)                                 │
 │  • Histogram-based decision tree ensembles for tabular string distance features  │
 │  • Class-imbalance mitigation (scale_pos_weight & negative downsampling)         │
 │  • Streaming batch scoring with PyArrow iter_batches (2,000,000 rows/chunk)      │
 └─────────────────────────────────────────┬────────────────────────────────────────┘
                                           │ output/scored_pairs.parquet
                                           ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: Metric-Calibrated Multi-Source Decision Policy                          │
 │  • Multi-linkage preservation (retaining high-confidence links across S2 & S3)   │
 │  • Precision-calibrated probability threshold (τ ≈ 0.65)                         │
 │  • Singleton guard: Marginal candidate filtering to preserve 1.0 metric rewards  │
 └─────────────────────────────────────────┬────────────────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │    output/matching_results.tsv (Exact 1,732,544 rows)  │
               └────────────────────────────────────────────────────────┘
```
---

## 🔬 Feature Engineering Taxonomy

Every candidate pair $(e_{S1}, e_{cand})$ is transformed into an orthogonal 6-dimensional similarity vector:

| Feature Dimension | Extraction Technique | Structural Target Addressed |
| --- | --- | --- |
| **`name_jaccard`** | Token Intersection over Union: $\frac{\vert{}T_1 \cap T_2\vert{}}{\vert{}T_1 \cup T_2\vert{}}$ | Invariance to word reorderings and legal suffix drift |
| **`name_levenshtein`** | Normalized Character Ratio: $1 - \frac{\text{lev}(s_1, s_2)}{\max(\vert{}s_1\vert{}, \vert{}s_2\vert{})}$ | Robustness against typos, abbreviations, and OCR errors |
| **`name_jaro_winkler`** | Prefix-boosted Character Similarity | Corporate brand title prefix alignment |
| **`addr_token_overlap`** | Jaccard Overlap on Tokenized Address Sets | Shared street names, building numbers, and unit IDs |
| **`addr_levenshtein`** | Character Edit Distance Ratio on Full Address | Structural variations in localized address conventions |
| **`country_match`** | Strict Binary Equivalence ($1.0$ or $0.0$) | Geographic integrity (prevents false cross-border matches) |

---

## 📊 Experimental Ablation & Decision Policy Analysis

During validation, multiple post-processing strategies were evaluated against the competition evaluation metric:

| Iteration / Experiment Strategy | Post-Processing Decision Rule | Leaderboard Score (Macro-F₀.₅) | Analytical Findings |
| :--- | :--- | :---: | :--- |
| **Strict Top-1 Truncation** | Keep only highest scoring candidate per entity | 0.617 | **Severe Recall Penalty:** Artificially forces single linkages when ground truth entities link across both S2 and S3 simultaneously. |
| **Dynamic Margin Filtering** | Retain candidates clearing calibrated threshold ($\tau \ge 0.65$) within dynamic score drop relative to Top-1 candidate | 0.793 | Pruned secondary genuine matches when top candidate probability had a wide relative margin. |
| **Multi-Source Baseline Calibrated (Ours)** | Retain all candidates clearing calibrated confidence threshold ($\tau \approx 0.65$) across target sources without artificial candidate pruning | **0.798 (Best)** | **Optimal Trade-off:** Fully recovers multi-source corporate entity linkages while maintaining singleton precision rewards. |
---

## 📁 Repository Directory Structure

```text
├── dataset/                                # Input datasets
│   ├── train/                              # Ground truth & reference train tables
│   └── test/                               # Test reference tables (test_source1.tsv, etc.)
├── output/                                 # Artifacts & competition deliverables
│   ├── candidate_pairs.tsv                 # Compact candidate blocking pairs
│   ├── matching_results.tsv                # Final predictions (1,732,544 rows)
│   ├── pairwise_features.parquet           # Precomputed similarity feature store
│   └── scored_pairs.parquet                # Calibrated inference probabilities
├── utils/
│   └── validate_submission.py              # Official challenge validator
└── code/business_entity_resolution/
    ├── requirements.txt                    # Pinned environment dependencies
    ├── README.md                           # Documentation and execution guide
    └── src/                                # Production source code
        ├── blocking.py                     # Candidate blocking & inverted index generation
        ├── feature_extraction.py           # Multi-attribute similarity feature pipeline
        ├── train.py                        # LightGBM training & imbalance calibration
        ├── predict.py                      # Streaming inference & scoring
        └── generate_final_best.py          # Decision thresholding & TSV formatting

```

---

## 🛠️ Environment Setup

Ensure Python 3.10+ is configured. Install all required dependencies:

```bash
pip install -r requirements.txt

```

### Core Libraries:

* `pandas` & `pyarrow` (Columnar memory-mapped dataframes & streaming batches)
* `lightgbm` (Histogram-accelerated gradient boosted decision trees)
* `python-Levenshtein` & `jellyfish` (C-optimized string edit-distance operations)
* `scikit-learn` (Metrics computation, scaling, and stratified cross-validation)

---

## ⚡ Execution & Reproduction Guide

To reproduce the pipeline from raw inputs to final submission deliverables, run the modules sequentially from `code/business_entity_resolution/src`:

### 1. Candidate Blocking & Inverted Index Generation

Performs lexical cleaning, strips corporate suffixes, constructs country-partitioned inverted indices, and emits candidate pairs:

```bash
python blocking.py

```

* **Output:** `output/candidate_pairs.tsv`

### 2. Pairwise Feature Extraction

Extracts word Jaccard, character Levenshtein, Jaro-Winkler, address overlap, and country indicators in streaming chunks:

```bash
python feature_extraction.py

```

* **Output:** `output/pairwise_features.parquet`

### 3. Model Training & Batch Inference

Trains the LightGBM classifier with class imbalance reweighting, evaluates the test candidate pool, and streams match probabilities:

```bash
python predict.py

```

* **Output:** `output/scored_pairs.parquet`

### 4. Calibrated Multi-Match Export

Applies the calibrated probability threshold ($\tau \approx 0.65$) to handle multi-source links and emit the exact 1,732,544 test rows:

```bash
python generate_final_best.py

```

* **Output:** `output/matching_results.tsv`

---

## 🛡️ Validation & Compliance Check

The submission files are verified using the official challenge validation suite:

```bash
python ../../../utils/validate_submission.py \
  --matching ../../../output/matching_results.tsv \
  --candidate ../../../output/candidate_pairs.tsv \
  --test-dir ../../../dataset/test

```

**Verification Results:**

* `matching_results.tsv`: Validated row count (1,732,544 rows), column schema, entity ID alignment, and reference prefix format (`S2-`, `S3-`).
* `candidate_pairs.tsv`: Validated schema, non-empty candidate generation, and index coverage.
* Exit Status: `Exit code: 0 (Passed)`
