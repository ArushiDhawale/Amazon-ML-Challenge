import os
import sys
import tempfile
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalization import normalize_name, normalize_address, normalize_country, normalize_record
from features import parse_candidate_pairs, build_pairwise_features

def test_normalization():
    print("Testing Normalization...")
    assert normalize_name("Tata Motors Pvt. Ltd.") == "tata motors private limited"
    assert normalize_name("Johnson & Johnson Corp.") == "johnson and johnson corporation"
    assert normalize_name("Colorado Coordinator Co.") == "colorado coordinator company"
    assert normalize_address("123 Main St., 4th Fl., Ste 200") == "123 main street 4th floor suite 200"
    assert normalize_address("Station Rd.") == "station road"
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""
    assert normalize_name("   ") == ""
    assert normalize_address(None) == ""
    assert normalize_country("US") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("  FRANCE ") == "france"
    assert normalize_country("Côte d'Ivoire") == "cote divoire"
    print("✔ All Normalization tests passed!")

def test_feature_pipeline():
    print("Testing Feature Calculation & Candidate Parsing...")
    raw_records = {
        "S1-100": {
            "business_name": "Apple Incorporated",
            "business_address": "1 Infinite Loop, Cupertino",
            "country": "US"
        },
        "S2-200": {
            "business_name": "Apple Inc.",
            "business_address": "Infinite Loop, Cupertino",
            "country": "US"
        },
        "S3-300": {
            "business_name": "Banana Corp.",
            "business_address": "742 Evergreen Terrace",
            "country": "US"
        }
    }

    # Normalize records through normalization module
    normalized_records = {k: normalize_record(v) for k, v in raw_records.items()}

    # Simulate Candidate Pairs TSV
    tsv_content = "source1_entity_id\tcandidate_entity_ids\nS1-100\tS2-200,S3-300\n"
    with tempfile.NamedTemporaryFile("w+", suffix=".tsv", delete=False) as tmp:
        tmp.write(tsv_content)
        tmp_name = tmp.name

    df_pairs = parse_candidate_pairs(tmp_name)
    os.remove(tmp_name)

    assert len(df_pairs) == 2, f"Expected 2 pairs, got {len(df_pairs)}"
    features = build_pairwise_features(df_pairs, normalized_records)

    assert len(features) == 2
    assert features.isna().sum().sum() == 0, "No NaNs allowed!"
    
    # Pair 0: S1-100 vs S2-200 (both normalized to 'apple incorporated') -> exact match 1.0
    assert features.loc[0, "country_match"] == 1.0
    assert features.loc[0, "name_levenshtein"] == 1.0
    assert features.loc[0, "name_jaccard"] == 1.0
    
    # Pair 1: S1-100 vs S3-300 ('apple incorporated' vs 'banana corporation') -> low similarity
    assert features.loc[1, "name_levenshtein"] < 0.5
    print("✔ Feature pipeline tests passed!")

if __name__ == "__main__":
    test_normalization()
    test_feature_pipeline()
    print("\nALL DAY 1 CHECKS PASSED.")