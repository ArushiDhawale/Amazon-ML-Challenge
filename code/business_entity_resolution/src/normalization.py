import re
import unicodedata
from typing import Any, Dict
import pandas as pd

NAME_REPLACEMENTS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bdept\b": "department",
    r"\bmfg\b": "manufacturing",
    r"\bintl\b": "international",
    r"\bassoc\b": "associates",
    r"\btech\b": "technologies",
}

ADDRESS_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bhwy\b": "highway",
    r"\bfl\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bstr\b": "street",
}

def _clean_unicode_and_case(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    text = str(val).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()

def normalize_name(name: Any) -> str:
    text = _clean_unicode_and_case(name)
    if not text:
        return ""
    text = re.sub(r"&", " and ", text)
    for pattern, replacement in NAME_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_address(address: Any) -> str:
    text = _clean_unicode_and_case(address)
    if not text:
        return ""
    for pattern, replacement in ADDRESS_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_country(country: Any) -> str:
    text = _clean_unicode_and_case(country)
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(record)
    res["business_name_normalized"] = normalize_name(res.get("business_name"))
    res["business_address_normalized"] = normalize_address(res.get("business_address"))
    res["country_normalized"] = normalize_country(res.get("country"))
    return res

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["business_name_normalized"] = df_out["business_name"].apply(normalize_name)
    df_out["business_address_normalized"] = df_out["business_address"].apply(normalize_address)
    df_out["country_normalized"] = df_out["country"].apply(normalize_country)
    return df_out