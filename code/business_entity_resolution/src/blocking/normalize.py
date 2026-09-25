import os
import sys

# Connect to Person 2 normalization module
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from normalization import normalize_name, normalize_address, normalize_country, normalize_record

__all__ = ["normalize_name", "normalize_address", "normalize_country", "normalize_record"]