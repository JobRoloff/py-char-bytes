"""Dataset loader and preprocessor for the committed CLEAR corpus CSV."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("DatasetLoader")

# Define path relative to repository root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAR_CSV_PATH = BASE_DIR / "data" / "raw" / "clear_corpus.csv"


def load_clear_dataframe(csv_path: Path = CLEAR_CSV_PATH) -> pd.DataFrame:
    """Load, clean, and normalize the local CLEAR dataset into a standardized schema.

    Standardized Schema:
    - document_id (str): Unique passage identifier
    - raw_text (str): Full text excerpt under evaluation
    - clear_easiness (float): Ground-truth human rating (higher = easier to read)
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CLEAR dataset not found at {csv_path}. "
            f"Please ensure 'clear_corpus.csv' is saved inside data/raw/"
        )

    df = pd.read_csv(csv_path)

    # Map raw CSV headers to normalized pipeline contract
    rename_map = {}
    for col in df.columns:
        col_clean = col.strip().lower()
        if col_clean in ["id", "doc_id", "document_id"]:
            rename_map[col] = "document_id"
        elif col_clean in ["excerpt", "text", "raw_text", "passage"]:
            rename_map[col] = "raw_text"
        elif col_clean in ["bt easiness", "bt_easiness", "target", "easiness"]:
            rename_map[col] = "clear_easiness"

    df = df.rename(columns=rename_map)

    required_cols = ["document_id", "raw_text", "clear_easiness"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CLEAR dataset missing required columns: {missing}. Found headers: {list(df.columns)}"
        )

    # Clean missing or malformed rows and filter down to target schema
    cleaned_df = df[required_cols].dropna().copy()
    cleaned_df["document_id"] = cleaned_df["document_id"].astype(str)
    cleaned_df["raw_text"] = cleaned_df["raw_text"].astype(str)
    cleaned_df["clear_easiness"] = cleaned_df["clear_easiness"].astype(float)

    logger.info(f"Successfully loaded {len(cleaned_df)} sanitized CLEAR passages.")
    return cleaned_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_clear_dataframe()
    print(f"Sanitized Data Preview:\n{data.head()}")
