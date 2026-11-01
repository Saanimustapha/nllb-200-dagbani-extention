import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import unicodedata
import re
from collections import Counter
import logging

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = r"C:\Users\MSI\Desktop\nllb-dag\ghana-corpus-builder\dag_en_parallel.csv"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.05          # 5% for final test
VAL_SIZE = 0.05           # 5% for validation (from the remaining)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("prepare_data")

# -----------------------------
# Text cleaning functions
# -----------------------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Remove control characters except newline/tab
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def is_good_pair(dag: str, eng: str, min_len=3, max_len=250) -> bool:
    if not dag or not eng:
        return False
    if len(dag) < min_len or len(eng) < min_len:
        return False
    if len(dag) > max_len or len(eng) > max_len:
        return False
    # Very simple filter for mostly empty or garbage
    if dag.count("?") > 5 or eng.count("?") > 5:
        return False
    return True

# -----------------------------
# Main processing
# -----------------------------
def main():
    logger.info(f"Loading CSV from: {CSV_PATH}")
    
    # Try different encodings in order of likelihood
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    df = None
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            logger.info(f"Successfully loaded with encoding: {enc}")
            break
        except Exception as e:
            logger.warning(f"Failed with {enc}: {e}")
    
    if df is None:
        raise ValueError("Could not read the CSV with any common encoding.")

    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Expected columns after standardization
    required = {"dag", "en"}
    if not required.issubset(set(df.columns)):
        rename_map = {}
        for col in df.columns:
            if "dag" in col:
                rename_map[col] = "dag"
            elif col in ["en", "eng", "english"]:
                rename_map[col] = "en"
        df = df.rename(columns=rename_map)
    
    if not {"dag", "en"}.issubset(df.columns):
        raise ValueError(f"Could not find 'dag' and 'en' columns. Found: {list(df.columns)}")

    # Keep only needed columns
    df = df[["dag", "en"]].copy()
    
    # Clean text
    logger.info("Cleaning text...")
    df["dag"] = df["dag"].astype(str).apply(normalize_text)
    df["en"]  = df["en"].astype(str).apply(normalize_text)

    # Filter bad pairs
    mask = df.apply(lambda row: is_good_pair(row["dag"], row["en"]), axis=1)
    df = df[mask].reset_index(drop=True)
    logger.info(f"After cleaning & filtering: {len(df)} pairs")

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["dag", "en"]).reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} exact duplicates → {len(df)} pairs remaining")

    # -----------------------------
    # Split
    # -----------------------------
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
    )

    logger.info(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

    # -----------------------------
    # Create Hugging Face Dataset
    # -----------------------------
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val.reset_index(drop=True)),
        "test": Dataset.from_pandas(test.reset_index(drop=True)),
    })

    # Save
    save_path = OUTPUT_DIR / "dag_en_hf"
    dataset.save_to_disk(str(save_path))
    logger.info(f"Saved Hugging Face dataset to: {save_path}")

    # Also save clean CSVs for inspection
    train.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val.to_csv(OUTPUT_DIR / "validation.csv", index=False, encoding="utf-8")
    test.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")
    logger.info("Also saved clean CSV splits.")

    # -----------------------------
    # Quick statistics
    # -----------------------------
    logger.info("\n===== Quick Statistics =====")
    for split_name, split_df in [("train", train), ("validation", val), ("test", test)]:
        avg_dag = split_df["dag"].str.len().mean()
        avg_en  = split_df["en"].str.len().mean()
        logger.info(f"{split_name:12} | pairs: {len(split_df):6} | avg dag chars: {avg_dag:.1f} | avg en chars: {avg_en:.1f}")

    # Sample
    logger.info("\n===== Random samples =====")
    print(train.sample(5, random_state=42)[["dag", "en"]].to_string())

    logger.info("\nPhase 2 completed successfully.")

if __name__ == "__main__":
    main()