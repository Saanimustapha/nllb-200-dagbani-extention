import os
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2
from collections import Counter
import logging
import json
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/dag_en_hf"
OUTPUT_DIR = Path("models/tokenizer")
RESULTS_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "facebook/nllb-200-distilled-600M"
NEW_SPM_VOCAB_SIZE = 8000
MAX_NEW_TOKENS = 3000
MIN_TOKEN_FREQ = 5

# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("extend_tokenizer")

def calculate_fertility(texts, tokenizer, sample_size=3000):
    ratios = []
    for text in texts[:sample_size]:
        words = text.split()
        if not words:
            continue
        tokens = tokenizer.tokenize(text)
        ratios.append(len(tokens) / len(words))
    return float(np.mean(ratios))

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "base_model": BASE_MODEL,
        "max_new_tokens_requested": MAX_NEW_TOKENS,
        "min_token_freq": MIN_TOKEN_FREQ,
    }

    logger.info("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)
    dag_texts = dataset["train"]["dag"]
    eng_texts = dataset["train"]["en"]

    # -------------------------------------------------
    # 1. Measure original fertility
    # -------------------------------------------------
    logger.info("Loading original NLLB tokenizer...")
    original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    fert_dag_orig = calculate_fertility(dag_texts, original_tokenizer)
    fert_eng_orig = calculate_fertility(eng_texts, original_tokenizer)

    results["original_fertility"] = {
        "dagbani_tokens_per_word": round(fert_dag_orig, 3),
        "english_tokens_per_word": round(fert_eng_orig, 3),
    }

    logger.info(f"Original fertility - Dagbani: {fert_dag_orig:.3f}")
    logger.info(f"Original fertility - English: {fert_eng_orig:.3f}")

    # -------------------------------------------------
    # 2. Train Dagbani SentencePiece
    # -------------------------------------------------
    corpus_file = "data/dag_corpus.txt"
    logger.info(f"Writing Dagbani corpus to {corpus_file}...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in dag_texts:
            f.write(t + "\n")

    spm_prefix = str(OUTPUT_DIR / "dag_spm")
    logger.info("Training SentencePiece model...")

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=spm_prefix,
        vocab_size=NEW_SPM_VOCAB_SIZE,
        character_coverage=0.9995,
        model_type="bpe",
        num_threads=8,
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        pad_id=-1,
    )
    logger.info("SentencePiece training finished.")

    # -------------------------------------------------
    # 3. Extract new tokens
    # -------------------------------------------------
    logger.info("Extracting new tokens...")

    original_spm = model_pb2.ModelProto()
    with open(original_tokenizer.vocab_file, "rb") as f:
        original_spm.ParseFromString(f.read())
    existing_pieces = {p.piece for p in original_spm.pieces}

    new_spm = model_pb2.ModelProto()
    with open(f"{spm_prefix}.model", "rb") as f:
        new_spm.ParseFromString(f.read())

    sp = spm.SentencePieceProcessor()
    sp.load(f"{spm_prefix}.model")

    freq = Counter()
    for text in dag_texts:
        pieces = sp.encode_as_pieces(text)
        freq.update(pieces)

    candidates = []
    for p in new_spm.pieces:
        piece = p.piece
        if piece in existing_pieces:
            continue
        if freq[piece] < MIN_TOKEN_FREQ:
            continue
        candidates.append((freq[piece], piece))

    candidates.sort(reverse=True)
    selected_pieces = [piece for _, piece in candidates[:MAX_NEW_TOKENS]]

    results["new_tokens"] = {
        "num_selected": len(selected_pieces),
        "top_40_tokens": selected_pieces[:40],
        "top_40_with_freq": [(piece, freq[piece]) for piece in selected_pieces[:40]],
    }

    logger.info(f"Selected {len(selected_pieces)} new tokens.")

    # -------------------------------------------------
    # 4. Build extended tokenizer
    # -------------------------------------------------
    logger.info("Building extended tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    num_added = tokenizer.add_tokens(selected_pieces)
    tokenizer.add_special_tokens({"additional_special_tokens": ["dag_Latn"]})

    results["tokenizer"] = {
        "num_tokens_added": num_added,
        "final_vocab_size": len(tokenizer),
        "language_code_added": "dag_Latn",
    }

    save_path = OUTPUT_DIR / "nllb_dag_extended"
    tokenizer.save_pretrained(save_path)
    logger.info(f"Extended tokenizer saved to: {save_path}")

    # -------------------------------------------------
    # 5. Measure new fertility
    # -------------------------------------------------
    fert_dag_new = calculate_fertility(dag_texts, tokenizer)

    results["new_fertility"] = {
        "dagbani_tokens_per_word": round(fert_dag_new, 3),
        "improvement": round(fert_dag_orig - fert_dag_new, 3),
    }

    logger.info(f"New fertility - Dagbani: {fert_dag_new:.3f}")
    logger.info(f"Improvement: {fert_dag_orig - fert_dag_new:.3f} tokens/word")

    # -------------------------------------------------
    # 6. Save all results to files
    # -------------------------------------------------
    # Full JSON (machine-readable)
    json_path = RESULTS_DIR / f"tokenizer_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_path = RESULTS_DIR / f"tokenizer_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 3 — TOKENIZER EXTENSION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Base model: {BASE_MODEL}\n\n")

        f.write("1. ORIGINAL FERTILITY\n")
        f.write(f"   Dagbani : {fert_dag_orig:.3f} tokens/word\n")
        f.write(f"   English : {fert_eng_orig:.3f} tokens/word\n\n")

        f.write("2. NEW TOKENS ADDED\n")
        f.write(f"   Number of new tokens selected : {len(selected_pieces)}\n")
        f.write(f"   Final vocabulary size         : {len(tokenizer)}\n")
        f.write(f"   Language code added           : dag_Latn\n\n")

        f.write("3. NEW FERTILITY (after extension)\n")
        f.write(f"   Dagbani : {fert_dag_new:.3f} tokens/word\n")
        f.write(f"   Improvement : {fert_dag_orig - fert_dag_new:.3f} tokens/word reduction\n\n")

        f.write("4. SAMPLE OF NEWLY ADDED TOKENS (top 40 by frequency)\n")
        for i, piece in enumerate(selected_pieces[:40], 1):
            f.write(f"   {i:2d}. {piece}  (freq={freq[piece]})\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Files saved:\n")
        f.write(f"  - {json_path}\n")
        f.write(f"  - {summary_path}\n")
        f.write(f"  - Tokenizer: {save_path}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Results saved to:\n  {json_path}\n  {summary_path}")
    logger.info("Phase 3 completed successfully.")

if __name__ == "__main__":
    main()