import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import logging
import json
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
LOCAL_MODEL_PATH = "models/nllb-200-distilled-600M"          
TOKENIZER_PATH = "models/tokenizer/nllb_dag_extended"
OUTPUT_DIR = Path("models/nllb_dag_resized")
RESULTS_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Memory-friendly settings for 6 GB VRAM
USE_4BIT = True
MOSSI_BIAS_WEIGHT = 0.12          # small pull toward Mossi

# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("resize_embeddings")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": timestamp}

    # -------------------------------------------------
    # 1. Load extended tokenizer
    # -------------------------------------------------
    logger.info(f"Loading extended tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    new_vocab_size = len(tokenizer)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    results["new_vocab_size"] = new_vocab_size

    # -------------------------------------------------
    # 2. Load base model from LOCAL folder in 4-bit
    # -------------------------------------------------
    logger.info(f"Loading base model from local path: {LOCAL_MODEL_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LOCAL_MODEL_PATH,                     
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,                
    )

    original_vocab_size = model.config.vocab_size
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    results["original_vocab_size"] = original_vocab_size

    # -------------------------------------------------
    # 3. Resize token embeddings
    # -------------------------------------------------
    logger.info("Resizing token embeddings...")
    model.resize_token_embeddings(new_vocab_size)
    logger.info("Resize completed.")

    # -------------------------------------------------
    # 4. Smart initialization of new embeddings
    # -------------------------------------------------
    logger.info("Initializing new token embeddings (subword average + Mossi bias)...")

    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight

        # Get Mossi language token embedding as weak prior
        mossi_emb = None
        mossi_bias = MOSSI_BIAS_WEIGHT         
        try:
            mossi_id = tokenizer.convert_tokens_to_ids("mos_Latn")
            logger.info(f"mos_Latn token ID = {mossi_id}")

            if mossi_id == tokenizer.unk_token_id:
                raise ValueError("mos_Latn resolved to UNK")

            # Move to same device as embeddings and convert to float for safety
            mossi_emb = embeddings[mossi_id].detach().to(dtype=torch.float32).clone()
            logger.info(f"Successfully extracted Mossi embedding (shape={mossi_emb.shape})")
            logger.info(f"Using Mossi bias weight = {mossi_bias}")

        except Exception as e:
            logger.warning(f"Could not extract mos_Latn embedding: {type(e).__name__}: {e}")
            logger.warning("Falling back to pure subword average (no Mossi bias)")
            mossi_emb = None
            mossi_bias = 0.0

        num_initialized = 0

        for token, idx in tokenizer.get_vocab().items():
            if idx < original_vocab_size:
                continue

            try:
                sub_tokens = tokenizer.tokenize(token)
                sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                valid_ids = [i for i in sub_ids if i < original_vocab_size and i != tokenizer.unk_token_id]

                if valid_ids:
                    avg_emb = embeddings[valid_ids].mean(dim=0)
                else:
                    avg_emb = torch.zeros_like(embeddings[0]) if mossi_emb is None else mossi_emb.clone()

                if mossi_emb is not None and MOSSI_BIAS_WEIGHT > 0:
                    mossi_emb_ = mossi_emb.to(device=avg_emb.device, dtype=avg_emb.dtype)
                    new_emb = (1.0 - mossi_bias) * avg_emb + mossi_bias * mossi_emb_
                else:
                    new_emb = avg_emb

                embeddings[idx] = new_emb
                num_initialized += 1

            except Exception:
                if mossi_emb is not None:
                    embeddings[idx] = mossi_emb + torch.randn_like(mossi_emb) * 0.01
                else:
                    embeddings[idx] = torch.randn_like(embeddings[0]) * 0.02
                num_initialized += 1

        logger.info(f"Initialized {num_initialized} new token embeddings.")

        if model.config.tie_word_embeddings:
            logger.info("Embeddings are tied – lm_head stays in sync.")
        else:
            model.lm_head.weight.data = embeddings.data.clone()

    results["num_new_embeddings_initialized"] = num_initialized
    results["mossi_bias_weight"] = MOSSI_BIAS_WEIGHT

    # -------------------------------------------------
    # 5. Prepare for k-bit training
    # -------------------------------------------------
    model = prepare_model_for_kbit_training(model)

    # -------------------------------------------------
    # 6. Save the resized model + tokenizer
    # -------------------------------------------------
    logger.info(f"Saving resized model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(OUTPUT_DIR / "resize_info.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_path = RESULTS_DIR / f"resize_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL RESIZE + EMBEDDING INITIALIZATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp                 : {timestamp}\n")
        f.write(f"Local base model          : {LOCAL_MODEL_PATH}\n")
        f.write(f"Original vocab size       : {original_vocab_size}\n")
        f.write(f"New vocab size            : {new_vocab_size}\n")
        f.write(f"New embeddings initialized: {num_initialized}\n")
        f.write(f"Mossi bias weight         : {MOSSI_BIAS_WEIGHT}\n")
        f.write(f"Quantization              : 4-bit NF4\n")
        f.write(f"Saved to                  : {OUTPUT_DIR}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Summary saved to {summary_path}")
    logger.info("Phase 4 completed successfully.")

if __name__ == "__main__":
    main()