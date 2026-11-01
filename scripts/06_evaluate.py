import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate
import json
from datetime import datetime
import logging
import random
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
BASE_MODEL_PATH = "models/nllb_dag_resized"
ADAPTER_PATH = "checkpoints/nllb-dag-lora/final_adapter"   # change if needed
DATA_PATH = "data/dag_en_hf"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 96
NUM_BEAMS = 5
NUM_EXAMPLES_TO_SHOW = 15

# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("evaluate")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("PHASE 6 — Evaluation & Inference")
    logger.info("=" * 70)

    # -------------------------------------------------
    # 1. Load tokenizer + base model + LoRA adapter
    # -------------------------------------------------
    logger.info(f"Loading tokenizer from {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
                                                                                                                                    
    logger.info("Loading base model in 4-bit (this can take 1–3 minutes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    logger.info("Model + adapter loaded successfully.")

    # -------------------------------------------------
    # 2. Load test set
    # -------------------------------------------------
    logger.info("Loading test set...")
    dataset = load_from_disk(DATA_PATH)
    test_dataset = dataset["test"]
    logger.info(f"Test set size: {len(test_dataset)}")

    # -------------------------------------------------
    # 3. Metrics
    # -------------------------------------------------
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    # -------------------------------------------------
    # 4. Generate translations with progress bar
    # -------------------------------------------------
    logger.info("Generating translations on the test set...")
    predictions = []
    references = []

    for example in tqdm(test_dataset, desc="Generating", unit="ex", ncols=100):
        src = f"dag_Latn {example['dag']}"
        inputs = tokenizer(
            src,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if pred.startswith("eng_Latn"):
            pred = pred[len("eng_Latn"):].strip()

        predictions.append(pred)
        references.append([example["en"]])

    # -------------------------------------------------
    # 5. Compute metrics
    # -------------------------------------------------
    logger.info("Computing BLEU and chrF...")
    bleu = sacrebleu.compute(predictions=predictions, references=references)
    chrf_score = chrf.compute(predictions=predictions, references=references)

    results = {
        "timestamp": timestamp,
        "adapter_path": str(ADAPTER_PATH),
        "test_size": len(test_dataset),
        "bleu": round(bleu["score"], 2),
        "chrf": round(chrf_score["score"], 2),
        "num_beams": NUM_BEAMS,
        "max_length": MAX_LENGTH,
    }

    logger.info("=" * 70)
    logger.info("TEST SET RESULTS")
    logger.info(f"BLEU  : {results['bleu']}")
    logger.info(f"chrF  : {results['chrf']}")
    logger.info("=" * 70)

    # -------------------------------------------------
    # 6. Qualitative examples
    # -------------------------------------------------
    logger.info("\n===== Qualitative Examples =====")
    indices = random.sample(range(len(test_dataset)), k=min(NUM_EXAMPLES_TO_SHOW, len(test_dataset)))

    examples = []
    for idx in indices:
        examples.append({
            "dagbani": test_dataset[idx]["dag"],
            "reference": test_dataset[idx]["en"],
            "prediction": predictions[idx],
        })
        print(f"\nDagbani   : {test_dataset[idx]['dag']}")
        print(f"Reference : {test_dataset[idx]['en']}")
        print(f"Prediction: {predictions[idx]}")
        print("-" * 60)

    results["examples"] = examples

    # -------------------------------------------------
    # 7. Save results
    # -------------------------------------------------
    json_path = RESULTS_DIR / f"test_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_path = RESULTS_DIR / f"test_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 6 — TEST SET EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp     : {timestamp}\n")
        f.write(f"Adapter       : {ADAPTER_PATH}\n")
        f.write(f"Test size     : {len(test_dataset)}\n")
        f.write(f"BLEU          : {results['bleu']}\n")
        f.write(f"chrF          : {results['chrf']}\n\n")
        f.write("Qualitative examples are in the JSON file.\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Results saved to:\n  {json_path}\n  {summary_path}")
    logger.info("Phase 6 completed successfully.")

if __name__ == "__main__":
    main()