import os
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft import PeftModel
import numpy as np
import evaluate
import logging
from datetime import datetime
import time
import json

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models/nllb_dag_resized"          
DATA_PATH = "data/dag_en_hf"
OUTPUT_DIR = Path("checkpoints/nllb-dag-lora")
LOGGING_DIR = Path("logs")
RESULTS_DIR = Path("results")
TIME_TRACKER_FILE = Path("results/total_training_time.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGGING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters (tuned for 6 GB)
MAX_LENGTH = 96
BATCH_SIZE = 2
GRAD_ACCUM = 16                    # effective batch size = 32
LEARNING_RATE = 1.5e-4
NUM_EPOCHS = 5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
SAVE_STEPS = 400
EVAL_STEPS = 400
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGGING_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_lora")

def load_total_time():
    if TIME_TRACKER_FILE.exists():
        with open(TIME_TRACKER_FILE, "r") as f:
            data = json.load(f)
            return data.get("total_seconds", 0.0)
    return 0.0

def save_total_time(total_seconds: float):
    TIME_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TIME_TRACKER_FILE, "w") as f:
        json.dump({
            "total_seconds": total_seconds,
            "total_hours": round(total_seconds / 3600, 2),
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)

def main():
    logger.info("=" * 70)
    logger.info("PHASE 5 — LoRA Fine-tuning")
    logger.info("=" * 70)

    # -------------------------------------------------
    # 1. Load tokenizer and model
    # -------------------------------------------------
    logger.info(f"Loading tokenizer and model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(model)

    # -------------------------------------------------
    # 2. Apply LoRA
    # -------------------------------------------------
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------
    # 3. Load dataset
    # -------------------------------------------------
    logger.info("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)

    def preprocess_function(examples):
        # Dagbani → English direction
        inputs = [f"dag_Latn {text}" for text in examples["dag"]]
        targets = [f"eng_Latn {text}" for text in examples["en"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing dataset (this may take a few minutes)...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # -------------------------------------------------
    # 4. Metrics
    # -------------------------------------------------
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple cleaning
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": bleu["score"],
            "chrf": chrf_score["score"],
        }

    # -------------------------------------------------
    # 5. Training Arguments (with resume support)
    # -------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,          
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(LOGGING_DIR),
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=50,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",          # memory efficient
        report_to="none",
        gradient_checkpointing=True,       # critical for 6 GB
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,          # safer on Windows
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -------------------------------------------------
    # 6. Train (with automatic resume)
    # -------------------------------------------------
    logger.info("Starting training...")
    logger.info(f"Checkpoints will be saved every {SAVE_STEPS} steps to: {OUTPUT_DIR}")
    logger.info("You can stop at any time with Ctrl+C and resume later.")

    # --- Time tracking ---
    previous_total = load_total_time()
    session_start = time.time()
    logger.info(f"Previous accumulated training time: {previous_total/3600:.2f} hours")
                
    try:
        trainer.train(resume_from_checkpoint=True if any(OUTPUT_DIR.glob("checkpoint-*")) else None)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
    finally:
        session_duration = time.time() - session_start
        new_total = previous_total + session_duration
        save_total_time(new_total)

    logger.info("=" * 70)
    logger.info("TRAINING TIME SUMMARY")
    logger.info(f"This session      : {session_duration/3600:.2f} hours")
    logger.info(f"Total accumulated : {new_total/3600:.2f} hours")
    logger.info("=" * 70)

    # -------------------------------------------------
    # 7. Save final adapter
    # -------------------------------------------------
    final_path = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Final LoRA adapter saved to: {final_path}")

    logger.info("Phase 5 completed successfully.")

if __name__ == "__main__":
    main()