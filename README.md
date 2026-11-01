# NLLB-Dagbani: Adapting NLLB-200 to a Low-Resource Ghanaian Language

Parameter-efficient fine-tuning of Meta’s **NLLB-200** (distilled 600M) for **Dagbani ↔ English** machine translation.

This project demonstrates a practical, reproducible pipeline for extending a massively multilingual translation model to a previously unsupported low-resource language under tight compute constraints (NVIDIA RTX 3060 6 GB).

---

## Motivation

Dagbani (also spelled Dabgani) is a Gur language spoken primarily in northern Ghana. It is severely under-represented in existing multilingual MT systems, including NLLB-200.  

The goal of this work is to:

- Add first-class support for Dagbani to NLLB-200
- Do so with limited parallel data (~100k sentence pairs)
- Remain feasible on consumer hardware (6 GB VRAM)
- Combine vocabulary extension with cross-lingual inductive bias from a related language (Mossi)

---

## Approach Overview

We use a **hybrid strategy**:

1. **New language tokens** for Dagbani (`dag_Latn`) + targeted vocabulary extension
2. **Weak cross-lingual inductive bias** from Mossi (`mos_Latn`), the closest available Oti-Volta language in FLORES-200 / NLLB
3. **Parameter-efficient fine-tuning** with LoRA + 4-bit quantization

This avoids both pure random initialization of new embeddings and aggressive bootstrapping from a weakly related language.

### Key Technical Components

| Component                    | Details                                      |
|-----------------------------|----------------------------------------------|
| Base model                  | `facebook/nllb-200-distilled-600M`           |
| Vocabulary extension        | +3,000 high-frequency Dagbani subwords       |
| Fertility improvement       | 1.696 → 1.237 tokens/word                    |
| Embedding initialization    | Subword average + small Mossi bias (α=0.12)  |
| Fine-tuning method          | LoRA (r=32, α=64) + 4-bit NF4                |
| Hardware                    | NVIDIA RTX 3060 6 GB                         |
| Training data               | ~101.7k Dagbani–English pairs                |

---

## Data

- Source: [ghananlpcommunity/ghana-corpus](https://huggingface.co/datasets/ghananlpcommunity/ghana-corpus) (via Ghana Corpus Builder)
- After cleaning & deduplication: **101,676** training pairs  
- Validation / Test: 5,649 pairs each  
- Domain: primarily religious / Bible parallel text (multiple English renderings kept as separate examples)

---

## Pipeline

The project is organized into clear phases:

| Phase | Description                                      | Script                          |
|-------|--------------------------------------------------|---------------------------------|
| 1     | Environment setup & GPU verification             | `scripts/01_check_env.py`       |
| 2     | Data loading, cleaning & HF Dataset creation     | `scripts/02_prepare_data.py`    |
| 3     | Fertility measurement + tokenizer extension      | `scripts/03_extend_tokenizer.py`|
| 4     | Model resize + smart embedding initialization    | `scripts/04_resize_and_init_embeddings.py` |
| 5     | LoRA fine-tuning (resumable)                     | `scripts/05_train_lora.py`      |
| 6     | Test-set evaluation + qualitative examples       | `scripts/06_evaluate.py`        |

All intermediate results (fertility reports, resize info, evaluation metrics) are saved under `results/`.

---

## Results (Vocabulary Extension)

| Metric                        | Value      |
|------------------------------|------------|
| Original Dagbani fertility   | 1.696      |
| Extended Dagbani fertility   | **1.237**  |
| Improvement                  | −0.459     |
| New tokens added             | 3,000      |
| Final vocabulary size        | 259,205    |

Top newly added tokens include high-frequency Dagbani items such as `▁zuɣu`, `▁zaŋ`, `▁ŋun`, `▁Yawɛ`, `▁Naawuni`, etc.

After testing the final model with the test set, the evaluation metrics obtained are as follows:

| Metric                        | Value      |
|------------------------------|------------|
| BLEU                          | **11.97**      |
| chrF                            | **31.63**  |

---

## Hardware & Efficiency

Training was performed entirely on a single **RTX 3060 (6 GB)** using:

- 4-bit NF4 quantization (`bitsandbytes`)
- LoRA (trainable parameters ≈ 2.7%)
- Gradient checkpointing
- Effective batch size 32 (batch size 2 × gradient accumulation 16)
- Max sequence length 96
- Full checkpointing + resume support (training can be stopped and continued at any time)

---

## Project Structure

```text
nllb-dag/
├── data/                       # Prepared HF datasets + clean CSVs
├── models/
│   ├── nllb-200-distilled-600M/   # Local base model
│   ├── tokenizer/                 # Extended tokenizer
│   └── nllb_dag_resized/          # Resized model with new embeddings
├── checkpoints/nllb-dag-lora/     # final adapter
├── results/                       # Fertility, resize, evaluation reports
├── scripts/                       # All pipeline scripts
├── requirements.txt
└── README.md
```

---

## Quick Start (Inference)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForSeq2SeqLM.from_pretrained(
    "path/to/nllb_dag_resized",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base, "path/to/final_adapter")
tokenizer = AutoTokenizer.from_pretrained("path/to/nllb_dag_resized")

def translate(text: str) -> str:
    inputs = tokenizer(f"dag_Latn {text}", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=96, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Limitations

- Training data is heavily skewed toward religious text.
- Only the Dagbani → English direction was the primary focus of the reported experiments.
- Mossi provides only a weak inductive bias (no strong sister language exists in NLLB-200).
- Evaluation on broader domains remains future work.

---

## Acknowledgments

- Meta AI for releasing NLLB-200
- GhanaNLP Community for the Ghana Corpus
- Hugging Face transformers, peft, and bitsandbytes ecosystems
