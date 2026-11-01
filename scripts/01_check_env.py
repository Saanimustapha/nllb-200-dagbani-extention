import torch
import transformers
import peft
import datasets
import sacrebleu

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"PyTorch          : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"VRAM             : {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
print(f"Transformers     : {transformers.__version__}")
print(f"PEFT             : {peft.__version__}")
print(f"Datasets         : {datasets.__version__}")
print(f"SacreBLEU        : {sacrebleu.__version__}")
print("=" * 60)
print("Environment is ready.")